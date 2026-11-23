# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The phase-1 layer kernel's ``upto = "ffn"`` build on 8 GPUs: the norm
stages against the torch reference as ``test_minimax_m3_pro_mono_layer_gpu``
holds them, then the router and every rank's FFN partial against
``ffn_reference`` (production's a4w4 MoE and MXFP8 shared expert) run on the
kernel's own x, x8, picks and weights.

Every rank holds its own random routed planes (384 experts, MXFP4, the
intermediate 640 zero padded to 768, shuffled as the loader leaves them) and
shared expert (MXFP8), and the same router; the code ranges keep the gate /
up past the SwiGLU limit now and then and the routed and shared outputs of
one size (``minimax_m3_pro_mono_ref``).

What must hold:
* no wait gives up (the debug build names a wait that never lands);
* every rank writes the same h_mid / x / x8 / la8 and the same route (picks,
  weights), bit for bit; h_mid / x / x8 / la8 match the norm's reference
  (``layer_failures``);
* the gathered logits are within one bf16 step of the reference on the
  kernel's x, the picks the top 4 of its own logits but for ties, the weights
  their renormalization x 2 (``route_failures``);
* each rank's FFN partial within ``FFN_BOUND`` per row (``ffn_failures``: the
  CPU test shows the bound passes summation-order noise and catches a lost
  scale, weight, operand, pick, K block, the clamp, the shared expert).

Needs 8 GPUs of 256 CUs (MI355X), aiter and FlyDSL, about 7 GB a GPU; skipped
otherwise.
"""

import os

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("flydsl")
if torch.cuda.device_count() < 8:
    pytest.skip("needs 8 GPUs", allow_module_level=True)

from minimax_m3_pro_mono_ref import (
    BIAS_STD,
    GATE_STD,
    SHARED_CODES,
    SHARED_STD,
    TOPK,
    W2_CODES,
    W13_CODES,
    case_inputs,
    dequant,
    dq_mxfp4_bytes,
    ffn_failures,
    ffn_reference,
    layer_failures,
    layer_reference,
    layer_report,
    route_failures,
    route_reference,
    row_relnorm,
)
from test_minimax_m3_pro_mono_layer_gpu import (
    EPS,
    HIDDEN,
    LAYER,
    TIMEOUT_S,
    TP,
    _free_port,
    _norm_weights,
    _weight,
)

EXPERTS = 384
INTER_V, INTER_R, INTER_S = 640, 768, 640
# rows: one, a partial wave, an 8-row verify, two passes, the widest
CASES = ((1, True), (3, True), (8, True), (9, True), (16, True), (16, False))


def _inputs(case):
    s = CASES[case][0]
    return case_inputs(s, [_weight(r) for r in range(TP)], 3000 * (case + 1))


def _router():
    g = torch.Generator().manual_seed(23)
    w = (torch.randn(EXPERTS, HIDDEN, generator=g) * GATE_STD).to(torch.bfloat16)
    return w, torch.randn(EXPERTS, generator=g) * BIAS_STD


def _planes(rank, dev):
    """This rank's routed (logical bytes and codes, kernel images) and shared
    (dequantized, kernel images) weights."""
    from aiter.ops.shuffle import (
        shuffle_mxfp8_bpreshuffle_scale,
        shuffle_scale,
        shuffle_weight,
    )

    g = torch.Generator(device=dev).manual_seed(9100 + rank)

    def bytes_(*shape):
        return torch.randint(0, 256, shape, generator=g, dtype=torch.uint8, device=dev)

    def codes(lo_hi, *shape):
        return torch.randint(*lo_hi, shape, generator=g, dtype=torch.uint8, device=dev)

    b13 = bytes_(EXPERTS, 2 * INTER_R, HIDDEN // 2)
    b13.view(EXPERTS, 2, INTER_R, -1)[:, :, INTER_V:] = 0  # the loader's pad
    c13 = codes(W13_CODES, EXPERTS, 2 * INTER_R, HIDDEN // 32)
    b2 = bytes_(EXPERTS, HIDDEN, INTER_R // 2)
    b2[:, :, INTER_V // 2 :] = 0
    c2 = codes(W2_CODES, EXPERTS, HIDDEN, INTER_R // 32)
    routed = {
        "b13": b13, "c13": c13, "b2": b2, "c2": c2,
        "w13": shuffle_weight(b13, layout=(16, 16)).contiguous(),
        "s13": shuffle_scale(c13.view(-1, HIDDEN // 32)).contiguous(),
        "w2": shuffle_weight(b2, layout=(16, 16)).contiguous(),
        "s2": shuffle_scale(c2.view(-1, INTER_R // 32)).contiguous(),
    }  # fmt: skip

    def f8(r, c):
        w = (torch.randn(r, c, generator=g, device=dev) * SHARED_STD).to(
            torch.float8_e4m3fn
        )
        return w, codes(SHARED_CODES, r, c // 32)

    (f13, k13), (f2, k2) = f8(2 * INTER_S, HIDDEN), f8(HIDDEN, INTER_S)
    shared = {
        "deq": (dequant(f13, k13), dequant(f2, k2)),
        "w_sgu": shuffle_weight(f13.view(torch.uint8), layout=(16, 16)),
        "s_sgu": shuffle_mxfp8_bpreshuffle_scale(k13).contiguous(),
        "w_sd": shuffle_weight(f2.view(torch.uint8), layout=(16, 16)),
        "s_sd": shuffle_mxfp8_bpreshuffle_scale(k2).contiguous(),
    }
    return routed, shared


def _worker(rank, port, queue, inputs):
    """``inputs``: per case, (this rank's core rows, the residual)."""
    import torch.distributed as dist
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale, shuffle_weight

    from atom.models.minimax_m3.pro_mono import layout as L
    from atom.models.minimax_m3.pro_mono.kernels.layer import ABI, build
    from atom.mono.runtime.compile import compile_kernels
    from atom.mono.runtime.debug import given_up_waits, region_namer
    from atom.mono.runtime.mailboxes import StepMailboxes
    from atom.mono.runtime.peer_memory import PeerBuffer
    from atom.mono.runtime.uncached import uncached_bytes

    try:
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
        dist.init_process_group(
            "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=TP
        )
        group = dist.group.WORLD
        keys = [
            L.P1Build(tokens=s, upto="ffn", lookahead=la, debug=True) for s, la in CASES
        ]
        launches = compile_kernels([(build(k), ABI) for k in keys])
        peers = PeerBuffer(L.peer_bytes(), group, rank, TP, dev)
        scratch = uncached_bytes(L.scratch_bytes(keys[0]), dev)
        plain = uncached_bytes(L.plain_bytes(keys[0]), dev)
        mbx = StepMailboxes(peers, scratch, debug=True)
        diag = scratch[: L.scratch_layout(keys[0])["diag"][1]]
        route_at = L.scratch_layout(keys[0])["route"][0]
        rlog_at = L.peer_layout()["rlog"][0]
        wq, ws = _weight(rank)
        w_o = shuffle_weight(wq.view(torch.uint8).to(dev), layout=(16, 16))
        s_o = shuffle_mxfp8_bpreshuffle_scale(ws.to(dev)).contiguous()
        w_post, w_ahead = (t.to(dev) for t in _norm_weights())
        w_gate, bias = (t.to(dev) for t in _router())
        routed, shared = _planes(rank, dev)
        band = w_gate[L.E_RANK * rank : L.E_RANK * (rank + 1)]
        table = {
            "w_gate": band, "bias": bias, "w13": routed["w13"], "s13": routed["s13"],
            "w2": routed["w2"], "s2": routed["s2"], "w_sgu": shared["w_sgu"],
            "s_sgu": shared["s_sgu"], "w_sd": shared["w_sd"], "s_sd": shared["s_sd"],
        }  # fmt: skip
        moe_w = torch.tensor(
            [table[n].data_ptr() for n in L.MOE_WEIGHTS], dtype=torch.int64, device=dev
        )
        deq = {}

        def routed_of(e):
            if e not in deq:
                deq[e] = (
                    dq_mxfp4_bytes(routed["b13"][e], routed["c13"][e]),
                    dq_mxfp4_bytes(routed["b2"][e], routed["c2"][e]),
                )
            return deq[e]

        results = []
        for key, launch, (core, res) in zip(keys, launches, inputs):
            s = key.tokens
            core, res = core.to(dev), res.to(dev)

            def z(*shape, dt=torch.uint8):
                return torch.zeros(*shape, dtype=dt, device=dev)

            out = {
                "h_mid": z(s, HIDDEN, dt=torch.bfloat16),
                "x": z(s, HIDDEN, dt=torch.bfloat16),
                "x8": z(s, HIDDEN), "x8s": z(s, HIDDEN // 32),
                "la8": z(s, HIDDEN), "la8s": z(s, HIDDEN // 32),
                "ffn_out": z(s, HIDDEN, dt=torch.bfloat16),
            }  # fmt: skip
            args = ABI.pack(
                {
                    "core": core.data_ptr(), "w_o": w_o.data_ptr(),
                    "s_o": s_o.data_ptr(), "res": res.data_ptr(),
                    "g_post": w_post.data_ptr(), "g_ahead": w_ahead.data_ptr(),
                    "moe_w": moe_w.data_ptr(),
                    **{k: v.data_ptr() for k, v in out.items()},
                    "scratch": scratch.data_ptr(), "plain": plain.data_ptr(),
                    "peers": peers.addresses.data_ptr(), "rank": rank,
                    "layer": LAYER, "tl": 0,
                }
            )  # fmt: skip
            dist.barrier(group=group)
            mbx.begin_step()
            _run_compiled(launch, *args, torch.cuda.current_stream())
            torch.cuda.synchronize()
            waits = given_up_waits(diag, region_namer(L.REGIONS))
            waits += [f"fence: rank {r}" for r in mbx.missing_ranks()]
            # the route record and the gathered logits (tagged pairs: values)
            rec = scratch[route_at : route_at + s * 2 * TOPK * 8].view(torch.int32)
            rec = rec.view(s, 2 * TOPK, 2)[..., 0]
            picks = rec[:, :TOPK].long()
            weights = rec[:, TOPK:].contiguous().view(torch.float32)
            n = TP * L.MAX_ROWS_LAYOUT * L.E_RANK * 8
            rl = peers.bytes[rlog_at : rlog_at + n].view(torch.int32)
            rl = rl.view(TP, L.MAX_ROWS_LAYOUT, L.E_RANK, 2)[:, :s, :, 0]
            logits = (
                rl.contiguous().view(torch.float32).permute(1, 0, 2).reshape(s, -1)
            ).to(torch.bfloat16)
            x8 = out["x8"].view(torch.float8_e4m3fn)
            ref_logits = route_reference(out["x"], w_gate, bias)[0]
            route_bad = route_failures(
                logits.cpu(), picks.cpu(), weights.cpu(), ref_logits.cpu(), bias.cpu()
            )
            want = ffn_reference(
                out["x"], x8, out["x8s"], picks, weights, routed_of,
                shared["deq"],
            )  # fmt: skip
            got = out["ffn_out"]
            norm = {k: v.cpu() for k, v in out.items() if k != "ffn_out"}
            norm["x8"] = norm["x8"].view(torch.float8_e4m3fn)
            norm["la8"] = norm["la8"].view(torch.float8_e4m3fn)
            if not key.lookahead:
                del norm["la8"], norm["la8s"]
            results.append(
                {
                    "rows": s, "lookahead": key.lookahead, "waits": waits,
                    "norm": norm, "picks": picks.cpu(), "weights": weights.cpu(),
                    "route_bad": route_bad,
                    "ffn_bad": ffn_failures(got.cpu(), want.cpu()),
                    "ffn_rel": row_relnorm(got.cpu(), want.cpu()),
                    "union": len(set(picks.flatten().tolist())),
                }
            )  # fmt: skip
        queue.put((rank, results, None))
    except Exception as e:  # noqa: BLE001 - reported to the parent by name
        import traceback

        queue.put((rank, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


@pytest.fixture(scope="module")
def ranks():
    """Every rank's results of every case: {rank: [dict]}."""
    import torch.multiprocessing as mp

    os.environ.setdefault("ARCH", "gfx950")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    ins = [_inputs(i) for i in range(len(CASES))]
    procs = [
        ctx.Process(
            target=_worker,
            args=(r, port, queue, [(cores[r], res) for cores, _, res in ins]),
        )
        for r in range(TP)
    ]
    for p in procs:
        p.start()
    got, errors = {}, []
    try:
        for _ in range(TP):
            rank, results, err = queue.get(timeout=TIMEOUT_S)
            if err:
                errors.append(f"rank {rank}: {err}")
            else:
                got[rank] = results
    finally:
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
    assert not errors, "\n".join(errors)
    return got


def test_no_wait_gave_up(ranks):
    stuck = [
        (rank, r["rows"], r["lookahead"], r["waits"])
        for rank, results in ranks.items()
        for r in results
        if r["waits"]
    ]
    assert not stuck


def test_every_rank_has_the_same_norm_and_route(ranks):
    for i, ref0 in enumerate(ranks[0]):
        for rank in range(1, TP):
            got = ranks[rank][i]
            for k, v in ref0["norm"].items():
                assert torch.equal(
                    v.view(torch.uint8), got["norm"][k].view(torch.uint8)
                ), f"rows {ref0['rows']}: rank {rank}'s {k} differs from rank 0's"
            assert torch.equal(ref0["picks"], got["picks"]), (i, rank)
            assert torch.equal(ref0["weights"], got["weights"]), (i, rank)


@pytest.mark.parametrize("case", range(len(CASES)))
def test_the_norm_matches_the_reference(ranks, case):
    r = ranks[0][case]
    _, parts, res = _inputs(case)
    w_post, w_ahead = _norm_weights()
    w_ahead = w_ahead if r["lookahead"] else None
    want = layer_reference(parts, res, w_post, w_ahead, EPS, EPS)
    report = layer_report(r["norm"], want, w_ahead, EPS)
    assert not layer_failures(report), (layer_failures(report), report)


@pytest.mark.parametrize("case", range(len(CASES)))
def test_the_router_matches_the_reference(ranks, case):
    for rank in range(TP):
        assert not ranks[rank][case]["route_bad"], (rank, ranks[rank][case])


@pytest.mark.parametrize("case", range(len(CASES)))
def test_every_ranks_ffn_partial_matches_the_reference(ranks, case):
    for rank in range(TP):
        r = ranks[rank][case]
        print(
            f"rows {r['rows']} rank {rank}: ffn rel {r['ffn_rel']:.3g},"
            f" |U| {r['union']}"
        )
        assert not r["ffn_bad"], (rank, r["rows"], r["ffn_bad"])
