# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The phase-1 layer kernel (``upto = "norm"``) on 8 GPUs against the torch
reference (``minimax_m3_pro_mono_ref``): one process a rank, each with its own
core rows and o_proj weight, all with the same residual and norm weights.

The inputs (``case_inputs``): one size a token over 6 decades, the same on
every rank, and each residual row 1, 0.1 or 10 times its all-reduced row's norm,
so no rank's partial and no residual is drowned by another row or input.

What must hold:
* every rank writes the same bytes (they read the same gathered sums and sums of
  squares) -- bit for bit;
* the lookahead written into a ring slot's rows ``[S, 2S)`` (where
  ``ProMonoRunner.launch`` points it: ``ProLookaheadState.lookahead_rows``) is
  the bytes the same launch writes to its own tensors, and nothing else in the
  ring changes;
* (x8, x8s) is MXFP8 of the stored x, bit for bit; la8 is the lookahead norm of
  the stored h_mid within one e4m3 step;
* residual_mid, x, x8 and la8 match the reference row by row (``BOUNDS``: the
  largest per-row relative error norm), at least ``H_MID_EXACT_FLOOR`` of h_mid
  bit for bit. Not all bit for bit: the kernel's o_proj sums K in another order
  than the fp64 reference, so a partial can differ in its last bf16 bit. The
  bounds pass every partial 1 ulp off, and the reference's CPU test
  (``test_minimax_m3_pro_mono_layout``) shows they catch a dropped rank, token,
  K block or scale column and a missing, shifted or wrong-token residual.

The build is the debug one: a wait that never lands gives up after 10 s and is
reported by name instead of hanging the GPUs.

Needs 8 GPUs of 256 CUs (MI355X), aiter and FlyDSL; skipped otherwise.
"""

import functools
import os
import socket

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("flydsl")
if torch.cuda.device_count() < 8:
    pytest.skip("needs 8 GPUs", allow_module_level=True)

from minimax_m3_pro_mono_ref import (
    case_inputs,
    layer_failures,
    layer_reference,
    layer_report,
    mxfp8,
)

TP = 8
HIDDEN = 7168
O_K = 3072
LAYER = 4
EPS = 1e-6
# (rows, lookahead): one pass, a partial wave, two passes, 16 rows; then the
# second token tile and AR #1's third pass: one wave's third token (17), half
# the waves' (20), the widest norm build (24)
CASES = (
    (1, True), (3, True), (8, True), (9, True), (16, True), (16, False),
    (17, True), (20, True), (24, True), (24, False),
)  # fmt: skip
TIMEOUT_S = 900


@functools.cache
def _weight(rank):
    """This rank's o_proj columns as the checkpoint holds them: MXFP8 1x32
    (e4m3 [7168, 3072], E8M0 [7168, 96])."""
    g = torch.Generator().manual_seed(7000 + rank)
    w = torch.randn(HIDDEN, O_K, generator=g) * 0.02
    # a few rows 2^6 louder: their scale codes differ from the rest
    w[torch.randint(0, HIDDEN, (64,), generator=g)] *= 64.0
    return mxfp8(w.to(torch.bfloat16))


def _norm_weights():
    g = torch.Generator().manual_seed(17)
    w_post = (torch.randn(HIDDEN, generator=g) * 0.3).to(torch.bfloat16)
    w_ahead = (torch.randn(HIDDEN, generator=g) * 0.3).to(torch.bfloat16)
    return w_post, w_ahead


@functools.cache
def _inputs(case):
    """Case ``case``'s (cores of every rank, o partials of every rank, residual)."""
    s = CASES[case][0]
    return case_inputs(s, [_weight(r) for r in range(TP)], 1000 * (case + 1))


def _worker(rank, port, queue, done, inputs):
    """``inputs``: per case, (this rank's core rows, the residual). The worker stays
    alive until the parent has received every rank's results (``done``): torch hands
    tensors over the queue as file descriptors that the sender must still hold."""
    import torch.distributed as dist
    from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
    from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale, shuffle_weight

    from atom.models.minimax_m3.pro_mono import layout as L
    from atom.models.minimax_m3.pro_mono.kernels.layer import ABI, build
    from atom.mono.runtime.compile import compile_kernels
    from atom.mono.runtime.debug import DIAG_BYTES, given_up_waits, region_namer
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
        keys = [L.P1Build(tokens=s, lookahead=la, debug=True) for s, la in CASES]
        launches = compile_kernels([(build(k), ABI) for k in keys])
        peers = PeerBuffer(L.peer_bytes(), group, rank, TP, dev)
        scratch = uncached_bytes(L.scratch_bytes(keys[0]), dev)
        mbx = StepMailboxes(peers, scratch, debug=True)
        diag = scratch[: L.scratch_layout(keys[0])["diag"][1]]
        assert diag.numel() == DIAG_BYTES
        wq, ws = _weight(rank)
        # the native MXFP8 loader's own transforms (``linear.py``): the weight
        # (16, 16)-shuffled, the scale as the bpreshuffle GEMM reads it
        w_o = shuffle_weight(wq.view(torch.uint8).to(dev), layout=(16, 16))
        s_o = shuffle_mxfp8_bpreshuffle_scale(ws.to(dev)).contiguous()
        results = []
        w_post, w_ahead = (t.to(dev) for t in _norm_weights())

        def step(launch, core, res, out):
            args = ABI.pack(
                {
                    "core": core.data_ptr(), "w_o": w_o.data_ptr(),
                    "s_o": s_o.data_ptr(), "res": res.data_ptr(),
                    "g_post": w_post.data_ptr(), "g_ahead": w_ahead.data_ptr(),
                    "moe_w": 0, "ffn_out": 0, "plain": 0, "nvalid": 0,
                    **{k: v.data_ptr() for k, v in out.items()},
                    "scratch": scratch.data_ptr(),
                    "peers": peers.addresses.data_ptr(), "rank": rank,
                    "layer": LAYER, "tl": 0,
                }
            )  # fmt: skip
            dist.barrier(group=group)
            mbx.begin_step()
            _run_compiled(launch, *args, torch.cuda.current_stream())
            torch.cuda.synchronize()
            return given_up_waits(diag, region_namer(L.REGIONS)) + [
                f"fence: rank {r}" for r in mbx.missing_ranks()
            ]

        for key, launch, (core, res) in zip(keys, launches, inputs):
            s = key.tokens
            core, res = core.to(dev), res.to(dev)
            out = {
                "h_mid": torch.zeros(s, HIDDEN, dtype=torch.bfloat16, device=dev),
                "x": torch.zeros(s, HIDDEN, dtype=torch.bfloat16, device=dev),
                "x8": torch.zeros(s, HIDDEN, dtype=torch.uint8, device=dev),
                "x8s": torch.zeros(s, HIDDEN // 32, dtype=torch.uint8, device=dev),
                "la8": torch.zeros(s, HIDDEN, dtype=torch.uint8, device=dev),
                "la8s": torch.zeros(s, HIDDEN // 32, dtype=torch.uint8, device=dev),
            }

            waits = step(launch, core, res, out)
            got = {k: v.cpu() for k, v in out.items()}
            if key.lookahead:
                # again with the lookahead into slot 1's rows [s, 2s) of a ring
                # (ProLookaheadState's: 3 slots of 2 x 16 rows), filled around them
                ring = (
                    torch.full((3, 32, HIDDEN), 0x55, dtype=torch.uint8, device=dev),
                    torch.full((3, 32, HIDDEN // 32), 0x66, dtype=torch.uint8, device=dev),
                )  # fmt: skip
                waits += step(
                    launch,
                    core,
                    res,
                    {
                        **out,
                        "la8": ring[0][1, s : 2 * s],
                        "la8s": ring[1][1, s : 2 * s],
                    },
                )
                got["ring_q"], got["ring_s"] = ring[0].cpu(), ring[1].cpu()
            got["x8"] = got["x8"].view(torch.float8_e4m3fn)
            got["la8"] = got["la8"].view(torch.float8_e4m3fn)
            if not key.lookahead:
                del got["la8"], got["la8s"]
            results.append((key.tokens, key.lookahead, waits, got))
        queue.put((rank, results, None))
        done.wait(TIMEOUT_S)
    except Exception as e:  # noqa: BLE001 - reported to the parent by name
        import traceback

        queue.put((rank, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def ranks():
    """Every rank's outputs of every case: {rank: [(rows, lookahead, waits,
    outputs)]}."""
    import torch.multiprocessing as mp

    os.environ.setdefault("ARCH", "gfx950")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    done = ctx.Event()
    port = _free_port()
    ins = [_inputs(i) for i in range(len(CASES))]
    procs = [
        ctx.Process(
            target=_worker,
            args=(r, port, queue, done, [(cores[r], res) for cores, _, res in ins]),
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
        done.set()
        for p in procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
    assert not errors, "\n".join(errors)
    return got


def test_no_wait_gave_up(ranks):
    stuck = [
        (rank, s, la, w)
        for rank, results in ranks.items()
        for s, la, w, _ in results
        if w
    ]
    assert not stuck


def test_every_rank_writes_the_same_bytes(ranks):
    for i, (s, la, _, ref0) in enumerate(ranks[0]):
        for rank in range(1, TP):
            got = ranks[rank][i][3]
            for k, v in ref0.items():
                assert torch.equal(
                    v.view(torch.uint8), got[k].view(torch.uint8)
                ), f"rows {s} lookahead {la}: rank {rank}'s {k} differs from rank 0's"


def test_the_lookahead_into_ring_rows_is_the_own_tensors_bytes(ranks):
    for rank, results in ranks.items():
        for s, la, _, got in results:
            if not la:
                continue
            want_q = torch.full((3, 32, HIDDEN), 0x55, dtype=torch.uint8)
            want_s = torch.full((3, 32, HIDDEN // 32), 0x66, dtype=torch.uint8)
            want_q[1, s : 2 * s] = got["la8"].view(torch.uint8)
            want_s[1, s : 2 * s] = got["la8s"]
            assert torch.equal(got["ring_q"], want_q), f"rank {rank} rows {s}: la8"
            assert torch.equal(got["ring_s"], want_s), f"rank {rank} rows {s}: la8s"


@pytest.mark.parametrize("case", range(len(CASES)))
def test_matches_the_reference(ranks, case):
    s, la, _, got = ranks[0][case]
    _, parts, res = _inputs(case)
    w_post, w_ahead = _norm_weights()
    w_ahead = w_ahead if la else None
    want = layer_reference(parts, res, w_post, w_ahead, EPS, EPS)
    report = layer_report(got, want, w_ahead, EPS)
    print(f"rows {s} lookahead {la}: {report}")
    assert not layer_failures(report), (layer_failures(report), report)
