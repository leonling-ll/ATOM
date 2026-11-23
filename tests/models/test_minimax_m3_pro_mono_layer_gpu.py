# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The phase-1 layer kernel (``upto = "norm"``) on 8 GPUs against the torch
reference (``minimax_m3_pro_mono_ref``): one process a rank, each with its own
core rows and o_proj weight, all with the same residual and norm weights.

What must hold:
* every rank writes the same bytes (they read the same gathered sums and sums of
  squares) -- bit for bit;
* residual_mid, x, x8 and the lookahead's la8 match the reference. Not bit for
  bit: the kernel's o_proj sums K in another order than the fp64 reference, so
  a partial can differ in its last bf16 bit; the bounds are relative norms a
  misread weight, scale or column (errors of order 1) cannot meet, and the
  MXFP8 scale codes within one code.

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
    code_distance,
    dequant,
    layer_reference,
    mxfp8,
    o_partial,
    rows,
)

TP = 8
HIDDEN = 7168
O_K = 3072
LAYER = 4
EPS = 1e-6
# (rows, lookahead): one pass, a partial wave, two passes, the widest build
CASES = ((1, True), (3, True), (8, True), (9, True), (16, True), (16, False))
TIMEOUT_S = 900


def _core(rank, s):
    return rows(s, O_K, 100 * s + rank)


@functools.cache
def _weight(rank):
    """This rank's o_proj columns as the checkpoint holds them: MXFP8 1x32
    (e4m3 [7168, 3072], E8M0 [7168, 96])."""
    g = torch.Generator().manual_seed(7000 + rank)
    w = torch.randn(HIDDEN, O_K, generator=g) * 0.02
    # a few rows 2^6 louder: their scale codes differ from the rest
    w[torch.randint(0, HIDDEN, (64,), generator=g)] *= 64.0
    return mxfp8(w.to(torch.bfloat16))


def _shared(s):
    g = torch.Generator().manual_seed(17)
    w_post = (torch.randn(HIDDEN, generator=g) * 0.3).to(torch.bfloat16)
    w_ahead = (torch.randn(HIDDEN, generator=g) * 0.3).to(torch.bfloat16)
    return rows(s, HIDDEN, 300 + s, scale=4.0), w_post, w_ahead


def _worker(rank, port, queue):
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
        for key, launch in zip(keys, launches):
            s = key.tokens
            res, w_post, w_ahead = (t.to(dev) for t in _shared(s))
            core = _core(rank, s).to(dev)
            out = {
                "h_mid": torch.zeros(s, HIDDEN, dtype=torch.bfloat16, device=dev),
                "x": torch.zeros(s, HIDDEN, dtype=torch.bfloat16, device=dev),
                "x8": torch.zeros(s, HIDDEN, dtype=torch.uint8, device=dev),
                "x8s": torch.zeros(s, HIDDEN // 32, dtype=torch.uint8, device=dev),
                "la8": torch.zeros(s, HIDDEN, dtype=torch.uint8, device=dev),
                "la8s": torch.zeros(s, HIDDEN // 32, dtype=torch.uint8, device=dev),
            }
            args = ABI.pack(
                {
                    "core": core.data_ptr(), "w_o": w_o.data_ptr(),
                    "s_o": s_o.data_ptr(), "res": res.data_ptr(),
                    "g_post": w_post.data_ptr(), "g_ahead": w_ahead.data_ptr(),
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
            waits = given_up_waits(diag, region_namer(L.REGIONS))
            waits += [f"fence: rank {r}" for r in mbx.missing_ranks()]
            got = {k: v.cpu() for k, v in out.items()}
            got["x8"] = got["x8"].view(torch.float8_e4m3fn)
            got["la8"] = got["la8"].view(torch.float8_e4m3fn)
            if not key.lookahead:
                del got["la8"], got["la8s"]
            results.append((key.tokens, key.lookahead, waits, got))
        queue.put((rank, results, None))
    except Exception as e:  # noqa: BLE001 - reported to the parent by name
        import traceback

        queue.put((rank, None, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _relnorm(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@pytest.fixture(scope="module")
def ranks():
    """Every rank's outputs of every case: {rank: [(rows, lookahead, waits,
    outputs)]}."""
    import torch.multiprocessing as mp

    os.environ.setdefault("ARCH", "gfx950")
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, port, queue)) for r in range(TP)]
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


@pytest.mark.parametrize("case", range(len(CASES)))
def test_matches_the_reference(ranks, case):
    s, la, _, got = ranks[0][case]
    res, w_post, w_ahead = _shared(s)
    parts = [o_partial(_core(r, s), *_weight(r)) for r in range(TP)]
    want = layer_reference(parts, res, w_post, w_ahead if la else None, EPS, EPS)
    report = {
        "h_mid_exact": float((got["h_mid"].view(torch.int16) == want["h_mid"].view(torch.int16)).float().mean()),
        "h_mid": _relnorm(got["h_mid"], want["h_mid"]),
        "x": _relnorm(got["x"], want["x"]),
        "x8": _relnorm(dequant(got["x8"], got["x8s"]), want["x"]),
        "x8s_far": int((code_distance(got["x8s"], want["x8s"]) > 1).sum()),
    }  # fmt: skip
    if la:
        la_x = dequant(want["la8"], want["la8s"])
        report["la8"] = _relnorm(dequant(got["la8"], got["la8s"]), la_x)
        report["la8s_far"] = int((code_distance(got["la8s"], want["la8s"]) > 1).sum())
    print(f"rows {s} lookahead {la}: {report}")
    assert report["h_mid"] <= 2.0**-7, report
    assert report["x"] <= 2.0**-6, report
    # MXFP8's own error is 2^-4 of a block's max at worst
    assert report["x8"] <= 2.0**-4, report
    assert report["x8s_far"] == 0, report
    if la:
        assert report["la8"] <= 2.0**-4, report
        assert report["la8s_far"] == 0, report
