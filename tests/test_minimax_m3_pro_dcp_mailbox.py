# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3-Pro's DCP decode merge (``dcp_mailbox``) on real ranks: aiter's mailbox
combine at a rank's 16 heads x 192, captured and replayed as decode is, stored into the
layer's output, merges what the all-to-all merges at three heads of 64 a head (the LSE
repeated), including a row one rank holds no key of (its partial 0, its LSE -inf).

``DCP_MAILBOX_BENCH=1`` (with ``-s``) also prints, per decode width, the microseconds a
merge of each path takes in a graph of 68 (one a layer), the measurement that sets
``dcp_mailbox``: the widest width the mailbox still leads at."""

import os
import socket
from types import SimpleNamespace

import pytest
import torch

HEADS, DIM, LAYERS = 16, 192, 68
WIDTHS = (8, 85, 384)
BENCH_WIDTHS = (16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768)


def _inputs(rows, world, rank, of, lse, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    of.copy_(torch.randn(of.shape, device="cuda", generator=g))
    lse.copy_(torch.randn(lse.shape, device="cuda", generator=g) * 3)
    if rank == 1 and rows > 2:  # rank 1 sees no key of row 2
        of[2] = 0
        lse[2] = float("-inf")


def _all_to_all(dcp_ops, group, of, lse, out):
    """The merge as `ProSparseAttentionImpl._dcp_merge` runs it without the mailbox."""
    rows = out.shape[0]
    merged = dcp_ops.dcp_lse_merge(
        of.view(rows, -1, 64), lse.repeat_interleave(3, dim=1), group, "a2a"
    )
    return out.copy_(merged.reshape(rows, -1))


def _check(dcp_ops, group, world, rank, rows):
    of = torch.empty(rows, world * HEADS * DIM, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(rows, world * HEADS, device="cuda")
    out = torch.zeros(rows, HEADS * DIM, device="cuda", dtype=torch.bfloat16)
    _inputs(rows, world, rank, of, lse, 100 + rank)
    graph, side = torch.cuda.CUDAGraph(), torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.cuda.graph(graph, stream=side):
        for _ in range(3):  # three layers in one graph
            got = dcp_ops.dcp_mailbox_merge(
                of.view(rows, -1, DIM), lse, group, out.view(rows, HEADS, DIM)
            )
            assert got is not None and got.data_ptr() == out.data_ptr()
    torch.cuda.current_stream().wait_stream(side)
    want = torch.empty_like(out)
    for step in range(3):  # fresh inputs, both mailbox parities
        _inputs(rows, world, rank, of, lse, 1000 * step + rank)
        graph.replay()
        _all_to_all(dcp_ops, group, of, lse, want)
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, want, rtol=1e-2, atol=1e-2)


def _bench(dcp_ops, group, world, rank):
    import torch.distributed as dist

    rows_us = []
    for rows in BENCH_WIDTHS:
        of = torch.randn(rows, world * HEADS * DIM, device="cuda").to(torch.bfloat16)
        lse = torch.randn(rows, world * HEADS, device="cuda")
        out = torch.empty(rows, HEADS * DIM, device="cuda", dtype=torch.bfloat16)
        us = []
        for mailbox in (False, True):
            graph, side = torch.cuda.CUDAGraph(), torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.cuda.graph(graph, stream=side):
                for _ in range(LAYERS):
                    if mailbox:
                        dcp_ops.dcp_mailbox_merge(
                            of.view(rows, -1, DIM), lse, group, out.view(rows, -1, DIM)
                        )
                    else:
                        _all_to_all(dcp_ops, group, of, lse, out)
            torch.cuda.current_stream().wait_stream(side)
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(20):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            us.append(start.elapsed_time(end) * 1e3 / (20 * LAYERS))
        rows_us.append((rows, *us))
    if rank == 0:
        print(f"\nPro DCP{world} merge, us a layer (graph of {LAYERS}):")
        print(f"{'rows':>6} {'all-to-all':>11} {'mailbox':>9}")
        for rows, a2a, mbox in rows_us:
            print(f"{rows:>6} {a2a:>11.1f} {mbox:>9.1f}")


def _worker(rank, world, port, bench, results):
    import torch.distributed as dist

    from atom.model_ops import dcp_ops

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world
    )
    group = SimpleNamespace(
        world_size=world,
        rank_in_group=rank,
        cpu_group=dist.new_group(backend="gloo"),
        device_group=dist.group.WORLD,
    )
    widest = max(BENCH_WIDTHS if bench else WIDTHS)
    cap = widest * world * HEADS * DIM
    assert dcp_ops.prepare_dcp_mailbox(group, HEADS, DIM, widest, max_elems=cap)
    for rows in WIDTHS:
        _check(dcp_ops, group, world, rank, rows)
    if bench:
        _bench(dcp_ops, group, world, rank)
    torch.cuda.synchronize()
    dist.barrier()
    results[rank] = True
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="needs two GPUs",
)
def test_the_mailbox_merges_a_pro_decode_as_all_to_all_does():
    pytest.importorskip("aiter.ops.flydsl.dcp_mailbox")
    import torch.multiprocessing as mp

    world = min(
        torch.cuda.device_count(), int(os.environ.get("DCP_MAILBOX_WORLD", "8"))
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    bench = os.environ.get("DCP_MAILBOX_BENCH") == "1"
    results = mp.Manager().dict()
    mp.spawn(_worker, args=(world, port, bench, results), nprocs=world, join=True)
    assert all(results.get(r) for r in range(world))
