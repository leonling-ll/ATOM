# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""ATOM's DCP push collectives (``atom.model_ops.dcp_push``, the ``dcp_push`` variant).

On one GPU, eight plain buffers stand in for eight ranks' (every push launched, then every
receive): the gathers copy the bits of a concatenation and the merge adds what
``cp_lse_a2a``'s combine adds, bit for bit, empty ranks (LSE -inf, output NaN) included.

On eight GPUs, real peer memory, captured and replayed as a decode step runs them (keys,
q, merge, three layers a graph, fresh inputs each replay): the bits of the custom
all-gathers and of the all-to-all merge."""

import os
import socket
from types import SimpleNamespace

import pytest
import torch

W, HEADS, QDIM, VDIM, TOPK = 8, 16, 256, 192, 128
ROWS = (1, 37, 384)

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _bits(t):
    return t.contiguous().view(torch.uint8)


def _merge_inputs(rows, seed, device):
    g = torch.Generator(device=device).manual_seed(seed)
    of = torch.randn(rows, W * HEADS * VDIM, device=device, generator=g)
    lse = torch.randn(rows, W * HEADS, device=device, generator=g) * 4
    empty = torch.rand(rows, 1, device=device, generator=g) < 0.2
    if rows > 2:
        empty[2] = True
    lse = torch.where(empty, float("-inf"), lse)
    of = torch.where(empty, float("nan"), of).to(torch.bfloat16)
    return of, lse


def _a2a_combine(dcp_ops, ofs, lses, r):
    """cp_lse_a2a's pack of every rank and its combine of the rows the all-to-all hands
    rank r (three heads of 64 a head, the LSE repeated, as ``_dcp_merge`` calls it)."""
    import triton

    rows, parts = ofs[0].shape[0], VDIM // 64
    pack = dcp_ops._lse_pack_slots(torch.bfloat16)
    heads, local = W * HEADS * parts, HEADS * parts
    recv = torch.empty(
        W, rows, local, 64 + pack, dtype=torch.bfloat16, device=ofs[0].device
    )
    for s in range(W):
        o = ofs[s].view(rows, heads, 64)
        lse = lses[s].repeat_interleave(parts, dim=1)
        send = torch.empty_like(recv)
        dcp_ops._dcp_a2a_pack_kernel[(rows, W)](
            o, lse, lse, send, o.stride(0), o.stride(1), lse.stride(0), lse.stride(1),
            send.stride(0), send.stride(1), send.stride(2), H_LOCAL=local,
            H_ROUNDED=triton.next_power_of_2(local), HEAD_DIM=64, LSE_PACK=pack,
            HAS_OWNED_COUNTS=False,
        )  # fmt: skip
        recv[s] = send[r]
    out = torch.empty(rows, local, 64, dtype=torch.bfloat16, device=recv.device)
    dcp_ops._dcp_a2a_unpack_combine_kernel[(rows, local)](
        recv, out, recv, recv.stride(0), recv.stride(1), recv.stride(2), out.stride(0),
        out.stride(1), 0, 0, W, HEAD_DIM=64, LSE_PACK=pack, N_ROUNDED=W, WRITE_LSE=False,
    )  # fmt: skip
    return out.view(rows, -1)


@needs_gpu
def test_push_gathers_copy_the_bits_on_one_gpu():
    from atom.model_ops.dcp_push import PushComm, PushGather

    dev = torch.device("cuda", 0)
    for width, dtype, dim in (
        (HEADS * QDIM, torch.bfloat16, -1),
        (HEADS * QDIM, torch.float8_e4m3fn, -1),
        (TOPK, torch.int64, 0),
    ):
        row = width * torch.tensor([], dtype=dtype).element_size()
        peers = PushComm.test_peers(W, max(ROWS) * W * row // 4, dev)
        comms = [
            PushGather(None, max(ROWS), row, peers=peers, rank=r, device=dev)
            for r in range(W)
        ]
        for rows in ROWS:
            xs = [
                torch.randint(-(2**31), 2**31 - 1, (rows, row // 4), device=dev, dtype=torch.int32).view(dtype)
                for _ in range(W)
            ]  # fmt: skip
            for r in range(W):
                comms[r](xs[r], dim, phases=1)
            outs = [comms[r](xs[r], dim, phases=2) for r in range(W)]
            want = _bits(torch.cat(xs, dim=0 if dim == 0 else 1))
            assert all(_bits(o).equal(want) for o in outs), (width, dtype, rows)
        assert all(c.timeouts() == 0 for c in comms)
        assert all(int(c.state[0]) == len(ROWS) for c in comms)


@needs_gpu
def test_push_merge_adds_as_the_all_to_all_combine_on_one_gpu():
    from atom.model_ops import dcp_ops
    from atom.model_ops.dcp_push import PushComm, PushMerge

    dev = torch.device("cuda", 0)
    rmax = max(ROWS)
    peers = PushComm.test_peers(W, W * rmax * HEADS * (VDIM // 2 + 1), dev)
    comms = [
        PushMerge(None, rmax, HEADS, VDIM, peers=peers, rank=r, device=dev)
        for r in range(W)
    ]
    for rows in ROWS:
        ins = [_merge_inputs(rows, 10 * rows + s, dev) for s in range(W)]
        ofs, lses = [i[0] for i in ins], [i[1] for i in ins]
        for rep in (1, VDIM // 64):
            outs = [
                torch.empty(rows, HEADS * VDIM, dtype=torch.bfloat16, device=dev)
                for _ in range(W)
            ]
            lin = [lse.repeat_interleave(rep, dim=1) for lse in lses]
            for r in range(W):
                assert comms[r].fits(ofs[r], lin[r], outs[r])
                comms[r](ofs[r], lin[r], outs[r], phases=1)
            for r in range(W):
                comms[r](ofs[r], lin[r], outs[r], phases=2)
            for r in range(W):
                want = _a2a_combine(dcp_ops, ofs, lses, r)
                assert _bits(outs[r]).equal(_bits(want)), (rows, rep, r)
                assert not torch.isnan(outs[r]).any()
    assert all(c.timeouts() == 0 for c in comms)


def _decode_layers(push, group, keys, q, of, lse, out, layers):
    """A decode step's push collectives in its order, ``layers`` times."""
    res = []
    for _ in range(layers):
        k = push.keys(keys, 0)
        qf = push.q(q, -1)
        res.append((k.clone(), qf.clone(), push.merge(of, lse, out).clone()))
    return res


def _worker(rank, world, port, results):
    import torch.distributed as dist

    from atom.model_ops import dcp_ops
    from atom.model_ops.dcp_push import DcpPush

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
    push = DcpPush(group, max(ROWS), HEADS, QDIM, VDIM, TOPK)
    for rows in ROWS:
        keys = torch.empty(rows, TOPK, dtype=torch.int64, device="cuda")
        q = torch.empty(rows, HEADS * QDIM, dtype=torch.bfloat16, device="cuda")
        of, lse = _merge_inputs(rows, rank, "cuda")
        out = torch.empty(rows, HEADS * VDIM, dtype=torch.bfloat16, device="cuda")

        def fresh(step, keys=keys, q=q, of=of, lse=lse, rows=rows):
            g = torch.Generator(device="cuda").manual_seed(1000 * step + rank)
            keys.copy_(
                torch.randint(0, 1 << 40, keys.shape, device="cuda", generator=g)
            )
            q.copy_(torch.randn(q.shape, device="cuda", generator=g))
            o, s = _merge_inputs(rows, 7 * step + rank, "cuda")
            of.copy_(o)
            lse.copy_(s)

        fresh(0)
        _decode_layers(push, group, keys, q, of, lse, out, 1)  # eager, compiles
        graph, side = torch.cuda.CUDAGraph(), torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.cuda.graph(graph, stream=side):
            got = _decode_layers(push, group, keys, q, of, lse, out, 3)
        torch.cuda.current_stream().wait_stream(side)
        for step in range(1, 4):
            fresh(step)
            graph.replay()
            torch.cuda.synchronize()
            want_k = torch.empty(world * rows, TOPK, dtype=keys.dtype, device="cuda")
            dist.all_gather_into_tensor(want_k, keys)
            want_q = torch.empty(
                world, rows, HEADS * QDIM, dtype=q.dtype, device="cuda"
            )
            dist.all_gather_into_tensor(want_q.view(-1, HEADS * QDIM), q)
            want_q = want_q.permute(1, 0, 2).reshape(rows, -1)
            merged = dcp_ops.dcp_lse_merge(
                of.view(rows, -1, 64), lse.repeat_interleave(3, dim=1), group, "a2a"
            )
            want_m = merged.reshape(rows, -1)
            for k, qf, m in got:
                assert _bits(k).equal(_bits(want_k)), (rows, step, "keys")
                assert _bits(qf).equal(_bits(want_q)), (rows, step, "q")
                assert _bits(m).equal(_bits(want_m)), (rows, step, "merge")
        assert all(c.timeouts() == 0 for c in push.comms())
        graph.reset()
    torch.cuda.synchronize()
    dist.barrier()
    for c in push.comms():
        c.close()
    results[rank] = True
    dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 8,
    reason="needs eight GPUs",
)
def test_push_collectives_on_eight_ranks_match_the_custom_and_all_to_all():
    pytest.importorskip("aiter.ops.flydsl.dcp_mailbox.peer")
    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    results = mp.Manager().dict()
    mp.spawn(_worker, args=(W, port, results), nprocs=W, join=True)
    assert all(results.get(r) for r in range(W))


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    raise SystemExit(pytest.main([__file__, "-q", "-rf"]))
