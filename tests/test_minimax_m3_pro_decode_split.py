# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's decode indexer split (``index_split_decode``) at any sequence count.

Each of the TP ranks scores and selects ceil(S / TP) whole sequences; the last shares
run short or empty and are padded to the same rows for the all-gather, whose rows past
the graph's are cut. The all-gather here is a stand-in that records each rank's share and
hands back the ranks' shares in rank order, with the padded rows set to a marker, so a
padded row that reached a real one would show.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

pytest.importorskip("triton")
pytest.importorskip("aiter", reason="pro_attn imports the ATOM attention base")

from dataclasses import replace
from types import SimpleNamespace

from test_minimax_m3_pro_index_fused import DEV, _batch, _mini_impl

from atom.model_ops.minimax_m3 import pro_kernels as K
from atom.model_ops.minimax_m3.pro_attn import ProSparseAttentionImpl, ProVariants

TP, K_SEL, MARK = 8, 8, -7
COUNTS = [*range(1, 8), *range(9, 16), *range(18, 31)]


@pytest.fixture
def tp(monkeypatch):
    """Eight TP ranks; ``tp.rank`` is the rank ``decode_split`` sees."""
    ps = SimpleNamespace(get_tensor_model_parallel_world_size=lambda: TP)
    ps.get_tensor_model_parallel_rank = lambda: ps.rank
    ps.rank = 0
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", ps)
    comm = SimpleNamespace(tensor_model_parallel_all_gather=None)
    monkeypatch.setitem(sys.modules, "aiter.dist.communication_op", comm)
    ps.comm = comm
    return ps


def _lens(s, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(300, 2600, (s,), generator=g).tolist()


def _v(**kw):
    return ProVariants(
        kernels=True, attn="aiter", topk="split", score_tq=8, index_split_decode=1, **kw
    )


def replace_v(**kw):
    return replace(_v(), **kw)


def _gathered(shares, n):
    """The ranks' shares in rank order, their rows past ``n`` set to ``MARK``."""
    full = torch.cat(shares).clone()
    full[n:] = MARK
    return full


def _split_run(tp, impl, args, v, n, s, run=None):
    """Every rank's split forward through the recording all-gather, then rank 0's over
    the ranks' gathered shares. Returns the shares and rank 0's second run."""
    run = run or (lambda ds: impl._kernel_attention(*args, v, None, True, dsplit=ds))
    shares = []
    tp.comm.tensor_model_parallel_all_gather = lambda x, dim: (
        shares.append(x.clone()),
        x.repeat(TP, 1),
    )[1]
    plans = []
    for r in range(TP):
        tp.rank = r
        plans.append(impl.decode_split(n, _v(), s))
        run(plans[-1])
    tp.comm.tensor_model_parallel_all_gather = lambda x, dim: _gathered(shares, n)
    tp.rank = 0
    return shares, plans, run(plans[0])


def test_shares_are_whole_sequences_padded_to_one_size(tp):
    """For every count up to 30 and 1, 4, 6 or 8 rows a sequence: each rank's share is
    ceil(S / 8) sequences; the real rows it selects are its own share's head, whole
    sequences, and together cover every row once; a rank past the last sequence
    selects that sequence again, gathered past the graph's rows."""
    impl = object.__new__(ProSparseAttentionImpl)
    for rps in (1, 4, 6, 8):
        for s in range(1, 31):
            n = s * rps
            share = -(-s // TP) * rps
            covered = []
            for r in range(TP):
                tp.rank = r
                first, end, sh = impl.decode_split(n, _v(), s)
                assert sh == share and first % rps == 0 and end % rps == 0
                if r * share < n:
                    assert (first, end) == (r * share, min(r * share + share, n))
                    covered += range(first, end)
                else:
                    assert (first, end) == (n - rps, n)
            assert covered == list(range(n)), (s, rps)


def test_split_declines(tp):
    impl = object.__new__(ProSparseAttentionImpl)
    assert impl.decode_split(60, _v(), 7) is None  # rows not whole sequences
    assert impl.decode_split(36, replace_v(index_split_decode=64), 6) is None
    tp.get_tensor_model_parallel_world_size = lambda: 1
    assert impl.decode_split(36, _v(), 6) is None


@pytest.mark.parametrize(
    "s,rps",
    [(s, 6) for s in COUNTS] + [(s, r) for s in (10, 18, 20, 22) for r in (4, 8)],
)
def test_split_picks_equal_one_rank(tp, monkeypatch, s, rps):
    """At sequence counts that are not a multiple of 8, and 4, 6 or 8 rows a sequence
    (DSpark k = 3, 5, 7): every rank gathers the page tables and lengths one rank
    selecting all rows gives, bit for bit, from shares of one size."""
    b = _batch(_lens(s, s * rps), rps, seed=s + rps)
    impl, seen, args = _mini_impl(b, K_SEL, monkeypatch)
    impl._kernel_attention(*args, _v(), None, static=True)
    want_rows, want_ctx = seen.pop()
    shares, plans, _ = _split_run(tp, impl, args, _v(), b.n, s)
    share = -(-s // TP) * rps
    assert [tuple(x.shape) for x in shares] == [(share, K_SEL + 1)] * TP
    assert all(p[2] == share for p in plans)
    rows, ctx = seen[-1]
    assert torch.equal(rows, want_rows) and torch.equal(ctx, want_ctx)


@pytest.mark.parametrize("s", [3, 11, 21])
def test_split_block_ids_equal_one_rank(tp, monkeypatch, s):
    """The block-id branch (``select_logical``, for the day0 decode / Triton attention):
    the gathered ids of real rows equal one rank's, from shares of one size."""
    rps = 6
    b = _batch(_lens(s, s), rps, seed=s)
    impl, _, args = _mini_impl(b, K_SEL, monkeypatch)
    sels = []
    monkeypatch.setattr(
        K,
        "attention",
        lambda q, kc4, bt, seq, pos, sel, *a, **kw: sels.append(sel.clone()),
    )
    tv = replace_v(attn="triton")
    impl._kernel_attention(*args, tv, None, static=True)
    want = sels.pop()
    run = lambda ds: impl._kernel_attention(*args, tv, None, True, dsplit=ds)
    shares, _, _ = _split_run(tp, impl, args, tv, b.n, s, run)
    assert {tuple(x.shape) for x in shares} == {(-(-s // TP) * rps, K_SEL)}
    assert torch.equal(sels[-1], want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
@pytest.mark.parametrize("rank", [4, 5, 7])  # a full share, a short one, an empty one
def test_split_graph_replays_another_batch(tp, monkeypatch, rank):
    """Rank ``rank``'s split captured in a graph at 11 sequences of 6 rows (shares of 2
    sequences: ranks 0-4 full, 5 short, 6-7 empty), replayed on other contexts and
    positions: the gathered page tables and lengths equal one rank's on the new batch.
    """
    s, rps, width = 11, 6, 200
    b = _batch(_lens(s, 1), rps, seed=1, width=width)
    impl, seen, args = _mini_impl(b, K_SEL, monkeypatch)
    v = _v()
    tp.rank = rank
    plan = impl.decode_split(b.n, v, s)
    others = [torch.zeros(plan[2], K_SEL + 1, dtype=torch.int32, device=DEV)] * TP
    others = [o.clone() for o in others]

    def gather(x, dim):
        return torch.cat([x if r == rank else others[r] for r in range(TP)])

    tp.comm.tensor_model_parallel_all_gather = gather
    run = lambda: impl._kernel_attention(*args, v, None, True, dsplit=plan)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(side)
    seen.clear()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()
    rows, ctx = seen.pop()
    # another batch in the same buffers: other contexts, positions and block tables
    # (shorter contexts, so the new block tables fit the cache)
    lens = [n - d % 280 for n, d in zip(_lens(s, 1), _lens(s, 2))]
    nb = _batch(lens, rps, seed=2, width=width)
    for dst, src in (
        (b.pos, nb.pos),
        (b.iq, nb.iq),
        (b.bt, nb.bt),
        (args[6][: nb.ic.shape[0]], nb.ic.to(args[6].dtype)),
    ):
        dst.copy_(src)
    shares, _, _ = _split_run(tp, impl, args, v, b.n, s)
    for r in range(TP):
        others[r].copy_(shares[r])
    tp.comm.tensor_model_parallel_all_gather = lambda x, dim: _gathered(shares, b.n)
    impl._kernel_attention(*args, v, None, static=True)
    want_rows, want_ctx = seen.pop()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(rows, want_rows) and torch.equal(ctx, want_ctx)
    assert int((want_ctx > 0).sum()) == b.n
