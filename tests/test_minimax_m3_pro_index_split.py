# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's prefill indexer split over TP ranks (``ProVariants.index_split``).

Each rank scores and selects its ``row_split_index`` rows; the page tables are all-gathered
and put back in row order (``assemble_rows``). Rows are scored independently, so the
result must equal the unsplit selection bit for bit, for prefills that continue a cached
prefix (chunked prefill, prefix caching), end mid-block, or mix with a decode row.
Without a GPU the kernels run in Triton's interpreter.
"""

from __future__ import annotations

import importlib
import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

pytest.importorskip("triton")
pytest.importorskip("aiter", reason="pro_attn imports the ATOM attention base")

from types import SimpleNamespace

from atom.model_ops.minimax_m3 import pro_attn
from atom.model_ops.minimax_m3 import pro_kernels as K
from atom.model_ops.minimax_m3.pro_attn import (
    ProSparseAttentionImpl,
    ProVariants,
    assemble_rows,
    row_split_index,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PG, NI, DI, H, D, VD = 16, 16, 128, 16, 256, 192


@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("n", [1, 15, 16, 17, 1024, 1031])
def test_ranks_cover_every_row_and_assemble_in_order(n, tp):
    parts, seen = [], set()
    for r in range(tp):
        idx, seg = row_split_index(n, tp, r)
        assert idx.shape == (2 * seg,) and int(idx.max()) < n
        seen |= set(idx.tolist())
        parts.append(torch.stack([idx, idx * 10], dim=1))  # a row's payload names it
    assert seen == set(range(n))
    full = assemble_rows(torch.cat(parts), seg, n, tp)
    assert full[:, 0].tolist() == list(range(n))
    assert torch.equal(full[:, 1], full[:, 0] * 10)


def _batch(chunks, seed=0):
    """Rows of a prefill batch: ``chunks`` ``(context length, new tokens)`` per sequence
    (new tokens are the last of the context: the earlier ones come from a cached prefix
    or an earlier chunk), random index q / cache over non-contiguous block tables."""
    g = torch.Generator().manual_seed(seed)
    nblk = [(c + PG - 1) // PG for c, _ in chunks]
    total = sum(nblk) + 3
    perm = torch.randperm(total, generator=g)
    bt = torch.zeros(len(chunks), max(nblk), dtype=torch.int32)
    o = 0
    for s, nb in enumerate(nblk):
        bt[s, :nb] = perm[o : o + nb].to(torch.int32)
        o += nb
    pos = torch.cat([torch.arange(c - t, c) for c, t in chunks])
    seq = torch.cat([torch.full((t,), s) for s, (_, t) in enumerate(chunks)])
    n = pos.shape[0]
    return SimpleNamespace(
        n=n,
        nb=max(nblk),
        bt=bt.to(DEV),
        pos=pos.to(DEV),
        seq=seq.to(DEV),
        iq=torch.randn(n, NI, DI, generator=g).to(DEV),
        ic=torch.randn(total, PG, DI, generator=g).to(DEV),
        cq=torch.randn(DI, generator=g).to(DEV),
    )


# a continued prefix, a fresh prompt, a last chunk ending mid-block, a decode row
CHUNKS = [(300, 120), (150, 150), (205, 13), (40, 1)]
MODES = {
    "pool4/max": (K.BLOCK_POOL4MAX, K.HEAD_MAX, 1.0),
    "lse/lse": (K.BLOCK_TOKLSE, K.HEAD_LSE, DI**-0.5),
}


def _select(b, rows, mode, k):
    block, head, sc = MODES[mode]
    score = K.scores(
        b.iq[rows],
        b.ic,
        b.bt,
        b.seq[rows],
        b.pos[rows],
        b.nb,
        b.cq,
        1,
        2,
        sc,
        block_mode=block,
        head_mode=head,
        pool_temp=DI**-0.5,
    )
    return K.select_rows(score, b.bt, b.seq[rows], b.pos[rows], k)


@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("tp", [3, 8])
def test_split_selection_is_the_full_selection(mode, tp):
    b, k = _batch(CHUNKS), 4
    assert b.nb > k  # the selection runs
    rows, ctx = _select(b, torch.arange(b.n, device=DEV), mode, k)
    parts = []
    for r in range(tp):
        idx, seg = row_split_index(b.n, tp, r, DEV)
        rr, cc = _select(b, idx, mode, k)
        parts.append(torch.cat([rr, cc[:, None]], dim=1))
    full = assemble_rows(torch.cat(parts), seg, b.n, tp)
    assert torch.equal(full[:, :k], rows) and torch.equal(full[:, k], ctx)


def _mini_impl(b, k, monkeypatch):
    """A ProSparseAttentionImpl over batch ``b`` whose prep hands back ``b``'s index q and
    whose aiter attention records the page tables it is given (returned list)."""
    impl = object.__new__(ProSparseAttentionImpl)
    impl.page, impl.topk, impl.init_blocks, impl.local_blocks = PG, k, 1, 2
    impl.index_head_dim = DI
    impl.__dict__["index_compress_query"] = SimpleNamespace(data=b.cq)
    q = torch.zeros(b.n, H, D, device=DEV)
    monkeypatch.setattr(K, "prep", lambda *a, **kw: (q, b.iq))
    seen = []
    monkeypatch.setattr(
        K,
        "attention_aiter",
        lambda q, kc, vc, rows, ctx, out, *a, **kw: seen.append((rows, ctx)),
    )
    impl.rotary_emb = SimpleNamespace(
        cos_cache=torch.zeros(1, 1, device=DEV), sin_cache=torch.zeros(1, 1, device=DEV)
    )
    impl.q_norm = impl.k_norm = impl.index_q_norm = impl.index_k_norm = SimpleNamespace(
        weight=SimpleNamespace(data=None), variance_epsilon=1e-6
    )
    impl.num_heads, impl.scale, impl.layer_num = H, D**-0.5, 0
    layer = SimpleNamespace(
        k_cache=torch.zeros(1, 1, 1, 1, 1, device=DEV),
        v_cache=torch.zeros(1, 1, 1, 1, 1, device=DEV),
    )
    return impl, seen, (None, b.pos, b.seq, None, b.bt, layer, b.ic, b.nb)


def test_kernel_attention_with_a_split_hands_aiter_the_full_tables(monkeypatch):
    """``_kernel_attention(split=...)`` on rank 1 of 4: the page tables it gives the attention
    equal the unsplit path's (the other ranks' shares stand in for the all-gather)."""
    tp, rank, k = 4, 1, 4
    b = _batch(CHUNKS, seed=3)
    impl, seen, args = _mini_impl(b, k, monkeypatch)
    v = ProVariants(kernels=True, attn="aiter", topk="fused")
    shares = []
    for r in range(tp):
        idx, seg = row_split_index(b.n, tp, r, DEV)
        rr, cc = _select(b, idx, "pool4/max", k)
        shares.append(torch.cat([rr, cc[:, None]], dim=1))
    comm = importlib.import_module("aiter.dist.communication_op")

    def gather(mine, dim=0):
        assert torch.equal(mine, shares[rank])
        return torch.cat(shares)

    monkeypatch.setattr(comm, "tensor_model_parallel_all_gather", gather)
    idx, seg = row_split_index(b.n, tp, rank, DEV)
    impl._kernel_attention(*args, v, None, split=(idx, seg, b.n, tp))
    impl._kernel_attention(*args, v, None)
    (rows_split, ctx_split), (rows, ctx) = seen
    assert torch.equal(rows_split, rows) and torch.equal(ctx_split, ctx)


@pytest.mark.parametrize("rows_per_seq", [1, 4])
def test_static_verify_takes_query_blocked_scores(monkeypatch, rows_per_seq):
    """In the decode graph's static path a speculative verify (k + 1 rows a sequence) scores
    with score_tq and selects as the one-row kernel does; a plain decode keeps one row.
    """
    b = _batch([(c, rows_per_seq) for c in (90, 160, 300, 41)], seed=8)
    impl, seen, args = _mini_impl(b, 4, monkeypatch)
    calls = []
    real = K.scores
    monkeypatch.setattr(
        K, "scores", lambda *a, **kw: calls.append(kw["tq"]) or real(*a, **kw)
    )
    for tq in (0, 4):
        v = ProVariants(kernels=True, attn="aiter", topk="fused", score_tq=tq)
        impl._kernel_attention(*args, v, None, static=True)
    assert calls == [0, 4 if rows_per_seq > 1 else 0]
    (r0, c0), (r1, c1) = seen
    assert torch.equal(r0, r1) and torch.equal(c0, c1)


@pytest.mark.parametrize("rows_per_seq", [2, 4, 8])
def test_static_verify_tile_never_spans_two_sequences(monkeypatch, rows_per_seq):
    """With score_tq 8, a verify of k + 1 rows a sequence (k = 3 gives 4) scores in tiles
    of the largest power of two dividing its rows, so no tile spans two sequences (which
    the kernel would score row by row), and selects as the one-row kernel does.
    """
    b = _batch([(c, rows_per_seq) for c in (90, 160, 300, 41)], seed=9)
    impl, seen, args = _mini_impl(b, 4, monkeypatch)
    calls = []
    real = K.scores
    monkeypatch.setattr(
        K, "scores", lambda *a, **kw: calls.append(kw["tq"]) or real(*a, **kw)
    )
    for tq in (0, 8):
        v = ProVariants(kernels=True, attn="aiter", topk="fused", score_tq=tq)
        impl._kernel_attention(*args, v, None, static=True)
    assert calls == [0, rows_per_seq]
    (r0, c0), (r1, c1) = seen
    assert torch.equal(r0, r1) and torch.equal(c0, c1)


def _plan(monkeypatch, n_tok, ctx_lens, prefill=True, **variant):
    impl = object.__new__(ProSparseAttentionImpl)
    impl.page, impl.topk = PG, 128
    monkeypatch.setattr(impl, "_use_kernels", lambda v: True, raising=False)
    monkeypatch.setattr(
        pro_attn,
        "refresh_variants",
        lambda: ProVariants(kernels=True, attn="aiter", topk="fused", **variant),
    )
    md = SimpleNamespace(
        cu_seqlens_q=torch.tensor([0, n_tok]),
        context_lens=torch.tensor(ctx_lens),
        slot_mapping=torch.zeros(n_tok, dtype=torch.long),
    )
    fwd = SimpleNamespace(
        attn_metadata=md,
        context=SimpleNamespace(is_prefill=prefill, is_dummy_run=False),
    )
    monkeypatch.setattr(pro_attn, "get_forward_context", lambda: fwd)
    ps = importlib.import_module("aiter.dist.parallel_state")

    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda: 3)
    return impl.index_split(n_tok), md


def test_plan_splits_long_selecting_prefills_once_a_forward(monkeypatch):
    plan, md = _plan(monkeypatch, 4096, [8192], index_split=True)
    idx, seg = row_split_index(4096, 8, 3)
    assert plan[1:] == (seg, 4096, 8) and torch.equal(plan[0], idx)
    assert md._m3pro_split is plan


@pytest.mark.parametrize(
    "n_tok,ctx,prefill,on",
    [
        (4096, [8192], True, False),  # switched off
        (1023, [8192], True, True),  # too few rows
        (2048, [2048], True, True),  # 128 blocks: no selection
        (4096, [8192], False, True),  # decode
    ],
)
def test_plan_declines(monkeypatch, n_tok, ctx, prefill, on):
    plan, _ = _plan(monkeypatch, n_tok, ctx, prefill, index_split=on)
    assert plan is None


# ---- query-blocked scoring (ProVariants.score_tq) ------------------------------------
QMODES = {**MODES, "max/sum": (K.BLOCK_TOKMAX, K.HEAD_SUM, 1.0)}


def _scores(b, mode, tq):
    block, head, sc = QMODES[mode]
    return K.scores(
        b.iq, b.ic, b.bt, b.seq, b.pos, b.nb, b.cq, 1, 2, sc,
        block_mode=block, head_mode=head, pool_temp=DI**-0.5, tq=tq,
    )  # fmt: skip


@pytest.mark.parametrize("mode", list(QMODES))
@pytest.mark.parametrize("tq", [2, 8])
def test_query_blocked_scores_equal_the_row_kernel(mode, tq):
    """Tiles inside one sequence share the keys; tiles across sequences (and the ragged
    last one) score row by row: both as _score_tile_kernel does, and so is the selection.
    """
    b = _batch(CHUNKS, seed=5)
    ref, got = _scores(b, mode, 0), _scores(b, mode, tq)
    assert torch.equal(torch.isinf(ref), torch.isinf(got))
    assert torch.equal(ref >= 1e29, got >= 1e29)  # the same pins
    fin = torch.isfinite(ref)
    assert torch.allclose(got[fin], ref[fin], rtol=1e-5, atol=1e-5)
    k = 4
    assert all(
        torch.equal(x, y)
        for x, y in zip(
            K.select_rows(ref, b.bt, b.seq, b.pos, k),
            K.select_rows(got, b.bt, b.seq, b.pos, k),
        )
    )


def test_query_blocked_scores_ignore_unwritten_slots():
    """Keys past a chunk's last token (unwritten slots) are NaN here: they only reach the
    rows' own (pinned) or future blocks, as in _score_tile_kernel."""
    b = _batch([(150, 150)], seed=6)
    last = int(b.pos[-1])
    blk, off = b.bt[0, last // PG], last % PG
    b.ic[blk, off + 1 :] = float("nan")
    for mode in QMODES:
        ref, got = _scores(b, mode, 0), _scores(b, mode, 4)
        assert not torch.isnan(got).any() and torch.equal(
            torch.isinf(ref), torch.isinf(got)
        )
