# SPDX-License-Identifier: MIT
"""The decode / verify index scorer (``pro_kernels.scores`` with ``tq``, the served
``score_tq`` launch) against ``pro_attn``'s torch indexer, at the small-concurrency shapes:
8 / 16 / 24 rows of k7 verify (8 rows a sequence) and 4 / 8 / 12 of k3, contexts from under
128 causal blocks to 128k, a padded graph sequence, init / local pins.

Each program strides over its rows' causal tiles with their index q loaded once
(``tl.num_programs(1)`` programs a row tile). The scores of every causal block must not
depend on how many programs stride, and ``select_logical(split=True)`` must pick the same
ids from them as from the reference scores; with the causal launch nothing past a row's
own block is written, which the selection never reads.

On a GPU the kernels are compiled (bf16 q, fp8 keys, as served); without one they run in
Triton's CPU interpreter on short contexts, the same values in fp32.
"""

from __future__ import annotations

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
from atom.model_ops.minimax_m3.pro_attn import ProSparseAttentionImpl, ProVariants

GPU = torch.cuda.is_available()
DEV = "cuda" if GPU else "cpu"
PG, NI, DI, TOPK = K.PAGE, K.IDX_HEADS, K.IDX_DIM, K.TOPK
INIT, LOCAL = 1, 9  # the served pins (pro_attn's defaults)
# contexts a sequence: under 128 causal blocks, a few tiles, and 16k / 64k / 128k on a GPU
CTXS = (1000, 2100, 16389, 65536, 131064) if GPU else (1000, 2100)
# (rows a sequence, sequences): k7 verify at 8 / 16 / 24 rows, k3 at 4 / 8 / 12
SHAPES = [(8, 1), (8, 2), (8, 3), (4, 1), (4, 2), (4, 3)]
WIDTH = 8256  # block-table width at max-model-len 132096
FP8 = torch.float8_e4m3fn


def _impl(cq):
    m = object.__new__(ProSparseAttentionImpl)
    m.page, m.topk, m.index_head_dim = PG, TOPK, DI
    m.__dict__["index_compress_query"] = SimpleNamespace(data=cq)
    return m


def _case(rps, seqs, ctx, exact, pad=False, seed=0):
    """``seqs`` sequences of ``rps`` verify rows ending at ``ctx`` (each a little shorter
    than the last), fp8 index pages in a shuffled pool, a WIDTH-wide block table. ``exact``:
    q and keys on a 0.5 grid and a zero pool query (equal pool4 weights), so every score is
    exact in fp32 and ties abound. ``pad``: one more graph sequence reusing the last
    sequence's block-table row and positions, as graph replay pads a step."""
    g = torch.Generator().manual_seed(seed + ctx + 31 * rps + 7 * seqs)
    ends = [ctx - 37 * s for s in range(seqs)]
    nblk = [-(-e // PG) for e in ends]
    pool = torch.randperm(sum(nblk) + 5, generator=g).to(torch.int32)
    bt = torch.zeros(seqs + pad, WIDTH, dtype=torch.int32)
    o = 0
    for s, n in enumerate(nblk):
        bt[s, :n] = pool[o : o + n]
        o += n
    pos = [torch.arange(e - rps, e) for e in ends]
    if pad:
        bt[seqs] = bt[seqs - 1]
        pos.append(pos[-1])
        ends.append(ends[-1])
    pos = torch.cat(pos)
    sid = torch.arange(seqs + pad).repeat_interleave(rps)
    pages = sum(nblk) + 5
    if exact:
        ik = torch.randint(-2, 3, (pages, PG, DI), generator=g) * 0.5
        iq = torch.randint(-2, 3, (len(pos), NI, DI), generator=g) * 0.5
        cq = torch.zeros(DI)
    else:
        ik = torch.randn(pages, PG, DI, generator=g) * 0.5
        iq = torch.randn(len(pos), NI, DI, generator=g)
        cq = torch.randn(DI, generator=g)
    # the served dtypes; the interpreter has no fp8 / bf16 dots, so there the same values in fp32
    ic, iq, cq = ik.to(FP8), iq.to(torch.bfloat16), cq.to(torch.bfloat16)
    if not GPU:
        ic, iq, cq = ic.float(), iq.float(), cq.float()
    return SimpleNamespace(
        iq=iq.to(DEV),
        ic=ic.to(DEV),
        bt=bt.to(DEV),
        sid=sid.to(DEV),
        pos=pos.to(DEV),
        cq=cq.to(DEV),
        ends=ends,
        rps=rps,
    )


def _reference(c):
    """pro_attn's torch indexer (the served variants): pool4 centroids, the per-head block
    max, the head max, pins, -inf past each row's own block. ``[T, WIDTH]`` fp32."""
    v = ProVariants()
    impl = _impl(c.cq.float())
    out = torch.full((len(c.pos), WIDTH), float("-inf"), device=DEV)
    j = torch.arange(WIDTH, device=DEV)
    for s, end in enumerate(c.ends):
        rows = (c.sid == s).nonzero().flatten()
        n = -(-end // PG)
        ik = c.ic[c.bt[s, :n].long()].float().reshape(n * PG, DI)
        bmax = impl._block_max(c.iq[rows].float(), ik, None, v)  # [R, NI, n]
        score = pro_attn._reduce_heads(bmax, v.head_reduce, 1.0)
        out[rows, :n] = score
    qb = (c.pos // PG)[:, None]
    out = torch.where(j < INIT, pro_attn._PIN_INIT, out)
    out = torch.where(j >= qb + 1 - LOCAL, pro_attn._PIN_LOCAL, out)
    return torch.where(j <= qb, out, float("-inf"))


def _scores(c, **kw):
    if GPU:  # a NaN-filled block for the output, so an unwritten causal block shows
        torch.full((len(c.pos), WIDTH), float("nan"), device=DEV)
    return K.scores(
        c.iq,
        c.ic,
        c.bt,
        c.sid,
        c.pos,
        WIDTH,
        c.cq,
        INIT,
        LOCAL,
        1.0,
        block_mode=K.BLOCK_POOL4MAX,
        head_mode=K.HEAD_MAX,
        pool_temp=ProVariants().pool_scale,
        tq=c.rps,
        **kw,
    )


def _causal(c):
    return torch.arange(WIDTH, device=DEV)[None, :] <= (c.pos // PG)[:, None]


@pytest.mark.parametrize("ctx", CTXS)
@pytest.mark.parametrize("rps,seqs", SHAPES)
def test_exact_scores_and_picks_equal_the_reference(rps, seqs, ctx):
    c = _case(rps, seqs, ctx, exact=True, pad=seqs == 2)
    ref = _reference(c)
    causal = _causal(c)
    want = K.select_logical(ref, c.pos, TOPK, split=True)
    for kw in ({"causal": True}, {"causal": True, "programs": 3}, {"causal": False}):
        got = _scores(c, **kw)
        assert torch.equal(got[causal], ref[causal]), kw
        if not kw["causal"]:  # the full launch writes -inf past each row's own block
            assert torch.equal(got, ref), kw
        ids = K.select_logical(got, c.pos, TOPK, split=True)
        assert torch.equal(ids, want), kw


@pytest.mark.parametrize("ctx", CTXS)
@pytest.mark.parametrize("rps,seqs", [(8, 1), (8, 3), (4, 2)])
def test_random_scores_match_the_reference(rps, seqs, ctx):
    # continuous q, keys and pool query: the reference pools the keys before the dot, the
    # kernel weights the dots, so they agree to fp32 rounding; the pins exactly
    c = _case(rps, seqs, ctx, exact=False)
    ref = _reference(c)
    causal = _causal(c)
    got = _scores(c, causal=True)
    assert torch.allclose(got[causal], ref[causal], rtol=1e-3, atol=1e-3)
    pinned = causal & (ref >= pro_attn._PIN_LOCAL)
    assert torch.equal(got[pinned], ref[pinned])


@pytest.mark.parametrize("rps,seqs", [(8, 1), (8, 3), (4, 3)])
def test_scores_do_not_depend_on_the_programs(rps, seqs):
    # however many programs stride over a row tile's tiles, a causal block's score is the
    # same bit for bit; one program a tile (no stride: the fp32 pool logits) agrees to
    # fp32 rounding and picks the same blocks
    c = _case(rps, seqs, CTXS[-1], exact=False, pad=True)
    causal = _causal(c)
    full = _scores(c, causal=False)
    one = _scores(c, causal=True, programs=1)
    for programs in (2, 7, None):
        got = _scores(c, causal=True, programs=programs)
        assert torch.equal(got[causal], one[causal]), programs
    flat = _scores(c, causal=True, programs=WIDTH)  # at least a program a tile
    assert torch.equal(flat[causal], full[causal])
    assert torch.allclose(one[causal], full[causal], rtol=1e-5, atol=1e-5)
    want = K.select_logical(full, c.pos, TOPK, split=True)
    assert torch.equal(K.select_logical(one, c.pos, TOPK, split=True), want)
