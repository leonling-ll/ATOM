# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's fused decode scorer (``ProVariants.index = "fused"``).

The decode graph's static path scores with aiter's FlyDSL scorer
(``pro_kernels.scores_fused``) when the variant asks for it and the reading is pool4 /
max; everything else keeps the Triton scores. Without a GPU the scorer is a stand-in
that checks its arguments and returns the Triton scores, so the selection must come
out as the Triton path's. On an MI355X with aiter's scorer the real one runs and its
picks must equal the Triton path's.
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

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PG, NI, DI, H, D = 16, 16, 128, 16, 256


def _batch(lens, rps, seed=0, width=None):
    """``len(lens)`` sequences of ``rps`` query rows each (a DSpark verify when
    ``rps`` > 1), the last at position ``len - 1``, over shuffled block tables
    ``width`` wide."""
    g = torch.Generator().manual_seed(seed)
    nblk = [(c + PG - 1) // PG for c in lens]
    width = width or max(nblk)
    total = sum(nblk) + 3
    perm = torch.randperm(total, generator=g)
    bt = torch.zeros(len(lens), width, dtype=torch.int32)
    o = 0
    for s, nb in enumerate(nblk):
        bt[s, :nb] = perm[o : o + nb].to(torch.int32)
        o += nb
    pos = torch.cat([torch.arange(c - rps, c) for c in lens])
    seq = torch.arange(len(lens)).repeat_interleave(rps)
    n = pos.shape[0]
    return SimpleNamespace(
        n=n,
        nb=width,
        bt=bt.to(DEV),
        pos=pos.to(DEV),
        seq=seq.to(DEV),
        iq=torch.randn(n, NI, DI, generator=g).to(torch.bfloat16).to(DEV),
        ic=torch.randn(total, PG, DI, generator=g).to(DEV),
        cq=torch.randn(DI, generator=g).to(torch.bfloat16).to(DEV),
    )


def _cache(b, dtype):
    """``b``'s index keys as the dispatch tests' cache: fp8 (what the fused scorer
    reads) or bf16 (what it does not)."""
    return b.ic.to(dtype)


def _mini_impl(b, k, monkeypatch, ic_dtype=torch.float8_e4m3fn, min_rows=0):
    """A ProSparseAttentionImpl over batch ``b`` whose prep hands back ``b``'s index q
    and whose aiter attention records the page tables it is given."""
    # the small batches here sit under the fp8 row floor unless a test keeps it
    monkeypatch.setattr(pro_attn, "_FUSED_MIN_ROWS_FP8", min_rows)
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
    ic = _cache(b, ic_dtype)
    return impl, seen, (None, b.pos, b.seq, None, b.bt, layer, ic, b.nb)


def _stand_in(calls):
    """A stand-in for aiter's scorer: records its call, checks that every group of
    ``rows`` rows is one sequence's, and writes the Triton pool4 / max scores of
    the causal blocks."""

    def scorer(iq, ic, cq, bt, seq, pos, out, init, local, sc, pool_temp, rows):
        t = iq.shape[0]
        groups = seq[: t - t % rows].view(-1, rows)
        assert bool((groups == groups[:, :1]).all()), "a group spans two sequences"
        calls.append(rows)
        out.copy_(
            K.scores(
                iq,
                ic,
                bt,
                seq,
                pos,
                out.shape[1],
                cq,
                init,
                local,
                sc,
                block_mode=K.BLOCK_POOL4MAX,
                head_mode=K.HEAD_MAX,
                pool_temp=pool_temp,
            )
        )

    return scorer


@pytest.mark.parametrize("rows_per_seq", [1, 2, 4, 8, 12])
@pytest.mark.parametrize("topk", ["fused", "split"])
def test_static_decode_takes_the_fused_scorer(monkeypatch, rows_per_seq, topk):
    """In the decode graph's static path ``"index": "fused"`` scores with the fused
    scorer, ``rows`` the largest power of two up to 8 dividing a sequence's rows, and
    selects what the Triton scores select."""
    b = _batch([90, 160, 300, 41], rows_per_seq, seed=rows_per_seq)
    impl, seen, args = _mini_impl(b, 4, monkeypatch)
    calls = []
    monkeypatch.setattr(K, "_fused_scorer", _stand_in(calls))
    for index in ("triton", "fused"):
        v = ProVariants(kernels=True, attn="aiter", topk=topk, score_tq=8, index=index)
        impl._kernel_attention(*args, v, None, static=True)
    assert calls == [min(8, rows_per_seq & -rows_per_seq)]
    (r0, c0), (r1, c1) = seen
    assert torch.equal(r0, r1) and torch.equal(c0, c1)


@pytest.mark.parametrize(
    "variant,static",
    [
        ({}, False),  # a prefill
        ({"topk": "torch"}, True),  # reads the whole width
        ({"block_score": "lse", "head_reduce": "lse"}, True),
        ({"head_reduce": "sum"}, True),
        ({"index": "triton"}, True),  # the default
    ],
)
def test_fused_scorer_declines(monkeypatch, variant, static):
    b = _batch([90, 160, 300], 4, seed=5)
    impl, _, args = _mini_impl(b, 4, monkeypatch)
    calls = []
    monkeypatch.setattr(K, "_fused_scorer", _stand_in(calls))
    kw = {"kernels": True, "attn": "aiter", "topk": "fused", "index": "fused"}
    impl._kernel_attention(*args, ProVariants(**{**kw, **variant}), None, static=static)
    assert calls == []


def test_fused_scorer_declines_a_bf16_index_cache(monkeypatch):
    """A bf16 index cache (``--index-cache-dtype bf16``, a served config) keeps the
    Triton scores and selects as the default does."""
    b = _batch([90, 160, 300], 8, seed=6)
    impl, seen, args = _mini_impl(b, 4, monkeypatch, ic_dtype=torch.bfloat16)
    calls = []
    monkeypatch.setattr(K, "_fused_scorer", _stand_in(calls))
    for index in ("triton", "fused"):
        v = ProVariants(kernels=True, attn="aiter", topk="split", index=index)
        impl._kernel_attention(*args, v, None, static=True)
    assert calls == []
    (r0, c0), (r1, c1) = seen
    assert torch.equal(r0, r1) and torch.equal(c0, c1)


@pytest.mark.parametrize("rows_per_seq,fused", [(8, False), (16, True)])
def test_fp8_cache_takes_the_fused_scorer_from_32_rows(
    monkeypatch, rows_per_seq, fused
):
    """On an fp8 index cache the fused scorer runs from 32 rows a step (2 x 16 here),
    and Triton's below (2 x 8)."""
    b = _batch([200, 300], rows_per_seq, seed=11)
    impl, _, args = _mini_impl(b, 4, monkeypatch, min_rows=32)
    calls = []
    monkeypatch.setattr(K, "_fused_scorer", _stand_in(calls))
    v = ProVariants(kernels=True, attn="aiter", topk="split", index="fused")
    impl._kernel_attention(*args, v, None, static=True)
    assert calls == ([8] if fused else [])


def test_default_index_is_triton():
    assert ProVariants().index == "triton"


def _real_scorer_available():
    return torch.cuda.is_available() and K.fused_scores_available()


@pytest.mark.skipif(
    not _real_scorer_available(), reason="aiter's FlyDSL Pro scorer on gfx950"
)
@pytest.mark.parametrize("nvfp4", [False, True])
@pytest.mark.parametrize(
    "lens,rps,width",
    [
        ([4100, 33000], 8, 2100),  # k7 verify, a graph's wider table
        ([65536, 30000, 700], 4, None),  # k3 verify
        ([131072], 8, 8256),  # 128k, the served table width
        ([5000, 6000, 7000, 9000], 1, 600),  # plain decode
    ],
)
def test_fused_picks_equal_the_triton_picks(nvfp4, lens, rps, width):
    """aiter's scorer and the Triton scorer, both followed by today's selection,
    pick the same blocks: each row's set of pages and of block ids, the token's
    own block last in the page table, and the context lengths are equal. The
    order of the full blocks follows the scores, which agree to fp32 rounding, so
    two blocks within an ulp of each other may swap places."""
    b = _batch(lens, rps, seed=len(lens) + rps, width=width)
    if nvfp4:
        r = K.nvfp4_rows(b.ic)  # [blocks, 16, 72]: a page is 16 x 64 B, then 16 x 8 B
        ic = torch.cat([r[..., :64].flatten(1), r[..., 64:].flatten(1)], 1).contiguous()
    else:
        ic = b.ic.to(torch.float8_e4m3fn)
    args = (b.iq, ic, b.bt, b.seq, b.pos, b.nb, b.cq, 1, 9, 1.0)
    tq = 0 if nvfp4 else min(8, rps & -rps) if rps > 1 else 0
    today = K.scores(
        *args,
        block_mode=K.BLOCK_POOL4MAX,
        head_mode=K.HEAD_MAX,
        pool_temp=DI**-0.5,
        causal=True,
        tq=tq,
    )
    rows = min(8, rps & -rps) if rps > 1 else 1
    fused = K.scores_fused(*args, DI**-0.5, rows)
    (r0, c0), (r1, c1) = (
        K.select_rows_split(s, b.bt, b.seq, b.pos, 128) for s in (today, fused)
    )
    assert torch.equal(c0, c1)
    assert torch.equal(r0.sort(1).values, r1.sort(1).values)
    # the own block (the last page within the context) sits at the same slot
    n_full = (c0 + PG - 1) // PG - 1
    own = n_full.long()[:, None]
    assert torch.equal(r0.gather(1, own), r1.gather(1, own))
    l0, l1 = (K.select_logical(s, b.pos, 128, split=True) for s in (today, fused))
    assert torch.equal(l0.sort(1).values, l1.sort(1).values)


def _gate_impl():
    impl = object.__new__(ProSparseAttentionImpl)
    impl.dcp_world, impl.layer_num = 1, 0
    return impl


@pytest.mark.parametrize(
    "compress,cache_rows,dtype,fused",
    [
        (True, 4, torch.bfloat16, True),  # the compressed cache: bf16 centroids
        (False, 16, torch.bfloat16, False),  # a bf16 token cache
        (True, 4, torch.float8_e4m3fn, False),  # not what index_compress writes
        (False, 16, torch.float8_e4m3fn, True),
    ],
)
def test_fused_index_takes_the_centroid_cache_only_with_index_compress(
    monkeypatch, compress, cache_rows, dtype, fused
):
    """``"index": "fused"`` reads a bf16 index cache only as ``index_compress``'s
    centroids."""
    monkeypatch.setattr(pro_attn, "_FUSED_MIN_ROWS_FP8", 0)
    monkeypatch.setattr(K, "_fused_scorer", lambda *a, **kw: None)
    ic = torch.zeros(8, cache_rows, DI, device=DEV).to(dtype)
    v = ProVariants(
        kernels=True, attn="aiter", topk="split", index="fused", index_compress=compress
    )
    assert _gate_impl()._fused_index(v, True, None, ic, 32) is fused


def test_fused_index_declines_centroids_without_the_aiter_format(monkeypatch):
    """An aiter whose scorer has no centroid format keeps the Triton centroid scores."""
    monkeypatch.setattr(K, "_fused_scorer", None)
    monkeypatch.setattr(
        K, "fused_scores_available", lambda centroid=False: not centroid
    )
    ic = torch.zeros(8, 4, DI, device=DEV).to(torch.bfloat16)
    v = ProVariants(
        kernels=True, attn="aiter", topk="split", index="fused", index_compress=True
    )
    assert _gate_impl()._fused_index(v, True, None, ic, 32) is False


@pytest.mark.skipif(
    not (torch.cuda.is_available() and K.fused_scores_available(centroid=True)),
    reason="aiter's FlyDSL Pro scorer with the centroid format on gfx950",
)
@pytest.mark.parametrize(
    "lens,rps,width",
    [
        ([65536, 30000, 700], 4, None),  # k3 verify
        ([4100, 33000], 8, 2100),  # k7 verify, a graph's wider table
        ([131072], 4, 8256),  # 128k, the served table width
        ([5000, 6000, 7000, 9000], 1, 600),  # plain decode
    ],
)
def test_fused_centroid_picks_equal_the_triton_picks(lens, rps, width):
    """On the compressed cache, aiter's scorer and the Triton BLOCK_CENTROID scorer,
    both followed by today's selection, pick the same sets, the own block in its
    slot."""
    b = _batch(lens, rps, seed=len(lens) + rps, width=width)
    g = torch.Generator().manual_seed(rps)
    ic = torch.randn(b.ic.shape[0], 4, DI, generator=g).to(torch.bfloat16).to(DEV)
    args = (b.iq, ic, b.bt, b.seq, b.pos, b.nb, None, 1, 9, 1.0)
    today = K.scores(
        *args,
        block_mode=K.BLOCK_CENTROID,
        head_mode=K.HEAD_MAX,
        causal=True,
        tq=min(8, rps & -rps) if rps > 1 else 0,
    )
    rows = min(8, rps & -rps) if rps > 1 else 1
    fused = K.scores_fused(*args[:6], b.cq, 1, 9, 1.0, DI**-0.5, rows)
    (r0, c0), (r1, c1) = (
        K.select_rows_split(s, b.bt, b.seq, b.pos, 128) for s in (today, fused)
    )
    assert torch.equal(c0, c1)
    assert torch.equal(r0.sort(1).values, r1.sort(1).values)
    n_full = (c0 + PG - 1) // PG - 1
    own = n_full.long()[:, None]
    assert torch.equal(r0.gather(1, own), r1.gather(1, own))
    l0, l1 = (K.select_logical(s, b.pos, 128, split=True) for s in (today, fused))
    assert torch.equal(l0.sort(1).values, l1.sort(1).values)


@pytest.mark.parametrize(
    "compress,dtype", [(True, torch.bfloat16), (False, torch.float8_e4m3fn)]
)
def test_fused_index_declines_a_decode_split(monkeypatch, compress, dtype):
    """Under ``index_split_decode`` a rank scores its share of the rows with Triton,
    which is faster there, whatever the cache."""
    monkeypatch.setattr(pro_attn, "_FUSED_MIN_ROWS_FP8", 0)
    monkeypatch.setattr(K, "_fused_scorer", lambda *a, **kw: None)
    ic = torch.zeros(8, 4 if compress else 16, DI, device=DEV).to(dtype)
    v = ProVariants(
        kernels=True, attn="aiter", topk="split", index="fused", index_compress=compress
    )
    impl = _gate_impl()
    assert impl._fused_index(v, True, None, ic, 32) is True
    assert impl._fused_index(v, True, None, ic, 32, dsplit=(0, 16)) is False


class _Asked(Exception):
    pass


def test_static_path_hands_its_decode_split_to_the_gate(monkeypatch):
    """The scoring step asks the gate with this rank's decode split."""
    b = _batch([90, 160, 300, 41], 4, seed=3)
    impl, _, args = _mini_impl(b, 4, monkeypatch)
    asked = []

    def gate(v, static, split, ic, n, dsplit=None):
        asked.append(dsplit)
        raise _Asked

    monkeypatch.setattr(impl, "_fused_index", gate)
    v = ProVariants(kernels=True, attn="aiter", topk="split", index="fused")
    with pytest.raises(_Asked):
        impl._kernel_attention(*args, v, None, static=True, dsplit=(0, 8))
    assert asked == [(0, 8)]
