# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's Triton attention kernels against the torch reference.

On a machine without a GPU the kernels run in Triton's CPU interpreter (fp32); with a GPU
they are compiled and run in bf16. Either way the reference is ``pro_attn``'s torch code
fed the same raw ``qkv`` rows, and the kernels are compared stage by stage: the normed /
roped q and index-q and the K / V / index-K the caches received, the blocks the scores
select, and the attention over them, for prefill (several tokens per sequence, two
sequences in one batch, non-contiguous block tables) and decode (the last token of each).
The aiter attention runs its page tables through a torch stand-in for aiter's kernel
(which needs a GPU); an fp8 KV cache is checked against the torch code on dequantized K / V.
"""

from __future__ import annotations

import itertools
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

import triton
import triton.language as tl

from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.minimax_m3 import pro_attn
from atom.model_ops.minimax_m3 import pro_kernels as K
from atom.model_ops.minimax_m3.pro_attn import (
    ProKOnlyLayout,
    ProSparseAttentionImpl,
    ProVariants,
)

H, D, NI, DI, PG, X, VD = 16, 256, 16, 128, 16, 8, 192
GPU = torch.cuda.is_available()
DEV = "cuda" if GPU else "cpu"
# PRO_TEST_BF16=1 runs the interpreter in bf16 too (the rounding the compiled kernels see)
BF16 = GPU or os.environ.get("PRO_TEST_BF16") == "1"
DT = torch.bfloat16 if BF16 else torch.float32
ATOL = 3e-2 if BF16 else 2e-4


# the pre-lse reading of the indexer (the kernels' default launch): token max, summed over heads
TOKMAX = {"block_score": "max", "head_reduce": "sum"}


def _impl(topk, init, local):
    m = object.__new__(ProSparseAttentionImpl)
    m.page, m.topk, m.init_blocks, m.local_blocks = PG, topk, init, local
    m.num_heads, m.head_dim, m.rotary_dim = H, D, 64
    m.num_idx_heads, m.index_head_dim, m.value_head_dim = NI, DI, VD
    m.scale = D**-0.5
    m.__dict__["index_compress_query"] = SimpleNamespace(
        data=torch.randn(DI, generator=torch.Generator().manual_seed(7)).to(DEV)
    )
    return m


class Case:
    def __init__(
        self,
        seq_lens,
        topk=4,
        init=1,
        local=2,
        seed=0,
        kv_dtype=None,
        ic_dtype=None,
        k_only=False,
    ):
        g = torch.Generator().manual_seed(seed)
        self.lens, self.topk, self.init, self.local = seq_lens, topk, init, local
        self.S, self.T = len(seq_lens), sum(seq_lens)
        w = H * D + D + NI * DI + DI
        self.qkv = (torch.randn(self.T, w, generator=g) * 2).to(DT).to(DEV)
        self.pos = torch.cat([torch.arange(n) for n in seq_lens]).to(DEV)
        self.seq = torch.cat([torch.full((n,), s) for s, n in enumerate(seq_lens)]).to(
            DEV
        )
        nblk = [(n + PG - 1) // PG for n in seq_lens]
        total = sum(nblk) + 3
        perm = torch.randperm(total, generator=g)
        bt = torch.zeros(self.S, max(nblk), dtype=torch.int32)
        c = 0
        for s, nb in enumerate(nblk):
            bt[s, :nb] = perm[c : c + nb].to(torch.int32)
            c += nb
        self.bt = bt.to(DEV)
        self.slot = (self.bt[self.seq, self.pos // PG].long() * PG + self.pos % PG).to(
            DEV
        )
        kvdt = kv_dtype or DT
        self.fp8 = kvdt.itemsize == 1
        # ATOM's shuffle packs 16 bytes; the K-only page is row-major (x = 256), no V
        self.x = D if k_only else 16 if self.fp8 else X
        self.kc4 = torch.zeros(total, D // self.x, PG, self.x, dtype=kvdt, device=DEV)
        self.vc4 = (
            None
            if k_only
            else torch.zeros(total, PG // self.x, D, self.x, dtype=kvdt, device=DEV)
        )
        scales = lambda: torch.zeros(total, 1, PG, device=DEV) if self.fp8 else None
        self.ks = scales()
        self.vs = None if k_only else scales()
        self.layer = SimpleNamespace(
            k_cache=self.kc4.unsqueeze(1),
            v_cache=self.kc4.unsqueeze(1) if k_only else self.vc4.unsqueeze(1),
            k_scale=self.ks,
            v_scale=self.vs,
        )
        self.ic = torch.zeros(total, PG, DI, dtype=ic_dtype or DT, device=DEV)
        mk = lambda n: (torch.randn(n, generator=g) * 0.1).to(DT).to(DEV)
        self.wq, self.wk, self.wiq, self.wik = mk(D), mk(D), mk(DI), mk(DI)
        maxpos = max(seq_lens) + 8
        ang = torch.arange(maxpos)[:, None] * (0.7 ** torch.arange(32))[None, :]
        self.cos, self.sin = ang.cos().float().to(DEV), ang.sin().float().to(DEV)
        self.impl = _impl(topk, init, local)
        self.eps = 1e-6

    # ---- the torch reference of the prep stage ----------------------------------------
    def ref_prep(self):
        t = self.T
        q = self.qkv[:, : H * D].reshape(t, H, D)
        k = self.qkv[:, H * D : H * D + D].reshape(t, 1, D)
        iq = self.qkv[:, H * D + D : H * D + D + NI * DI].reshape(t, NI, DI)
        ik = self.qkv[:, H * D + D + NI * DI :].reshape(t, 1, DI)
        nrm = lambda w, x: GemmaRMSNorm.forward_static(w, self.eps, x, None)
        c, s = self.cos[self.pos], self.sin[self.pos]
        r = self.impl._rope
        return (
            r(nrm(self.wq, q), c, s),
            r(nrm(self.wk, k), c, s),
            r(nrm(self.wiq, iq), c, s),
            r(nrm(self.wik, ik), c, s),
        )

    def run_prep(self):
        return K.prep(
            self.qkv,
            self.pos,
            self.slot,
            self.cos,
            self.sin,
            1.0,
            self.wq,
            self.wk,
            self.wiq,
            self.wik,
            self.eps,
            self.kc4,
            self.ic,
            H,
            vc4=self.vc4,
            k_scale=self.ks,
            v_scale=self.vs,
            k_unit=getattr(self, "k_unit", False),
        )

    def seq_slice(self, s):
        a = sum(self.lens[:s])
        return a, a + self.lens[s]

    def cached(self):
        """The K ``[T, D]`` and index-K ``[T, DI]`` rows ``prep`` left in the caches: the
        stages after ``prep`` are checked on the kernels' own (bf16-rounded) inputs, so a
        one-ulp difference in ``prep`` cannot flip a top-k and hide a real error there.
        An fp8 K comes back dequantized (fp32)."""
        blk, off = self.slot // PG, self.slot % PG
        k = self.kc4[blk, :, off, :].reshape(self.T, D)
        if self.fp8:
            k = k.float() * self.ks[blk, 0, off][:, None]
        return k, self.ic[blk, off]

    def cached_v(self):
        """The V ``[T, 192]`` ``prep`` left in the V cache (dequantized if fp8)."""
        blk, off = self.slot // PG, self.slot % PG
        v = self.vc4[blk, off // self.x, :VD, off % self.x]
        if self.fp8:
            v = v.float() * self.vs[blk, 0, off][:, None]
        return v


def _close(a, b, atol=ATOL):
    return torch.allclose(a.float(), b.float(), atol=atol, rtol=1e-2 if BF16 else 1e-4)


FP8 = torch.float8_e4m3fn


@triton.jit
def _cast_kernel(x_ptr, y_ptr):
    tl.store(y_ptr, tl.load(x_ptr).to(y_ptr.dtype.element_ty))


def _fp8_casts_round():
    """Triton 3.x's interpreter loses the mantissa carry when it rounds to fp8 (31.5 -> 16
    in e4m3): the fp8 cache is only checked where fp32 -> fp8 rounds right."""
    if GPU:
        return True
    y = torch.empty(1, dtype=FP8)
    _cast_kernel[(1,)](torch.tensor([31.5]), y)
    return y.float().item() == 32.0


needs_fp8 = pytest.mark.skipif(
    not _fp8_casts_round(), reason="the Triton interpreter rounds fp32 -> fp8 wrong"
)


class RefPA:
    """aiter's paged-attention decode in torch, called as ``run_pa_decode``: query ``i``
    attends the first ``context_lens[i]`` keys of the pages ``block_tables[i]`` lists
    (K ``[N, 1, D/x, page, x]``, V ``[N, 1, page/x, DV, x]``, per-token scales
    ``[N, 1, page, 1]``). Records each call's arguments."""

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        output,
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        softmax_scale,
        max_seqlen_q,
        partitions,
        partition_size,
        compute_type,
        q_scale,
        k_scale,
        v_scale,
        ps=False,
    ):
        self.calls.append(
            {
                "out": tuple(output.shape),
                "v": tuple(v_cache.shape),
                "partitions": partitions,
                "partition_size": partition_size,
                "compute_type": compute_type,
                "ps": ps,
                "max_seqlen_q": max_seqlen_q,
                "q_scale": q_scale,
                "k_scale": None if k_scale is None else tuple(k_scale.shape),
            }
        )
        assert output.shape[-1] == v_cache.shape[3]  # as the gluon wrapper checks
        xv = v_cache.shape[-1]
        for i in range(q.shape[0]):
            n = int(context_lens[i])
            p = torch.arange(n, device=q.device)
            blk, off = block_tables[i].long()[p // PG], p % PG
            k = k_cache[blk, 0, :, off, :].reshape(n, -1).float()
            v = v_cache[blk, 0, off // xv, :, off % xv].float()
            if k_scale is not None:
                k = k * k_scale[blk, 0, off, 0][:, None]
                v = v * v_scale[blk, 0, off, 0][:, None]
            w = torch.softmax(q[i].float() @ k.T * softmax_scale, dim=-1)
            output[i] = (w @ v).to(output.dtype)


def ref_topk_rows(score, next_n, lens, sel, rows, stride0, stride1, k):
    """aiter's ``top_k_per_row_decode`` as documented: row ``r``'s first ``lens[r]`` scores
    (``next_n`` 1), the rest of its ``k`` indices -1."""
    assert next_n == 1 and score.stride() == (stride0, stride1)
    sel.fill_(-1)
    for r in range(rows):
        n = int(lens[r])
        idx = score[r, :n].topk(min(k, n)).indices
        sel[r, : idx.numel()] = idx.to(sel.dtype)


@pytest.fixture(scope="module")
def case():
    c = Case([150, 97])
    c.q, c.iq = c.run_prep()
    c.rq, c.rk, c.riq, c.rik = c.ref_prep()
    return c


@pytest.fixture(scope="module")
def case8():
    # what --kv_cache_dtype fp8 gives: the index cache follows the KV cache's dtype
    c = Case([150, 97], seed=2, kv_dtype=FP8, ic_dtype=FP8)
    c.q, c.iq = c.run_prep()
    c.rq, c.rk, c.riq, c.rik = c.ref_prep()
    return c


def _tokmax_sel(c):
    nb = c.bt.shape[1]
    sc = K.scores(c.iq, c.ic, c.bt, c.seq, c.pos, nb, None, c.init, c.local, 1.0)
    return sc.topk(min(c.topk, nb), dim=-1, sorted=False).indices.to(torch.int32)


class TestPrep:
    def test_q_and_index_q(self, case):
        assert _close(case.q, case.rq)
        assert _close(case.iq, case.riq)

    def test_k_is_in_the_shuffled_cache(self, case):
        blk, off = case.slot // PG, case.slot % PG
        got = case.kc4[blk, :, off, :].reshape(case.T, D)
        assert _close(got, case.rk[:, 0])

    def test_index_k_is_in_the_index_cache(self, case):
        blk, off = case.slot // PG, case.slot % PG
        assert _close(case.ic[blk, off], case.rik[:, 0])

    def test_v_is_the_cached_k_tail_in_the_v_cache(self, case):
        assert torch.equal(case.cached_v(), case.cached()[0][:, D - VD :])

    def test_the_four_projections_apart_equal_the_packed_rows(self):
        packed, apart = Case([150, 97]), Case([150, 97])
        q0, iq0 = packed.run_prep()
        o = H * D
        cuts = [0, o, o + D, o + D + NI * DI, apart.qkv.shape[1]]
        parts = []
        for a, b in itertools.pairwise(cuts):
            # each in its own buffer, wider than the projection: another row stride
            buf = torch.zeros(apart.T, b - a + 32, dtype=DT, device=DEV)
            buf[:, : b - a] = apart.qkv[:, a:b]
            parts.append(buf[:, : b - a])
        apart.qkv = tuple(parts)
        q1, iq1 = apart.run_prep()
        assert torch.equal(q1, q0) and torch.equal(iq1, iq0)
        assert torch.equal(apart.kc4, packed.kc4) and torch.equal(apart.vc4, packed.vc4)
        assert torch.equal(apart.ic, packed.ic)

    def test_other_slots_untouched(self, case):
        used = torch.zeros(case.kc4.shape[0], dtype=torch.bool, device=DEV)
        used[case.slot // PG] = True
        assert not case.kc4[~used].any() and not case.ic[~used].any()
        assert not case.vc4[~used].any() and not case.vc4[:, :, VD:].any()


class TestScores:
    def _ref_selected(self, c, v):
        """The torch selector's block mask per token."""
        sel = []
        _, ik_all = c.cached()
        for s in range(c.S):
            a, b = c.seq_slice(s)
            n = c.lens[s]
            nb = (n + PG - 1) // PG
            lpad = nb * PG
            ik = torch.zeros(lpad, DI, dtype=ik_all.dtype, device=DEV)
            ik[:n] = ik_all[a:b]
            kpos = torch.arange(lpad, device=DEV)
            pc = torch.arange(n, device=DEV)
            vis = kpos[None, :] <= pc[:, None]
            bmax = c.impl._block_max(c.iq[a:b], ik, vis, v)
            sel.append(c.impl._select_blocks(bmax, pc // PG, v))
        return sel

    @pytest.mark.parametrize(
        "mode,variant",
        [
            (K.BLOCK_TOKMAX, TOKMAX),
            (K.BLOCK_POOL4MAX, {}),
            (
                K.BLOCK_POOL4MAX,
                {"block_score": "pool4", "head_reduce": "sum", "pool_scale": 1.0},
            ),
            (K.BLOCK_TOKLSE, {"block_score": "lse", "head_reduce": "sum"}),
            (
                K.BLOCK_TOKLSE,
                {"block_score": "lse", "head_reduce": "lse", "index_scale": 1.7},
            ),
        ],
    )
    def test_selected_blocks_match_the_reference(self, case, mode, variant):
        v = ProVariants(**variant)
        nb = case.bt.shape[1]
        sc = K.scores(
            case.iq,
            case.ic,
            case.bt,
            case.seq,
            case.pos,
            nb,
            case.impl.index_compress_query.data if mode else None,
            case.init,
            case.local,
            v.index_scale * (DI**-0.5 if mode == K.BLOCK_TOKLSE else 1.0),
            block_mode=mode,
            head_mode={"sum": K.HEAD_SUM, "max": K.HEAD_MAX, "lse": K.HEAD_LSE}[
                v.head_reduce
            ],
            pool_temp=v.pool_scale,
        )
        ref = self._ref_selected(case, v)
        k = min(case.topk, nb)
        idx = sc.topk(k, dim=-1).indices
        for s in range(case.S):
            a, b = case.seq_slice(s)
            got = torch.zeros(b - a, nb, dtype=torch.bool, device=DEV)
            got.scatter_(1, idx[a:b], True)
            valid = (
                torch.arange(nb, device=DEV)[None, :] <= (case.pos[a:b] // PG)[:, None]
            )
            got &= valid
            exp = torch.zeros_like(got)
            exp[:, : ref[s].shape[1]] = ref[s]
            assert torch.equal(got, exp), f"seq {s} mode {mode}"

    def test_future_blocks_are_minus_inf_and_forced_blocks_pinned(self, case):
        nb = case.bt.shape[1]
        sc = K.scores(
            case.iq,
            case.ic,
            case.bt,
            case.seq,
            case.pos,
            nb,
            None,
            case.init,
            case.local,
            1.0,
        )
        qb = case.pos // PG
        j = torch.arange(nb, device=DEV)[None, :]
        assert torch.isinf(sc[j > qb[:, None]]).all()
        local = (j >= qb[:, None] + 1 - case.local) & (j <= qb[:, None])
        assert (sc[local] == 1e29).all()

    @pytest.mark.parametrize(
        "mode", [K.BLOCK_TOKMAX, K.BLOCK_TOKLSE, K.BLOCK_POOL4MAX, K.BLOCK_POOL4MEAN]
    )
    def test_causal_scores_equal_the_full_width_ones(self, case, mode):
        # however many tiles a program strides over, a causal block's score is the full
        # launch's, bit for bit (same tile size); nothing past the token is read after
        nb = case.bt.shape[1]
        pool = mode in (K.BLOCK_POOL4MAX, K.BLOCK_POOL4MEAN)
        cq = case.impl.index_compress_query.data if pool else None
        args = (case.iq, case.ic, case.bt, case.seq, case.pos, nb, cq, 2, 3, 0.1)
        kw = {"block_mode": mode, "head_mode": K.HEAD_LSE, "pool_temp": DI**-0.5}
        causal = torch.arange(nb, device=DEV)[None, :] <= (case.pos // PG)[:, None]
        for bb in (4, 8):
            full = K.scores(*args, **kw, bb=bb)
            for programs in (1, 2, None):
                got = K.scores(*args, **kw, bb=bb, causal=True, programs=programs)
                assert torch.equal(got[causal], full[causal]), (bb, programs)
            for tq in (2, 8):  # the score_tq kernel: the same causal scores either way
                want = K.scores(*args, **kw, bb=bb, tq=tq)
                got = K.scores(*args, **kw, bb=bb, tq=tq, causal=True)
                assert torch.equal(got[causal], want[causal]), (bb, tq)

    @pytest.mark.parametrize(
        "mode", [K.BLOCK_TOKMAX, K.BLOCK_TOKLSE, K.BLOCK_POOL4MAX, K.BLOCK_POOL4MEAN]
    )
    def test_every_tile_size_scores_like_the_per_block_kernel(self, case, mode):
        nb = case.bt.shape[1]
        pool = mode in (K.BLOCK_POOL4MAX, K.BLOCK_POOL4MEAN)
        cq = case.impl.index_compress_query.data if pool else None
        args = (
            case.iq,
            case.ic,
            case.bt,
            case.seq,
            case.pos,
            nb,
            cq,
            case.init,
            case.local,
            0.1,
        )
        kw = {"block_mode": mode, "head_mode": K.HEAD_LSE, "pool_temp": DI**-0.5}
        ref = K.scores(*args, **kw, tiled=False)
        fin = torch.isfinite(ref)
        for bb in (4, 8, 16):
            got = K.scores(*args, **kw, bb=bb)
            assert torch.equal(torch.isfinite(got), fin), bb
            assert _close(got[fin], ref[fin]), bb


class TestCompressedIndex:
    """``BLOCK_CENTROID``: scores over vLLM's compressed index cache ``ic_c [pages, 4, 128]``
    (every 4 index keys softmax-pooled into one row when written): a block's score is the
    head-reduce of the max over its 4 rows of ``iq . c``, with the usual pins."""

    @staticmethod
    def _ic_c(c, dtype=torch.bfloat16):
        # the write path's rows: ATOM's own pool4 centroids of every page's 16 keys
        return c.impl._centroids(c.ic.float(), ProVariants()).to(dtype)

    @staticmethod
    def _ref(c, ic_c, nb, init, local, sc, head_mode):
        qb = c.pos // PG
        cent = ic_c.float()[c.bt[c.seq].long()]  # [T, NB, 4, DI]
        # scaled before the head reduce, as the kernel does (lse is not scale-equivariant)
        dots = torch.einsum("thd,tjrd->thjr", c.iq.float(), cent[:, :nb]).amax(-1) * sc
        if head_mode == K.HEAD_MAX:
            score = dots.amax(1)
        elif head_mode == K.HEAD_SUM:
            score = dots.sum(1)
        else:
            score = torch.logsumexp(dots, 1)
        j = torch.arange(nb, device=DEV)[None, :]
        score = torch.where(j < init, 1e30, score)
        score = torch.where(j >= qb[:, None] + 1 - local, 1e29, score)
        return torch.where(j <= qb[:, None], score, float("-inf"))

    @pytest.mark.parametrize("head_mode", [K.HEAD_MAX, K.HEAD_SUM, K.HEAD_LSE])
    def test_every_path_scores_as_the_reference(self, case, head_mode):
        c, nb = case, case.bt.shape[1]
        ic_c = self._ic_c(c)
        want = self._ref(c, ic_c, nb, 2, 3, 0.1, head_mode)
        causal = torch.arange(nb, device=DEV)[None, :] <= (c.pos // PG)[:, None]
        args = (c.iq, ic_c, c.bt, c.seq, c.pos, nb, None, 2, 3, 0.1)
        kw = {"block_mode": K.BLOCK_CENTROID, "head_mode": head_mode}
        runs = {}
        for bb in (4, 8, None):  # None: the centroid default, 16
            for causal_on in (False, True):
                runs[("tile", bb, causal_on)] = K.scores(
                    *args, **kw, bb=bb, causal=causal_on
                )
        for tq in (2, 8):
            for causal_on in (False, True):
                runs[("qtile", tq, causal_on)] = K.scores(
                    *args, **kw, tq=tq, causal=causal_on
                )
        for name, got in runs.items():
            assert torch.allclose(got[causal], want[causal], rtol=1e-3, atol=1e-3), name
            if not name[2]:  # the full launch also writes the future blocks as -inf
                assert torch.isinf(got[~causal]).all(), name

    def test_equals_pool4_on_the_same_keys(self, case):
        # pooling is linear: sum_i w_i (q . k_i) == q . (sum_i w_i k_i). With fp32 rows the
        # centroid scores are today's pool4 / max scores to rounding
        c, nb = case, case.bt.shape[1]
        causal = torch.arange(nb, device=DEV)[None, :] <= (c.pos // PG)[:, None]
        kw = {"head_mode": K.HEAD_MAX, "pool_temp": ProVariants().pool_scale}
        pool4 = K.scores(
            c.iq,
            c.ic,
            c.bt,
            c.seq,
            c.pos,
            nb,
            c.impl.index_compress_query.data,
            2,
            3,
            1.0,
            block_mode=K.BLOCK_POOL4MAX,
            **kw,
        )
        cent = K.scores(
            c.iq,
            self._ic_c(c, torch.float32),
            c.bt,
            c.seq,
            c.pos,
            nb,
            None,
            2,
            3,
            1.0,
            block_mode=K.BLOCK_CENTROID,
            **kw,
        )
        assert torch.allclose(cent[causal], pool4[causal], rtol=1e-3, atol=1e-3)

    def test_rejects_a_token_cache(self, case):
        with pytest.raises(AssertionError):
            K.scores(
                case.iq,
                case.ic,
                case.bt,
                case.seq,
                case.pos,
                case.bt.shape[1],
                None,
                2,
                3,
                1.0,
                block_mode=K.BLOCK_CENTROID,
            )


class TestAttention:
    def _run(self, case, tokens, splits, dense, nblk=4, n_blocks=None):
        sel = None if dense else _tokmax_sel(case)
        out = torch.zeros(case.T, H * VD, dtype=DT, device=DEV)
        K.attention(
            case.q,
            case.kc4,
            case.bt,
            case.seq,
            case.pos,
            sel,
            out,
            D**-0.5,
            splits,
            nblk=nblk,
            n_blocks=n_blocks,
            k_scale=case.ks,
        )
        return out

    def _ref(self, case, dense, v_cache=False):
        """The torch attention over the cached K (V its tail, or the V cache's V)."""
        v = ProVariants(dense_only=dense, **TOKMAX)
        k_all, ik_all = case.cached()
        v_all = case.cached_v() if v_cache else None
        outs = []
        for s in range(case.S):
            a, b = case.seq_slice(s)
            n = case.lens[s]
            outs.append(
                case.impl._attend_prefill_seq(
                    case.q[a:b],
                    case.iq[a:b],
                    torch.arange(n, device=DEV),
                    k_all[a:b],
                    ik_all[a:b],
                    v,
                    None if v_all is None else v_all[a:b],
                )
            )
        return torch.cat(outs)

    @pytest.mark.parametrize("splits,nblk", [(1, 1), (1, 4), (4, 2), (3, 4), (2, 8)])
    def test_sparse_prefill(self, case, splits, nblk):
        # topk 4 blocks of 16 tokens: the 150- and 97-token sequences drop blocks
        got = self._run(case, None, splits, dense=False, nblk=nblk)
        assert _close(got, self._ref(case, dense=False))

    @pytest.mark.parametrize("splits,nblk", [(1, 4), (3, 2), (8, 1)])
    def test_dense_matches_reference_dense(self, case, splits, nblk):
        got = self._run(case, None, splits, dense=True, nblk=nblk)
        assert _close(got, self._ref(case, dense=True))

    def test_dense_over_more_blocks_than_topk(self):
        # 40 blocks, dense, with a topk of 4: every block is read (n_blocks given)
        c = Case([40 * PG + 5], topk=4)
        c.q, c.iq = c.run_prep()
        out = torch.zeros(c.T, H * VD, dtype=DT, device=DEV)
        K.attention(
            c.q,
            c.kc4,
            c.bt,
            c.seq,
            c.pos,
            None,
            out,
            D**-0.5,
            2,
            nblk=4,
            n_blocks=c.bt.shape[1],
        )
        k_all, ik_all = c.cached()
        ref = c.impl._attend_prefill_seq(
            c.q,
            c.iq,
            torch.arange(c.T, device=DEV),
            k_all,
            ik_all,
            ProVariants(dense_only=True),
        )
        assert _close(out, ref)

    def test_sparse_differs_from_dense(self, case):
        sparse = self._run(case, None, 1, dense=False)
        dense = self._run(case, None, 1, dense=True)
        assert not _close(sparse[-1:], dense[-1:], atol=1e-3)

    def test_decode_rows_equal_prefill_rows(self, case):
        # the last token of each sequence, run as a 2-token decode batch over the same caches
        last = torch.tensor([case.lens[0] - 1, case.T - 1], device=DEV)
        nb = case.bt.shape[1]
        seq, pos = case.seq[last], case.pos[last]
        sc = K.scores(
            case.iq[last],
            case.ic,
            case.bt,
            seq,
            pos,
            nb,
            None,
            case.init,
            case.local,
            1.0,
        )
        sel = sc.topk(min(case.topk, nb), dim=-1, sorted=False).indices.to(torch.int32)
        out = torch.zeros(2, H * VD, dtype=DT, device=DEV)
        K.attention(case.q[last], case.kc4, case.bt, seq, pos, sel, out, D**-0.5, 2)
        ref = self._ref(case, dense=False)[last]
        assert _close(out, ref)


class TestSparseRows:
    """``sparse_rows``: a token's selected blocks as a page table of its own -- the full
    blocks in the selection's order, its own block last -- and a context length that
    stops at the token; dropped entries (negative, future) and unused slots."""

    @staticmethod
    def _ref(sel, c, nb):
        bt, seq, pos = c.bt.cpu(), c.seq.cpu(), c.pos.cpu()
        rows, ctx = [], []
        for i in range(c.T):
            qb = int(pos[i]) // PG
            js = range(nb) if sel is None else sel[i].tolist()
            full = [j for j in js if 0 <= j < qb]
            own = [j for j in js if j == qb]
            rows.append([int(bt[seq[i], j]) for j in full + own])
            ctx.append(len(full) * PG + (int(pos[i]) % PG + 1 if own else 0))
        return rows, ctx

    def _check(self, got, ref, width):
        rows, ctx = got
        assert rows.shape == (len(ref[0]), width) and ctx.tolist() == ref[1]
        for i, r in enumerate(ref[0]):
            assert rows[i, : len(r)].tolist() == r
            assert not rows[i, len(r) :].any()

    def test_selection(self, case):
        g = torch.Generator().manual_seed(11)
        nb = case.bt.shape[1]
        sel = torch.stack([torch.randperm(nb, generator=g)[:6] for _ in range(case.T)])
        sel[::3, 0] = -1
        sel = sel.to(torch.int32)
        got = K.sparse_rows(sel.to(DEV), case.bt, case.seq, case.pos)
        self._check(got, self._ref(sel, case, nb), 6)

    def test_dense(self, case):
        nb = case.bt.shape[1]
        got = K.sparse_rows(None, case.bt, case.seq, case.pos, nb)
        self._check(got, self._ref(None, case, nb), nb)


class TestSelectBlocks:
    """``select_blocks`` (aiter's per-row top-k) against ``torch.topk`` over each token's
    causal blocks, scores as ``scores`` writes them (pins finite, future blocks -inf).
    """

    @pytest.mark.skipif(not GPU, reason="aiter's top-k runs on a GPU")
    @pytest.mark.parametrize("width,k", [(40, 6), (1024, 128)])
    def test_the_aiter_kernel(self, width, k):
        g = torch.Generator(device=DEV).manual_seed(0)
        t = 37
        pos = torch.randint(0, width * PG, (t,), generator=g, device=DEV)
        pos[:3] = torch.tensor([0, 5 * PG + 3, width * PG - 1], device=DEV)
        j = torch.arange(width, device=DEV)
        score = torch.randn(t, width, generator=g, device=DEV)
        score[:, 0], score[:, 1] = 1e30, 1e29
        score = torch.where(j[None, :] <= (pos // PG)[:, None], score, float("-inf"))
        sel = K.select_blocks(score, pos, k)
        for r in range(t):
            n = int(pos[r]) // PG + 1
            want = set(score[r, :n].topk(min(k, n)).indices.tolist())
            got = sel[r].tolist()
            assert {x for x in got if x >= 0} == want, r
            assert got.count(-1) == k - min(k, n), r

    def test_calls_with_each_tokens_block_count(self, monkeypatch):
        seen = {}

        def spy(score, next_n, lens, sel, rows, s0, s1, k):
            seen.update(lens=lens.tolist(), rows=rows, k=k, dtype=sel.dtype)
            ref_topk_rows(score, next_n, lens, sel, rows, s0, s1, k)

        monkeypatch.setattr(K, "_topk_rows", spy)
        pos = torch.tensor([0, 15, 16, 70], device=DEV)
        score = torch.randn(4, 5, device=DEV)
        sel = K.select_blocks(score, pos, 3)
        assert seen == {"lens": [1, 1, 2, 5], "rows": 4, "k": 3, "dtype": torch.int32}
        assert sel[0].tolist() == [0, -1, -1] and sorted(sel[2].tolist()) == [-1, 0, 1]


class TestSelectRows:
    """``select_rows``: the page table ``sparse_rows`` builds for the ``k`` best causal blocks
    of each row (any order of the full ones, the token's own block last), scores as
    ``scores`` writes them: pins finite, future blocks -inf."""

    @pytest.mark.parametrize("select", ["select_rows", "select_rows_split"])
    @pytest.mark.parametrize("width,k", [(40, 8), (1024, 128), (100, 128)])
    def test_against_topk_and_the_rows_rule(self, width, k, select):
        g = torch.Generator(device=DEV).manual_seed(1)
        t, s = 37, 5
        bt = (
            torch.randperm(s * width, generator=g, device=DEV)
            .view(s, width)
            .to(torch.int32)
            + 1
        )
        seq = torch.randint(0, s, (t,), generator=g, device=DEV)
        pos = torch.randint(0, width * PG, (t,), generator=g, device=DEV)
        pos[:4] = torch.tensor([0, 5 * PG + 3, width * PG - 1, k * PG], device=DEV) % (
            width * PG
        )
        qb = pos // PG
        j = torch.arange(width, device=DEV)
        score = torch.randn(t, width, generator=g, device=DEV)
        score[:, 0] = 1e30
        score = torch.where((j[None, :] >= qb[:, None] - 2), 1e29, score)
        score = torch.where(j[None, :] <= qb[:, None], score, float("-inf"))
        rows, ctx = getattr(K, select)(score, bt, seq, pos, k)
        assert rows.shape == (t, k) and rows.dtype == ctx.dtype == torch.int32
        for r in range(t):
            n, q = int(qb[r]) + 1, int(qb[r])
            best = score[r, :n].topk(min(k, n)).indices.tolist()
            full = {int(bt[seq[r], x]) for x in best if x < q}
            got = rows[r].tolist()
            assert set(got[: len(full)]) == full, r
            assert got[len(full)] == int(
                bt[seq[r], q]
            ), r  # always picked: pinned local
            assert not any(got[len(full) + 1 :]), r
            assert int(ctx[r]) == len(full) * PG + int(pos[r]) % PG + 1, r

    @pytest.mark.parametrize("select", ["select_rows", "select_rows_split"])
    def test_ties_go_to_the_lower_block(self, select):
        score = torch.zeros(1, 16, device=DEV)
        score[0, 15] = 1.0
        bt = torch.arange(16, dtype=torch.int32, device=DEV)[None] + 100
        pos = torch.tensor([15 * PG], device=DEV)
        rows, ctx = getattr(K, select)(
            score, bt, torch.zeros(1, dtype=torch.long, device=DEV), pos, 4
        )
        assert sorted(rows[0, :3].tolist()) == [100, 101, 102] and rows[0, 3] == 115
        assert ctx.tolist() == [3 * PG + 1]


class TestSelectRowsSplit:
    """``select_rows_split`` writes ``select_rows``' tables bit for bit, however a row is
    split, merged or wide, and reads none of a row's scores past its own block."""

    @staticmethod
    def _case(t, width, seed, ties=False, ctx=None):
        g = torch.Generator(device=DEV).manual_seed(seed)
        bt = torch.randint(1, 1 << 20, (3, width), generator=g, device=DEV).to(
            torch.int32
        )
        seq = torch.randint(0, 3, (t,), generator=g, device=DEV)
        pos = torch.randint(0, width * PG, (t,), generator=g, device=DEV)
        pos[0], pos[-1] = width * PG - 1, 3  # the widest row and a one-block one
        if ctx is not None:  # a uniform decode batch, as the bench runs it
            pos.fill_(ctx - 1)
        j = torch.arange(width, device=DEV)
        qb = pos // PG
        if ties:  # a few distinct values: most picks are decided by the block index
            score = torch.randint(-2, 3, (t, width), generator=g, device=DEV).float()
            score[:, 1::7] = float("-inf")  # causal -inf blocks are never picked
        else:
            score = torch.randn(t, width, generator=g, device=DEV)
        score[:, :1] = 1e30
        score = torch.where(j[None, :] >= qb[:, None] - 2, 1e29, score)
        # what scores writes past the token is never read: garbage must not change a pick
        score = torch.where(j[None, :] <= qb[:, None], score, float("nan"))
        return score, bt, seq, pos

    @pytest.mark.parametrize("ties", [False, True])
    @pytest.mark.parametrize(
        "t,width,k,chunk,parts,fused",
        [
            (5, 40, 8, 32, 1, None),  # one span, looped chunks
            (5, 40, 8, 32, 2, True),
            (5, 40, 8, 32, 2, False),
            (3, 300, 8, 32, None, True),  # a span program per chunk
            (19, 300, 8, 32, 4, False),
            (19, 300, 8, 64, 3, None),
            (4, 1000, 32, 128, None, None),
            (2, 100, 128, 512, None, None),  # fewer blocks than k
        ],
    )
    def test_equals_select_rows(self, t, width, k, chunk, parts, fused, ties):
        score, bt, seq, pos = self._case(t, width, seed=t + width + k, ties=ties)
        want = K.select_rows(
            score.masked_fill(score.isnan(), float("-inf")), bt, seq, pos, k
        )
        got = K.select_rows_split(
            score, bt, seq, pos, k, parts=parts, chunk=chunk, fused=fused
        )
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])

    @pytest.mark.parametrize(
        "t,width,k,chunk,parts,fused",
        [
            (5, 40, 8, 32, 1, None),
            (5, 40, 8, 32, 2, True),
            (19, 300, 8, 32, 4, False),
            (2, 100, 128, 512, None, None),  # fewer blocks than k: -1 after
        ],
    )
    def test_logical_equals_select_logical(self, t, width, k, chunk, parts, fused):
        # the K-only cache's block ids (day0, Triton attention) through the split path
        score, bt, _seq, pos = self._case(t, width, seed=t + width + k, ties=True)
        want = K.select_logical(score.masked_fill(score.isnan(), float("-inf")), pos, k)
        got = K.select_rows_split(
            score, bt, pos, pos, k, parts=parts, chunk=chunk, fused=fused, logical=True
        )[0]
        assert torch.equal(got, want)
        if parts is None:
            assert torch.equal(K.select_logical(score, pos, k, split=True), want)

    def test_rearms_its_counters(self):
        # the fused merge zeroes its arrival counters: a second call (a graph replay)
        # selects alike
        score, bt, seq, pos = self._case(3, 300, seed=9)
        args = {"parts": None, "chunk": 32, "fused": True}
        a = K.select_rows_split(score, bt, seq, pos, 8, **args)
        b = K.select_rows_split(score, bt, seq, pos, 8, **args)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])

    @staticmethod
    def _poison_scratch():
        """Fill every partial scratch with keys above any real one (a block id read back
        from them is garbage); the arrival counters must be back at zero."""
        for (_, _, kind), bufs in K._split_buffers.items():
            for b in bufs:
                if kind == "part":
                    b.random_(1 << 62)
                else:
                    assert not b.any(), "an arrival counter was left armed"

    # (rows, width, ctx) in the 004 bench order that faulted (a few rows over the whole
    # table, then many rows with several spans each) and wider ones; the interpreter
    # runs them scaled down (k 8 over chunks of 32 blocks: the same spans and merges)
    SEQUENCE_GPU = (
        (16, 8192, 131072),
        (64, 8192, 1024),
        (64, 8192, 8192),
        (64, 8192, 32768),
        (200, 8192, 32768),
        (64, 8192, 131072),
        (200, 8192, 131072),
        (64, 65536, 32768),
        (200, 65536, 131072),
        (16, 65536, 1048576),
    )
    SEQUENCE_CPU = ((16, 256, 4096), (64, 256, 128), (64, 256, 2048), (40, 512, 8192))

    @pytest.mark.parametrize("fused", [None, True, False])
    def test_shape_sequence_over_garbage_scratch(self, fused):
        k, chunk = (128, 1024) if GPU else (8, 32)
        for i, (t, width, ctx) in enumerate(
            self.SEQUENCE_GPU if GPU else self.SEQUENCE_CPU
        ):
            score, bt, seq, pos = self._case(t, width, seed=i, ctx=ctx)
            want = K.select_rows(
                score.masked_fill(score.isnan(), float("-inf")), bt, seq, pos, k
            )
            for _ in range(2):
                self._poison_scratch()
                got = K.select_rows_split(
                    score, bt, seq, pos, k, chunk=chunk, fused=fused
                )
                assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]), (
                    t,
                    width,
                    ctx,
                )
        self._poison_scratch()

    @pytest.mark.skipif(not GPU, reason="wide rows take long in the interpreter")
    @pytest.mark.parametrize("t", [1, 16, 64, 200])
    @pytest.mark.parametrize("width", [1024, 8192, 65536])
    def test_wide_rows_equal_select_rows(self, t, width):
        score, bt, seq, pos = self._case(t, width, seed=t * width)
        want = K.select_rows(
            score.masked_fill(score.isnan(), float("-inf")), bt, seq, pos, 128
        )
        for fused in (True, False):
            got = K.select_rows_split(score, bt, seq, pos, 128, fused=fused)
            assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1])

    @pytest.mark.skipif(not GPU, reason="CUDA graphs need a GPU")
    @pytest.mark.parametrize("t", [4, 32])
    def test_graph_replays_other_rows(self, t):
        # captured once over a wide table, replayed for other scores and positions: the
        # spans past each row exit, the fused merge's counters are rearmed every replay
        score, bt, seq, pos = self._case(t, 8192, seed=t)
        s_buf, p_buf = score.clone(), pos.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            K.select_rows_split(s_buf, bt, seq, p_buf, 128)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = K.select_rows_split(s_buf, bt, seq, p_buf, 128)
        for seed in (1, 2):
            sc, _, _, po = self._case(t, 8192, seed=100 + seed)
            po[1] = 8192 * PG - 1 - seed  # keep a long row in the batch
            sc = torch.where(
                torch.arange(8192, device=DEV)[None, :] <= (po // PG)[:, None],
                sc.masked_fill(sc.isnan(), 0.5),
                float("nan"),
            )
            s_buf.copy_(sc), p_buf.copy_(po)
            g.replay()
            want = K.select_rows(
                sc.masked_fill(sc.isnan(), float("-inf")), bt, seq, po, 128
            )
            torch.cuda.synchronize()
            assert torch.equal(out[0], want[0]) and torch.equal(out[1], want[1])


class TestAttentionAiter:
    """``attention_aiter`` -- page tables, cache views and scales handed to aiter's paged
    attention -- with a torch stand-in for the kernel (``RefPA``)."""

    PARTS = 5

    def _run(self, c, sel, monkeypatch, n_blocks=None, **kw):
        pa = RefPA()
        monkeypatch.setattr(K, "_pa_decode", pa)
        monkeypatch.setattr(K, "_default_partitions", lambda t: self.PARTS)
        rows, ctx = K.sparse_rows(sel, c.bt, c.seq, c.pos, n_blocks)
        out = torch.zeros(c.T, H * VD, dtype=DT, device=DEV)
        K.attention_aiter(
            c.q,
            c.layer.k_cache,
            c.layer.v_cache,
            rows,
            ctx,
            out,
            D**-0.5,
            k_scale=c.ks,
            v_scale=c.vs,
            **kw,
        )
        (call,) = pa.calls
        assert call["out"] == (c.T, H, VD) and call["v"][3] == VD and call["ps"]
        assert call["max_seqlen_q"] == 1 and call["q_scale"] is None
        assert call["partitions"] == min(self.PARTS, -(-rows.shape[1] * PG // 256))
        return out, call

    @pytest.mark.parametrize("dense", [False, True])
    def test_equals_the_triton_attention(self, case, dense, monkeypatch):
        nb = case.bt.shape[1]
        sel = None if dense else _tokmax_sel(case)
        got, call = self._run(case, sel, monkeypatch, nb if dense else None)
        assert call["compute_type"] == DT and call["k_scale"] is None
        ref = torch.zeros_like(got)
        K.attention(
            case.q,
            case.kc4,
            case.bt,
            case.seq,
            case.pos,
            sel,
            ref,
            D**-0.5,
            2,
            n_blocks=nb if dense else None,
        )
        assert _close(got, ref)

    @pytest.mark.skipif(not GPU, reason="aiter's paged attention needs a GPU")
    @pytest.mark.parametrize(
        "kv_dtype,fp8_compute", [(None, False), (FP8, False), (FP8, True)]
    )
    @pytest.mark.parametrize("decode", [False, True])
    def test_the_aiter_kernel(self, kv_dtype, fp8_compute, decode):
        # the real kernel: Q/K 256, V 192 (a view of the 256-wide V cache), page 16
        c = Case([1500, 970], topk=40, seed=4, kv_dtype=kv_dtype)
        c.q, c.iq = c.run_prep()
        sel = _tokmax_sel(c)
        ref = TestAttention()._ref(c, dense=False, v_cache=True)
        tok = (
            torch.tensor([c.lens[0] - 1, c.T - 1], device=DEV)
            if decode
            else slice(None)
        )
        rows, ctx = K.sparse_rows(sel[tok], c.bt, c.seq[tok], c.pos[tok])
        q = c.q[tok]
        out = torch.zeros(q.shape[0], H * VD, dtype=DT, device=DEV)
        K.attention_aiter(
            q,
            c.layer.k_cache,
            c.layer.v_cache,
            rows,
            ctx,
            out,
            D**-0.5,
            k_scale=c.ks,
            v_scale=c.vs,
            fp8_compute=fp8_compute,
        )
        ref = ref[tok].float()
        err = ((out.float() - ref).norm() / ref.norm()).item()
        assert err < (0.05 if fp8_compute else 0.01), err

    def test_long_dense_context_is_partitioned(self, monkeypatch):
        c = Case([40 * PG + 5], topk=4)
        c.q, c.iq = c.run_prep()
        nb = c.bt.shape[1]
        got, call = self._run(c, None, monkeypatch, nb)
        assert call["partitions"] == 3  # 41 pages: 656 tokens in 256-token partitions
        k_all, ik_all = c.cached()
        ref = c.impl._attend_prefill_seq(
            c.q,
            c.iq,
            torch.arange(c.T, device=DEV),
            k_all,
            ik_all,
            ProVariants(dense_only=True),
        )
        assert _close(got, ref)


@needs_fp8
class TestFp8:
    """An fp8 KV cache: ``prep`` quantizes K and V per token (a scale each), the Triton
    attention dequantizes K (V its tail, on K's scale), aiter's reads V from the V cache.
    """

    def test_scales_and_rounding(self, case8):
        c = case8
        blk, off = c.slot // PG, c.slot % PG
        rk = c.rk[:, 0].float()
        rv = rk[:, D - VD :]
        ks, vs = c.ks[blk, 0, off], c.vs[blk, 0, off]
        # the reference and the kernel round the activations apart (as ``_close`` allows)
        rtol = 1e-2 if BF16 else 1e-4
        assert torch.allclose(ks * 448, rk.abs().amax(-1), rtol=rtol, atol=ATOL)
        assert torch.allclose(vs * 448, rv.abs().amax(-1), rtol=rtol, atol=ATOL)
        # e4m3 keeps 3 mantissa bits: within 1/16 of the value, or a subnormal step
        k, v = c.cached()[0], c.cached_v()
        assert (
            (k - rk).abs() <= rk.abs() * (1 / 16 + rtol) + ks[:, None] * 2**-9 + ATOL
        ).all()
        assert (
            (v - rv).abs() <= rv.abs() * (1 / 16 + rtol) + vs[:, None] * 2**-9 + ATOL
        ).all()

    def test_index_cache_is_a_plain_cast(self, case8):
        c = case8
        got = c.cached()[1].float()
        rik = c.rik[:, 0].float()
        rtol = 1e-2 if BF16 else 1e-4
        assert ((got - rik).abs() <= rik.abs() * (1 / 16 + rtol) + 2**-9 + ATOL).all()

    def test_selection_reads_the_fp8_index_cache(self, case8):
        # the reference scores the cached (fp8) keys too: the same blocks
        TestScores().test_selected_blocks_match_the_reference(
            case8, K.BLOCK_TOKLSE, {"block_score": "lse", "head_reduce": "lse"}
        )

    @pytest.mark.parametrize("dense", [False, True])
    def test_triton_attention_dequantizes_k(self, case8, dense):
        att = TestAttention()
        got = att._run(
            case8, None, 2, dense=dense, n_blocks=case8.bt.shape[1] if dense else None
        )
        assert _close(got, att._ref(case8, dense=dense))

    @pytest.mark.parametrize("fp8_compute", [True, False])
    def test_aiter_attention_reads_the_v_cache(self, case8, fp8_compute, monkeypatch):
        got, call = TestAttentionAiter()._run(
            case8, _tokmax_sel(case8), monkeypatch, fp8_compute=fp8_compute
        )
        assert call["k_scale"] == (case8.kc4.shape[0], 1, PG, 1)
        assert call["compute_type"] == (FP8 if fp8_compute else DT)
        assert _close(got, TestAttention()._ref(case8, dense=False, v_cache=True))


class TestUnwrittenSlots:
    """A cache slot nobody wrote (the tail of a sequence's last block, blocks not in use) may
    hold anything -- NaN included: nothing the kernels compute may depend on it."""

    @pytest.mark.parametrize("mode", [K.BLOCK_TOKMAX, K.BLOCK_POOL4MAX, K.BLOCK_TOKLSE])
    @pytest.mark.parametrize("dense", [False, True])
    def test_nan_in_unwritten_slots_does_not_leak(self, mode, dense):
        c = Case([150, 97], seed=5)
        c.kc4.fill_(float("nan"))
        c.ic.fill_(float("nan"))
        c.q, c.iq = c.run_prep()
        nb = c.bt.shape[1]
        cq = c.impl.index_compress_query.data if mode else None
        sc = K.scores(
            c.iq,
            c.ic,
            c.bt,
            c.seq,
            c.pos,
            nb,
            cq,
            c.init,
            c.local,
            1.0,
            block_mode=mode,
        )
        assert not torch.isnan(sc).any()
        sel = None
        if not dense:
            sel = sc.topk(min(c.topk, nb), dim=-1, sorted=False).indices.to(torch.int32)
        out = torch.zeros(c.T, H * VD, dtype=DT, device=DEV)
        K.attention(
            c.q, c.kc4, c.bt, c.seq, c.pos, sel, out, D**-0.5, 2, nblk=4, n_blocks=nb
        )
        assert not torch.isnan(out).any()
        if mode == K.BLOCK_TOKMAX:
            assert _close(out, TestAttention()._ref(c, dense=dense))


class TestImplGlue:
    """``ProSparseAttentionImpl._kernel_attention`` -- prep, selection and attention as the
    model calls them, with the YaRN magnitude plumbing -- against the torch reference.
    """

    MS = 1.3  # a stand-in for aiter's mscale baked into its cos / sin tables

    def _impl_with_rope(self, c, v):
        m = c.impl
        cs = lambda t: (t * self.MS)[:, None, :]
        m.rotary_emb = SimpleNamespace(
            cos_cache=cs(c.cos), sin_cache=cs(c.sin), mscale=self.MS
        )
        mk = lambda w: SimpleNamespace(
            weight=SimpleNamespace(data=w), variance_epsilon=c.eps
        )
        m.q_norm, m.k_norm = mk(c.wq), mk(c.wk)
        m.index_q_norm, m.index_k_norm = mk(c.wiq), mk(c.wik)
        return m

    def _independent_prep(self, c, v):
        """Reference q / index-q with the variant's cos / sin (the YaRN magnitude)."""
        cos, sin = c.impl._cos_sin(c.pos, v)
        t = c.T
        q = c.qkv[:, : H * D].reshape(t, H, D)
        iq = c.qkv[:, H * D + D : H * D + D + NI * DI].reshape(t, NI, DI)
        nrm = lambda w, x: GemmaRMSNorm.forward_static(w, c.eps, x, None)
        r = c.impl._rope
        return r(nrm(c.wq, q), cos, sin), r(nrm(c.wiq, iq), cos, sin)

    @pytest.mark.parametrize(
        "variant,kw",
        [
            ({}, {}),  # the defaults: pool4 block score, max over heads
            (TOKMAX, {}),
            ({"dense_only": True}, {}),
            ({"block_score": "pool4", "head_reduce": "lse", "index_scale": 0.5}, {}),
            ({"block_score": "lse"}, {}),
            ({"block_score": "lse", "head_reduce": "lse", "index_scale": 1.5}, {}),
            ({"head_reduce": "max", "init_blocks": 2, "local_blocks": 3}, {}),
            ({"mscale": "aiter"}, {}),
            ({"mscale": "1.1"}, {}),
            ({}, {"attn_splits": 3, "attn_nblk": 4}),
            ({}, {"attn": "aiter"}),
            (TOKMAX, {"attn": "aiter"}),
            ({"dense_only": True}, {"attn": "aiter"}),
            ({}, {"topk": "aiter"}),
            ({}, {"topk": "aiter", "attn": "aiter"}),
        ],
    )
    def test_matches_the_torch_reference(self, variant, kw, monkeypatch):
        # topk 6 > the 5 blocks init 2 + local 3 force, so scores still decide a slot
        c = Case([150, 97], topk=6, seed=3)
        v, seen, out = self._run(c, variant, kw, monkeypatch)
        # (1) the variant's rope magnitude reached the kernel: prep matches the torch rope
        rq, riq = self._independent_prep(c, v)
        assert _close(seen["q"], rq) and _close(seen["iq"], riq)
        # (2) selection + attention equal the torch reference on the very same inputs
        assert _close(out, self._reference(c, v, seen))

    @pytest.mark.parametrize(
        "variant",
        [
            {},
            TOKMAX,
            {"block_score": "lse", "head_reduce": "lse"},
            {"init_blocks": 2, "local_blocks": 3},
        ],
    )
    def test_fused_selection_matches_the_torch_reference(self, variant, monkeypatch):
        # select_rows takes a power-of-two k: 8 > the 5 blocks init 2 + local 3 force
        c = Case([150, 97], topk=8, seed=3)
        for topk in ("fused", "split"):
            v, seen, out = self._run(
                c, variant, {"topk": topk, "attn": "aiter"}, monkeypatch
            )
            assert _close(out, self._reference(c, v, seen))

    @needs_fp8
    @pytest.mark.parametrize("attn", ["triton", "aiter"])
    def test_fp8_cache_matches_the_torch_reference(self, attn, monkeypatch):
        c = Case([150, 97], topk=6, seed=3, kv_dtype=FP8, ic_dtype=FP8)
        v, seen, out = self._run(c, {}, {"attn": attn}, monkeypatch)
        assert _close(out, self._reference(c, v, seen, v_cache=attn == "aiter"))

    @pytest.mark.parametrize(
        "attn,topk",
        [
            ("triton", "torch"),
            ("aiter", "torch"),
            ("aiter", "aiter"),
            ("aiter", "fused"),
            ("aiter", "split"),
        ],
    )
    def test_static_decode_equals_the_dynamic_path(self, attn, topk, monkeypatch):
        # decode as a graph runs it: the last token of each sequence plus a padding row
        # (slot -1), over a block table wider than any sequence, selection throughout
        c = Case([150, 97], topk=8 if topk in ("fused", "split") else 6, seed=3)
        kw = {"attn": attn, "topk": topk}
        _, _, out = self._run(c, {}, kw, monkeypatch)
        caches = [t.clone() for t in (c.kc4, c.vc4, c.ic)]
        tok = torch.tensor([c.lens[0] - 1, c.T - 1, 0], device=DEV)
        slot = c.slot[tok].clone()
        slot[2] = -1
        seq = torch.tensor([0, 1, 1], device=DEV)
        wide = torch.cat([c.bt, torch.zeros_like(c.bt).repeat(1, 4)], dim=1)
        got = torch.zeros(3, H * VD, dtype=DT, device=DEV)
        c.impl._kernel_attention(
            c.qkv[tok],
            c.pos[tok],
            seq,
            slot,
            wide,
            c.layer,
            c.ic,
            wide.shape[1],
            ProVariants(**kw),
            got,
            static=True,
        )
        assert _close(got[:2], out[tok[:2]])
        assert all(torch.equal(a, b) for a, b in zip(caches, (c.kc4, c.vc4, c.ic)))

    @pytest.mark.parametrize("topk", ["fused", "split"])
    def test_decode_split_over_ranks_equals_one_rank(self, topk, monkeypatch):
        # index_split_decode: each of 4 ranks selects its contiguous quarter of 8 decode
        # rows; the all-gathered [page table | ctx] rows give every rank the same output as
        # one rank selecting all of them
        # the conftest stands torch.distributed in for aiter.dist
        comm = SimpleNamespace(tensor_model_parallel_all_gather=None)
        monkeypatch.setitem(sys.modules, "aiter.dist.communication_op", comm)
        c = Case([150, 97], topk=8, seed=3)
        self._run(c, {}, {"attn": "aiter", "topk": topk}, monkeypatch)
        tok = torch.tensor(
            [
                c.lens[0] - 1,
                c.T - 1,
                140,
                c.lens[0] + 90,
                100,
                c.lens[0] + 60,
                130,
                149,
            ],
            device=DEV,
        )
        seq = (tok >= c.lens[0]).long()
        wide = torch.cat([c.bt, torch.zeros_like(c.bt).repeat(1, 4)], dim=1)
        v = ProVariants(attn="aiter", topk=topk, index_split_decode=8)

        def run(dsplit):
            got = torch.zeros(len(tok), H * VD, dtype=DT, device=DEV)
            c.impl._kernel_attention(
                c.qkv[tok],
                c.pos[tok],
                seq,
                c.slot[tok],
                wide,
                c.layer,
                c.ic,
                wide.shape[1],
                v,
                got,
                static=True,
                dsplit=dsplit,
            )
            return got

        want = run(None)
        shares, tp = [], 4
        monkeypatch.setattr(
            comm,
            "tensor_model_parallel_all_gather",
            lambda x, dim: (shares.append(x.clone()), x.repeat(tp, 1))[1],
        )
        for r in range(tp):
            run((2 * r, 2 * r + 2))
        mine = list(shares)
        monkeypatch.setattr(
            comm, "tensor_model_parallel_all_gather", lambda x, dim: torch.cat(mine)
        )
        for r in range(tp):
            assert torch.equal(run((2 * r, 2 * r + 2)), want), r

    @pytest.mark.parametrize("attn", ["aiter", "triton"])
    @pytest.mark.parametrize("tq_causal", [False, True])
    @pytest.mark.parametrize("dsplit", [False, True])
    def test_verify_rows_select_as_single_row_decodes(
        self, attn, tq_causal, dsplit, monkeypatch
    ):
        # a DSpark verify (k = 7: 8 rows a sequence, score_tq 8 in the graph) must select,
        # row by row, the blocks a plain decode of that one row would: its own causal
        # top-k with the sink and local blocks of its own position. Two of the sequences'
        # 8 rows straddle a block boundary. The attention is replaced by a recorder of
        # what it is given (aiter: the page tables and lengths; Triton: the block ids).
        k1, init, local = 8, 1, 9
        c = Case([403, 512, 330], topk=16, seed=11, init=init, local=local)
        base = {"attn": attn, "topk": "split", "score_tq": 8}
        self._run(c, {}, base, monkeypatch)  # fills the K, V and index caches
        seen = []

        def rec_aiter(q, kc, vc, rows, ctx, out, *a, **kw):
            seen.append(torch.cat([rows, ctx[:, None]], dim=1).clone())
            return out

        def rec_triton(q, kc4, bt, seq, pos, sel, out, *a, **kw):
            seen.append(sel.clone())
            return out

        monkeypatch.setattr(K, "attention_aiter", rec_aiter)
        monkeypatch.setattr(K, "attention", rec_triton)
        wide = torch.cat([c.bt, torch.zeros_like(c.bt).repeat(1, 4)], dim=1)
        ends = [sum(c.lens[: s + 1]) for s in range(c.S)]
        tok = torch.cat([torch.arange(e - k1, e) for e in ends]).to(DEV)
        v = ProVariants(
            **base,
            init_blocks=init,
            local_blocks=local,
            score_tq_causal=tq_causal,
            index_split_decode=8 if dsplit else 0,
        )

        def run(rows, table, seq, ds=None):
            seen.clear()
            c.impl._kernel_attention(
                c.qkv[rows],
                c.pos[rows],
                seq,
                c.slot[rows],
                table,
                c.layer,
                c.ic,
                table.shape[1],
                v,
                torch.zeros(len(rows), H * VD, dtype=DT, device=DEV),
                static=True,
                dsplit=ds,
            )
            assert len(seen) == 1
            return seen[0]

        seq = torch.arange(c.S, device=DEV).repeat_interleave(k1)
        if dsplit and attn == "aiter":
            # 3 ranks, a sequence each, through a fake all-gather of their shares
            comm = SimpleNamespace(tensor_model_parallel_all_gather=None)
            monkeypatch.setitem(sys.modules, "aiter.dist.communication_op", comm)
            shares = []
            monkeypatch.setattr(
                comm,
                "tensor_model_parallel_all_gather",
                lambda x, dim: (shares.append(x.clone()), x.repeat(c.S, 1))[1],
            )
            for r in range(c.S):
                run(tok, wide, seq, (k1 * r, k1 * r + k1))
            mine = list(shares)
            monkeypatch.setattr(
                comm, "tensor_model_parallel_all_gather", lambda x, dim: torch.cat(mine)
            )
            verify = run(tok, wide, seq, (0, k1))
        else:
            verify = run(tok, wide, seq)
        for i, t in enumerate(tok.tolist()):
            s = i // k1
            alone = run(
                torch.tensor([t], device=DEV),
                wide[s : s + 1],
                torch.zeros(1, dtype=torch.long, device=DEV),
            )
            assert torch.equal(verify[i], alone[0]), (i, int(c.pos[t]))

    @pytest.mark.parametrize("topk", ["fused", "split"])
    def test_decode_split_logical_over_ranks_equals_one_rank(self, topk, monkeypatch):
        # the block-id branch of the decode split (the day0 decode / Triton attention take
        # select_logical's ids, not page tables): 4 ranks, a contiguous quarter of 8 decode
        # rows each, all-gathered; every rank's attention output equals one rank selecting
        # all of them, bit for bit
        comm = SimpleNamespace(tensor_model_parallel_all_gather=None)
        monkeypatch.setitem(sys.modules, "aiter.dist.communication_op", comm)
        c = Case([150, 97], topk=8, seed=3)
        self._run(c, {}, {"attn": "triton", "topk": topk}, monkeypatch)
        tok = torch.tensor(
            [
                c.lens[0] - 1,
                c.T - 1,
                140,
                c.lens[0] + 90,
                100,
                c.lens[0] + 60,
                130,
                149,
            ],
            device=DEV,
        )
        seq = (tok >= c.lens[0]).long()
        wide = torch.cat([c.bt, torch.zeros_like(c.bt).repeat(1, 4)], dim=1)
        v = ProVariants(attn="triton", topk=topk, index_split_decode=8)

        def run(dsplit):
            got = torch.zeros(len(tok), H * VD, dtype=DT, device=DEV)
            c.impl._kernel_attention(
                c.qkv[tok],
                c.pos[tok],
                seq,
                c.slot[tok],
                wide,
                c.layer,
                c.ic,
                wide.shape[1],
                v,
                got,
                static=True,
                dsplit=dsplit,
            )
            return got

        want = run(None)
        shares, tp = [], 4
        monkeypatch.setattr(
            comm,
            "tensor_model_parallel_all_gather",
            lambda x, dim: (shares.append(x.clone()), x.repeat(tp, 1))[1],
        )
        for r in range(tp):
            run((2 * r, 2 * r + 2))
        assert all(s.shape == (2, 8) for s in shares)  # block ids, not [table | ctx]
        mine = list(shares)
        monkeypatch.setattr(
            comm, "tensor_model_parallel_all_gather", lambda x, dim: torch.cat(mine)
        )
        for r in range(tp):
            assert torch.equal(run((2 * r, 2 * r + 2)), want), r

    def test_decode_split_plan(self, monkeypatch):
        ps = SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: 8,
            get_tensor_model_parallel_rank=lambda: 3,
        )
        monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", ps)
        c = Case([150, 97], topk=8, seed=3)
        v = ProVariants(attn="aiter", topk="split", index_split_decode=64)
        assert c.impl.decode_split(512, v) == (192, 256, 64)
        assert c.impl.decode_split(32, v) is None  # under the threshold
        assert c.impl.decode_split(68, v) == (27, 36, 9)  # ceil(68 / 8) rows a rank
        assert c.impl.decode_split(512, replace(v, index_split_decode=0)) is None
        assert c.impl.decode_split(512, replace(v, topk="torch")) is None
        assert c.impl.decode_split(512, replace(v, attn="triton")) is None
        # whole sequences a rank: 512 rows of 64 sequences (8 each) split, of 128 (4 each)
        # too; 520 rows of 65 sequences: shares of 9 sequences, the last one short; 4
        # sequences of 128 rows: a sequence a rank, ranks 4-7 past the last one
        assert c.impl.decode_split(512, v, 64) == (192, 256, 64)
        assert c.impl.decode_split(512, v, 128) == (192, 256, 64)
        assert c.impl.decode_split(520, v, 65) == (216, 288, 72)
        assert c.impl.decode_split(512, v, 4) == (384, 512, 128)
        # under DCP the index cache is sharded by block: DCP's own split, not this one
        from atom.distributed import dcp_utils

        monkeypatch.setattr(dcp_utils, "get_dcp_world_size", lambda: 8)
        assert c.impl.decode_split(512, v, 64) is None

    @pytest.mark.skipif(not GPU, reason="CUDA graphs need a GPU")
    @pytest.mark.parametrize(
        "attn,topk",
        [
            ("triton", "torch"),
            ("aiter", "torch"),
            ("aiter", "aiter"),
            ("aiter", "fused"),
            ("aiter", "split"),
        ],
    )
    def test_decode_graph_replays_another_batch(self, attn, topk):
        # the static path captured over fixed buffers (a graph's decode), replayed for
        # other sequences at other positions plus a padding row; the token -> sequence
        # map comes from the metadata as in ATOM, whose capture reuses its warmup's
        c = Case(
            [1500, 970, 610, 200], topk=64 if topk in ("fused", "split") else 40, seed=5
        )
        kw = {"attn": attn, "topk": topk}
        v = ProVariants(**kw)
        m = self._impl_with_rope(c, v)
        full = torch.zeros(c.T, H * VD, dtype=DT, device=DEV)
        m._kernel_attention(
            c.qkv, c.pos, c.seq, c.slot, c.bt, c.layer, c.ic, c.bt.shape[1], v, full
        )
        B, W = 4, 256
        qkv = torch.zeros(B, c.qkv.shape[1], dtype=c.qkv.dtype, device=DEV)
        pos = torch.zeros(B, dtype=torch.long, device=DEV)
        slot = torch.full((B,), -1, dtype=torch.long, device=DEV)
        bt = torch.zeros(B, W, dtype=torch.int32, device=DEV)
        md = SimpleNamespace(
            cu_seqlens_q=torch.arange(B + 1, dtype=torch.int32, device=DEV),
            block_tables=bt,
        )
        out = torch.zeros(B, H * VD, dtype=DT, device=DEV)

        def load(toks):
            qkv.zero_(), pos.zero_(), slot.fill_(-1), bt.zero_()
            for r, t in enumerate(toks):
                if t is not None:
                    qkv[r], pos[r], slot[r] = c.qkv[t], c.pos[t], c.slot[t]
                    bt[r, : c.bt.shape[1]] = c.bt[int(c.seq[t])]

        def run(o, seq=None):
            if seq is None:
                seq = ProSparseAttentionImpl._device_meta(md, B, DEV)
            m._kernel_attention(
                qkv, pos, seq, slot, bt, c.layer, c.ic, W, v, o, static=True
            )

        ends = [sum(c.lens[: s + 1]) - 1 for s in range(c.S)]
        load(ends)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                run(out)
        torch.cuda.current_stream().wait_stream(side)
        warm = md._m3pro_dev[1]
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            run(out)
        warm.zero_()  # the warmup's map, freed with its metadata, its memory reused
        toks = [c.lens[0] + 700, ends[0], None, sum(c.lens[:3]) + 150]
        load(toks)
        want = torch.zeros_like(out)
        run(want, seq=torch.arange(B, device=DEV))
        g.replay()
        torch.cuda.synchronize()
        real = [r for r, t in enumerate(toks) if t is not None]
        assert _close(out[real], want[real])
        assert _close(out[real], full[[toks[r] for r in real]])

    def test_padding_rows_map_into_the_block_table(self):
        md = SimpleNamespace(
            cu_seqlens_q=torch.tensor([0, 1, 2], device=DEV),
            block_tables=torch.zeros(2, 5, dtype=torch.int32, device=DEV),
        )
        seq = ProSparseAttentionImpl._device_meta(md, 4, DEV)
        assert seq.tolist() == [0, 1, 1, 1]

    def _run(self, c, variant, kw, monkeypatch):
        v = ProVariants(**variant)
        self._impl_with_rope(c, v)
        seen = {}
        real_prep = K.prep

        def spy(*a, **k):
            seen["q"], seen["iq"] = real_prep(*a, **k)
            return seen["q"], seen["iq"]

        monkeypatch.setattr(K, "prep", spy)
        pa = RefPA()
        monkeypatch.setattr(K, "_pa_decode", pa)
        monkeypatch.setattr(K, "_default_partitions", lambda t: 3)
        monkeypatch.setattr(K, "_topk_rows", ref_topk_rows)
        out = torch.zeros(c.T, H * VD, dtype=DT, device=DEV)
        c.impl._kernel_attention(
            c.qkv,
            c.pos,
            c.seq,
            c.slot,
            c.bt,
            c.layer,
            c.ic,
            c.bt.shape[1],
            ProVariants(**variant, **kw),
            out,
        )
        assert len(pa.calls) == (kw.get("attn") == "aiter")
        return v, seen, out

    def test_day0_takes_block_ids_beside_aiter_k_rows(self, monkeypatch):
        # K-only fp8 cache, aiter reading the K rows: a day0 prefill still gets
        # select_logical's block ids, never aiter's page tables (dispatch only: the
        # interpreter's fp8 rounding does not matter here)
        c = Case([150, 97], topk=8, seed=3, kv_dtype=FP8, k_only=True)
        c.impl.k_only = True
        self._impl_with_rope(c, ProVariants())
        monkeypatch.setattr(K, "TOPK", 8)
        monkeypatch.setattr(K, "aiter_value_rows_available", lambda: True)
        monkeypatch.setattr(K, "day0_prefill_available", lambda page=K.PAGE: True)
        got = {}

        def fake_day0(q, kc4, bt, seq, pos, sel, out, scale):
            got["sel"] = sel
            return out

        def no_aiter(*a, **k):
            pytest.fail("aiter page tables in a day0 prefill")

        monkeypatch.setattr(K, "attention_day0", fake_day0)
        monkeypatch.setattr(K, "attention_aiter", no_aiter)
        kw = {"topk": "fused", "attn": "aiter", "k_only": True}
        kw.update(k_plain_fp8=True, prefill_attn="day0")
        out = torch.zeros(c.T, H * VD, dtype=DT, device=DEV)
        c.impl._kernel_attention(
            c.qkv,
            c.pos,
            c.seq,
            c.slot,
            c.bt,
            c.layer,
            c.ic,
            c.bt.shape[1],
            ProVariants(**kw),
            out,
        )
        sel = got["sel"]
        assert sel is not None and tuple(sel.shape) == (c.T, 8)
        qb = (c.pos // PG).unsqueeze(1).expand_as(sel)
        valid = sel >= 0
        assert (sel[valid] <= qb[valid]).all()
        assert torch.equal(valid.sum(1), torch.clamp(c.pos // PG + 1, max=8))

    def _reference(self, c, v, seen, v_cache=False):
        k_all, ik_all = c.cached()
        v_all = c.cached_v() if v_cache else None
        outs = []
        for s_ in range(c.S):
            a, b = c.seq_slice(s_)
            outs.append(
                c.impl._attend_prefill_seq(
                    seen["q"][a:b],
                    seen["iq"][a:b],
                    torch.arange(c.lens[s_], device=DEV),
                    k_all[a:b],
                    ik_all[a:b],
                    v,
                    None if v_all is None else v_all[a:b],
                )
            )
        return torch.cat(outs)

    def test_use_kernels_gate(self):
        c = Case([20])
        on = ProVariants(kernels=True)
        m = c.impl
        m.num_heads = H
        assert m._use_kernels(on)
        assert m._use_kernels(ProVariants(kernels=True, block_score="lse"))
        assert m._use_kernels(ProVariants(kernels=True, attn="aiter"))
        assert m._use_kernels(ProVariants(kernels=True, attn="aiter", topk="aiter"))
        assert not m._use_kernels(ProVariants())  # off by default
        for bad in (
            {"v_src": "pre"},
            {"rope_style": "gptj"},
            {"index_rope": False},
            {"diag": True},
            {"attn_nblk": 3},
            {"head_reduce": "median"},
            {"block_score": "mean"},
            {"attn": "flash"},
            {"topk": "radix"},
            {"topk": "fused"},
        ):  # fused: page tables for the aiter attention only
            assert not m._use_kernels(ProVariants(kernels=True, **bad)), bad
        m.topk = 8
        assert m._use_kernels(ProVariants(kernels=True, attn="aiter", topk="fused"))
        m.topk = 6
        assert not m._use_kernels(ProVariants(kernels=True, attn="aiter", topk="fused"))


class TestKOnly:
    """The K-only cache (``k_only``): row-major K pages (x = 256), V read from K's tail,
    no V cache; the Triton attention over it, the selection as block ids."""

    @pytest.mark.parametrize("kv", [None, pytest.param(FP8, marks=needs_fp8)])
    def test_prep_writes_the_same_k_as_the_shuffled_cache(self, kv):
        sh = Case([150, 97], seed=3, kv_dtype=kv)
        ko = Case([150, 97], seed=3, kv_dtype=kv, k_only=True)
        q0, iq0 = sh.run_prep()
        q1, iq1 = ko.run_prep()
        assert torch.equal(q1, q0) and torch.equal(iq1, iq0)
        assert torch.equal(ko.ic, sh.ic)
        k0, k1 = sh.cached()[0], ko.cached()[0]
        assert torch.equal(k1, k0)
        if kv is not None:
            assert torch.equal(ko.ks, sh.ks)
        used = torch.zeros(ko.kc4.shape[0], dtype=torch.bool, device=DEV)
        used[ko.slot // PG] = True
        assert not ko.kc4[~used].view(torch.uint8).any()

    @pytest.mark.parametrize("kv", [None, pytest.param(FP8, marks=needs_fp8)])
    @pytest.mark.parametrize("dense", [False, True])
    def test_triton_attention_reads_v_from_the_row_cache(self, kv, dense):
        c = Case([150, 97], kv_dtype=kv, k_only=True)
        c.q, c.iq = c.run_prep()
        att = TestAttention()
        got = att._run(
            c, None, 2, dense=dense, n_blocks=c.bt.shape[1] if dense else None
        )
        assert _close(got, att._ref(c, dense=dense))

    @pytest.mark.parametrize("width,k", [(40, 8), (1024, 128), (100, 128)])
    def test_select_logical_is_select_rows_as_block_ids(self, width, k):
        g = torch.Generator(device=DEV).manual_seed(5)
        t, s = 29, 3
        bt = torch.randperm(s * width, generator=g, device=DEV).view(s, width)
        bt = bt.to(torch.int32) + 1
        seq = torch.randint(0, s, (t,), generator=g, device=DEV)
        pos = torch.randint(0, width * PG, (t,), generator=g, device=DEV)
        pos[:2] = torch.tensor([0, 3 * PG + 1], device=DEV)
        qb = pos // PG
        j = torch.arange(width, device=DEV)
        score = torch.randn(t, width, generator=g, device=DEV)
        score[:, 0] = 1e30
        score = torch.where((j[None, :] >= qb[:, None] - 2), 1e29, score)
        score = torch.where(j[None, :] <= qb[:, None], score, float("-inf"))
        sel = K.select_logical(score, pos, k)
        rows, _ctx = K.select_rows(score, bt, seq, pos, k)
        assert sel.shape == (t, k) and sel.dtype == torch.int32
        for r in range(t):
            n = min(k, int(qb[r]) + 1)
            got = sel[r].tolist()
            assert all(x >= 0 for x in got[:n]) and all(x == -1 for x in got[n:]), r
            phys = {int(bt[seq[r], x]) for x in got[:n]}
            assert phys == set(rows[r, :n].tolist()), r

    def test_pool_holds_k_and_one_scale_plane(self):
        from atom.model_ops.attentions.mha_kv_pool import MhaKvPool

        kw = {
            "layers": 2,
            "block_size": PG,
            "num_kv_heads": 1,
            "head_dim": D,
            "kv_dtype": FP8,
        }
        shuffle = MhaKvPool(**kw)
        konly = MhaKvPool(**kw, layout=ProKOnlyLayout(FP8))
        assert [f.name for f in konly.cache_fields + konly.scale_fields] == [
            "k",
            "k_scale",
        ]
        assert konly.entry_bytes < shuffle.entry_bytes * 0.52
        konly.allocate(5, "cpu")
        k, v = konly.kv_views(1)
        assert tuple(k.shape) == (5, 1, 1, PG, D) and v is k
        assert k[:, 0, 0].stride()[1:] == (
            D,
            1,
        )  # rows of 256 bytes, as day0 / moo read
        ks, vs = konly.scale_views(1)
        assert tuple(ks.shape) == (5, 1, PG) and vs is ks

    @needs_fp8
    def test_plain_fp8_k_is_the_clamped_cast(self):
        c = Case([150, 97], seed=4, kv_dtype=FP8, k_only=True)
        c.k_unit = True
        c.q, c.iq = c.run_prep()
        rk = c.ref_prep()[1][:, 0]
        blk, off = c.slot // PG, c.slot % PG
        assert torch.equal(c.ks[blk, 0, off], torch.ones_like(c.ks[blk, 0, off]))
        want = K.quant_fp8_rows(c.cached()[0], FP8, plain=True)[
            0
        ]  # a fixed point of it
        assert torch.equal(c.kc4[blk, 0, off], want)
        rtol = 1e-2 if BF16 else 1e-4
        got = c.cached()[0]
        assert (
            (got - rk.float()).abs()
            <= rk.float().abs() * (1 / 16 + rtol) + 2**-9 + ATOL
        ).all()

    @pytest.mark.skipif(
        not (GPU and K.day0_prefill_available()),
        reason="aiter's FlyDSL fp8 sparse prefill on a GPU",
    )
    @pytest.mark.parametrize("dense", [False, True])
    def test_day0_prefill_matches_the_triton_attention(self, dense):
        c = Case([300, 197], topk=K.TOPK, seed=6, kv_dtype=FP8, k_only=True)
        c.k_unit = True
        c.q, c.iq = c.run_prep()
        nb = c.bt.shape[1]
        sel = None
        if not dense:
            sc = K.scores(
                c.iq, c.ic, c.bt, c.seq, c.pos, nb, None, c.init, c.local, 1.0
            )
            sel = K.select_logical(sc, c.pos, K.TOPK)
        q16 = c.q[:, :16].contiguous()
        ref = torch.zeros(c.T, 16 * VD, dtype=DT, device=DEV)
        # the reference sees what day0 does: q cast to fp8
        K.attention(
            q16.to(FP8).to(DT),
            c.kc4,
            c.bt,
            c.seq,
            c.pos,
            sel,
            ref,
            D**-0.5,
            2,
            n_blocks=nb if dense else None,
            k_scale=c.ks,
        )
        got = torch.zeros_like(ref)
        K.attention_day0(q16, c.kc4, c.bt, c.seq, c.pos, sel, got, D**-0.5)
        assert _close(got, ref)


def _fake_flydsl(monkeypatch, **exports):
    """Stand in for ``aiter.ops.flydsl`` with ``exports`` and make ``amd_kernels``
    unimportable, as in the run environment."""
    import sys
    import types

    mod = types.ModuleType("aiter.ops.flydsl")
    for name, fn in exports.items():
        setattr(mod, name, fn)
    monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", mod)
    monkeypatch.setitem(sys.modules, "amd_kernels", None)
    monkeypatch.setattr(K, "_day0_prefill", None)


def test_day0_prefill_is_aiters_flydsl_kernel(monkeypatch):
    # prefill_attn="day0" runs aiter's minimax_m3_fp8_sparse_prefill: the fp8 q, the
    # K-only cache's page plane, the dense causal picks, the token metadata as int32
    calls = []
    _fake_flydsl(
        monkeypatch,
        minimax_m3_fp8_sparse_prefill=lambda *a: calls.append(a),
        minimax_m3_fp8_sparse_prefill_supported=lambda *a: True,
    )
    assert K.day0_prefill_available()
    t, pg = 5, 64
    q = torch.randn(t, 16, 256)
    kc4 = torch.zeros(3, 1, pg, 256, dtype=FP8)
    bt = torch.tensor([[2, 0]], dtype=torch.int32)
    seq = torch.zeros(t, dtype=torch.int64)
    pos = torch.tensor([0, 15, 16, 40, 70])
    out = torch.empty(t, 16 * 192, dtype=torch.bfloat16)
    assert K.attention_day0(q, kc4, bt, seq, pos, None, out, 0.0625) is out
    (args,) = calls
    qa, ka, sel, bta, seqa, posa, scale, outa = args
    assert qa.dtype == FP8 and torch.equal(qa.float(), q.to(FP8).float())
    assert ka.shape == (3, pg, 256) and ka.data_ptr() == kc4.data_ptr()
    assert sel.dtype == torch.int32 and sel.shape == (t, K.TOPK)
    want = torch.full((t, K.TOPK), -1, dtype=torch.int32)
    for i, p in enumerate(pos.tolist()):
        want[i, : p // K.PAGE + 1] = torch.arange(p // K.PAGE + 1)
    assert torch.equal(sel, want)
    assert bta is bt and seqa.dtype == posa.dtype == torch.int32
    assert torch.equal(posa, pos.int()) and scale == 0.0625
    assert outa.shape == (t, 16, 192) and outa.data_ptr() == out.data_ptr()


def test_day0_prefill_needs_the_aiter_export(monkeypatch):
    # an aiter without the port: "day0" falls back (Triton), never to amd_kernels
    _fake_flydsl(monkeypatch)
    assert not K.day0_prefill_available()


def test_day0_prefill_asks_aiter_whether_it_serves(monkeypatch):
    # aiter's supported() decides (gfx950, the layer's shape, the cache's page size):
    # where it says no, "day0" falls back to Triton
    asked = []

    def supported(*a):
        asked.append(a)
        return a[-1] == 64

    _fake_flydsl(
        monkeypatch,
        minimax_m3_fp8_sparse_prefill=lambda *a: None,
        minimax_m3_fp8_sparse_prefill_supported=supported,
    )
    assert K.day0_prefill_available(64)
    assert not K.day0_prefill_available(24)
    assert asked == [(16, 1, 256, 192, K.TOPK, K.PAGE, p) for p in (64, 24)]


def _fake_flydsl_decode(monkeypatch, **exports):
    """``_fake_flydsl`` with the decode's loader and warm-up record reset."""
    _fake_flydsl(monkeypatch, **exports)
    monkeypatch.setattr(K, "_day0_decode", None)
    monkeypatch.setattr(K, "_day0_decode_warmed", set(), raising=False)


def _decode_exports(calls, warms, supported=lambda *a: True):
    return {
        "minimax_m3_fp8_sparse_decode": lambda *a: calls.append(a),
        "minimax_m3_fp8_sparse_decode_warm_up": lambda *a: warms.append(a),
        "minimax_m3_fp8_sparse_prefill_supported": supported,
    }


def test_day0_decode_is_aiters_flydsl_kernel(monkeypatch):
    # day0_decode_rows runs aiter's minimax_m3_fp8_sparse_decode: the fp8 q, the K-only
    # cache's page plane, the picks, each request's length, qlen and the MXFP8 output
    calls = []
    _fake_flydsl_decode(monkeypatch, **_decode_exports(calls, []))
    assert K.day0_decode_available()
    qlen, reqs, pg = 2, 3, 32
    t = qlen * reqs
    q = torch.randn(t, 16, 256)
    kc4 = torch.zeros(4, 1, pg, 256, dtype=FP8)
    bt = torch.tensor([[1, 3], [0, 2], [0, 0]], dtype=torch.int32)
    pos = torch.tensor([40, 41, 7, 8, 0, 1])
    slots = torch.tensor([5, 6, 7, 8, -1, -1])  # request 2: a graph's padding request
    sel = torch.full((t, K.TOPK), -1, dtype=torch.int32)
    out = torch.empty(t, 16 * 192, dtype=FP8)
    scale = torch.empty(t, 16 * 192 // 32, dtype=torch.uint8)
    got = K.attention_day0_decode(
        q, kc4, bt, pos, slots, sel, out, 0.0625, qlen, out_scale=scale
    )
    assert got is out
    (args,) = calls
    qa, ka, sela, bta, seq_lens, qla, sc, outa, sca = args
    assert qa.dtype == FP8 and torch.equal(qa.float(), q.to(FP8).float())
    assert ka.shape == (4, pg, 256) and ka.data_ptr() == kc4.data_ptr()
    assert sela is sel and bta is bt and qla == qlen and sc == 0.0625
    assert seq_lens.dtype == torch.int32 and seq_lens.tolist() == [42, 9, 0]
    assert outa.shape == (t, 16, 192) and outa.data_ptr() == out.data_ptr()
    assert sca is scale


def test_day0_decode_needs_the_aiter_export(monkeypatch):
    # an aiter with the prefill port only, or none: the decode keeps the other attention,
    # never amd_kernels, and the forward's warm-up does nothing
    for exports in (
        {"minimax_m3_fp8_sparse_prefill_supported": lambda *a: True},
        {},
    ):
        _fake_flydsl_decode(monkeypatch, **exports)
        assert not K.day0_decode_available()
        K.day0_decode_warm_up("cuda:0", K.PAGE, True)
        assert K._day0_decode_warmed == set()


def test_day0_decode_asks_aiter_whether_it_serves(monkeypatch):
    # aiter's check decides (gfx950, the layer's shape): where it says no, the decode
    # keeps the other attention
    asked = []

    def supported(*a):
        asked.append(a)
        return False

    _fake_flydsl_decode(monkeypatch, **_decode_exports([], [], supported))
    assert not K.day0_decode_available()
    assert asked == [(16, 1, 256, 192, K.TOPK, K.PAGE, K.PAGE)]


def test_day0_decode_warms_up_once_outside_a_capture(monkeypatch):
    # aiter's warm-up compiles each (device, page, output) once, never inside a graph
    # capture and never for a CPU tensor
    warms = []
    _fake_flydsl_decode(monkeypatch, **_decode_exports([], warms))
    capturing = [False]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing[0])
    dev = torch.device("cuda", 0)
    K.day0_decode_warm_up(dev, 16, True)
    K.day0_decode_warm_up(dev, 16, True)
    K.day0_decode_warm_up("cpu", 16, False)
    capturing[0] = True
    K.day0_decode_warm_up(dev, 16, False)
    capturing[0] = False
    K.day0_decode_warm_up(dev, 16, False)
    assert warms == [(dev, 16, True), (dev, 16, False)]


def _nv_pages(rows):
    """``[N, PG, 72]`` NVFP4 rows -> the ``[N, PG * 72]`` pages ``prep`` writes: all the
    tokens' 64 data bytes, then all their 8 scale bytes."""
    n = rows.shape[0]
    return torch.cat([rows[..., :64].reshape(n, -1), rows[..., 64:].reshape(n, -1)], 1)


class TestNvfp4Index:
    """``--index-cache-dtype nvfp4`` for Pro: [page x 64 B e2m1 | page x 8 B e4m3] a block,
    written by ``prep`` as vLLM's ``_quant4``, read by the tiled scorer."""

    def test_reference_round_trip_error(self):
        g = torch.Generator().manual_seed(11)
        x = (torch.randn(64, DI, generator=g) * 3).to(torch.bfloat16)
        d = K.nvfp4_dequant_rows(K.nvfp4_rows(x)).reshape(64, 8, 16)
        xg = x.float().reshape(64, 8, 16)
        step = xg.abs().amax(-1, keepdim=True) / 6  # the e2m1 grid's unit, up to e4m3
        assert (
            (d - xg).abs() <= step * 1.13 + 1e-6
        ).all()  # half of the widest gap (2)

    @needs_fp8
    def test_prep_writes_the_reference_rows(self):
        ref = Case([150, 97], seed=8)
        ref.run_prep()
        nv = Case([150, 97], seed=8)
        nv.ic = torch.zeros(nv.ic.shape[0], PG * 72, dtype=torch.uint8, device=DEV)
        nv.run_prep()
        want = _nv_pages(K.nvfp4_rows(ref.ic))
        used = torch.zeros(nv.ic.shape[0], dtype=torch.bool, device=DEV)
        used[nv.slot // PG] = True
        # unwritten slots of a used page stay zero in both
        assert torch.equal(nv.ic[used], want[used]) and not nv.ic[~used].any()

    @pytest.mark.parametrize(
        "mode,var",
        [
            (K.BLOCK_POOL4MAX, {}),
            (K.BLOCK_TOKLSE, {"block_score": "lse", "head_reduce": "lse"}),
        ],
    )
    def test_scorer_reads_the_pages_as_their_dequantized_keys(self, case, mode, var):
        c = case
        rows = K.nvfp4_rows(c.ic)
        deq = K.nvfp4_dequant_rows(rows).to(c.ic.dtype)
        nb = c.bt.shape[1]
        head = K.HEAD_LSE if var else K.HEAD_MAX
        sc = DI**-0.5 if var else 1.0
        cq = c.impl.index_compress_query.data
        args = (c.bt, c.seq, c.pos, nb, cq, c.init, c.local, sc)
        kw = {"block_mode": mode, "head_mode": head, "pool_temp": DI**-0.5}
        want = K.scores(c.iq, deq, *args, **kw)
        got = K.scores(c.iq, _nv_pages(rows), *args, **kw)
        assert torch.allclose(got, want, rtol=1e-4, atol=1e-4)


class TestValidFirst:
    """``attention(valid_first=True)`` over ``select_logical``'s rows (valid blocks first):
    the splits share each row's valid blocks; the same output as the reference."""

    @pytest.mark.parametrize("splits", [1, 3, 8])
    def test_matches_the_reference(self, case, splits):
        c = case
        nb = c.bt.shape[1]
        sc = K.scores(c.iq, c.ic, c.bt, c.seq, c.pos, nb, None, c.init, c.local, 1.0)
        k = 1 << (min(c.topk, nb) - 1).bit_length() >> 1 or 1
        sel = K.select_logical(sc, c.pos, k)
        out = torch.zeros(c.T, H * VD, dtype=DT, device=DEV)
        K.attention(
            c.q,
            c.kc4,
            c.bt,
            c.seq,
            c.pos,
            sel,
            out,
            D**-0.5,
            splits,
            nblk=2,
            k_scale=c.ks,
            valid_first=True,
        )
        ref = torch.zeros_like(out)
        K.attention(
            c.q, c.kc4, c.bt, c.seq, c.pos, sel, ref, D**-0.5, 1, nblk=2, k_scale=c.ks
        )
        assert _close(out, ref)


class TestDay0Decode:
    """amd-kernels-day0's FlyDSL sparse decode (``day0_decode_rows``): decode / verify batches
    from 24 rows on the K-only fp8 cache, select_logical's block ids, and its MXFP8 epilogue
    handed to o_proj (dispatch only: the kernel itself is the package's, tested there).
    """

    def test_wrapper_lengths_and_arguments(self, monkeypatch):
        got = {}

        def fake(q, k_cache, topk, bt, seq_lens, qlen, scale, out, out_scale):
            got.update(
                q=q, k=k_cache, seq_lens=seq_lens, qlen=qlen, out=out, s=out_scale
            )

        monkeypatch.setattr(K, "_day0_decode", fake)
        qlen, reqs = 4, 3
        t = qlen * reqs
        pos = torch.cat([torch.arange(10, 14), torch.arange(50, 54), torch.arange(4)])
        slots = torch.arange(t)
        slots[8:] = -1  # request 2: a graph's padding request
        kc4 = torch.zeros(5, 1, PG, D, dtype=FP8)
        out = torch.empty(t, H * VD, dtype=FP8)
        scale = torch.empty(t, H * VD // 32, dtype=torch.uint8)
        q = torch.randn(t, H, D)
        K.attention_day0_decode(
            q, kc4, None, pos, slots, None, out, 0.1, qlen, out_scale=scale
        )
        assert got["seq_lens"].tolist() == [14, 54, 0]
        assert got["seq_lens"].dtype == torch.int32 and got["qlen"] == qlen
        assert got["q"].dtype == FP8 and got["k"].shape == (5, PG, D)
        assert got["out"].shape == (t, H, VD) and got["s"] is scale

    def _verify_case(self):
        lens = [150, 97, 60, 80, 120, 70]
        c = Case(lens, topk=8, seed=3, kv_dtype=FP8, k_only=True)
        c.impl.k_only = True
        TestImplGlue()._impl_with_rope(c, ProVariants())
        qlen = 4  # a verify of k + 1 = 4 rows a request
        tok = torch.cat(
            [
                torch.arange(c.seq_slice(s)[1] - qlen, c.seq_slice(s)[1])
                for s in range(c.S)
            ]
        ).to(DEV)
        seq = torch.arange(c.S, device=DEV).repeat_interleave(qlen)
        return c, tok, seq, qlen

    @pytest.mark.parametrize("rows_min,day0", [(24, True), (25, False)])
    def test_verify_batch_dispatch(self, monkeypatch, rows_min, day0):
        c, tok, seq, qlen = self._verify_case()
        monkeypatch.setattr(K, "TOPK", 8)
        monkeypatch.setattr(K, "aiter_value_rows_available", lambda: True)
        monkeypatch.setattr(K, "day0_decode_available", lambda: True)
        got = {}

        def fake_day0(q, kc4, bt, pos, slots, sel, out, scale, ql, out_scale=None):
            got.update(sel=sel, qlen=ql, out_scale=out_scale)
            return out

        def fake_aiter(*a, **k):
            got["aiter"] = True
            return a[5]

        monkeypatch.setattr(K, "attention_day0_decode", fake_day0)
        monkeypatch.setattr(K, "attention_aiter", fake_aiter)
        kw = {"topk": "split", "attn": "aiter", "k_only": True, "k_plain_fp8": True}
        v = ProVariants(**kw, day0_decode_rows=rows_min)
        n = tok.numel()
        wide = torch.cat([c.bt, torch.zeros_like(c.bt).repeat(1, 3)], dim=1)
        mx = (
            torch.empty(n, H * VD // 32, dtype=torch.uint8, device=DEV)
            if day0
            else None
        )
        out = torch.zeros(n, H * VD, dtype=FP8 if day0 else DT, device=DEV)
        c.impl._kernel_attention(
            c.qkv[tok], c.pos[tok], seq, c.slot[tok], wide, c.layer, c.ic,
            wide.shape[1], v, out, static=True, mx_scale=mx,
        )  # fmt: skip
        assert ("sel" in got) == day0 and ("aiter" in got) != day0
        if day0:
            sel, pos = got["sel"], c.pos[tok]
            assert got["qlen"] == qlen and got["out_scale"] is mx
            assert tuple(sel.shape) == (n, 8) and sel.dtype == torch.int32
            valid = sel >= 0
            qb = (pos // PG).unsqueeze(1).expand_as(sel)
            assert (sel[valid] <= qb[valid]).all()
            assert torch.equal(valid.sum(1), torch.clamp(pos // PG + 1, max=8))
            # valid first, -1 after
            assert (valid.int().diff(dim=1) <= 0).all()

    def _impl(self, mx_out):
        m = object.__new__(ProSparseAttentionImpl)
        m.num_heads, m.head_dim, m.value_head_dim, m.layer_num = H, D, VD, 1
        m.k_only, m.mx_out, m.index_cache = True, mx_out, None
        return m

    @pytest.mark.parametrize("mx_out", [True, False])
    def test_forward_returns_mxfp8_for_o_proj(self, monkeypatch, mx_out):
        n_tok, n = 32, 24  # 8 padding rows past the batch
        m = self._impl(mx_out)
        v = ProVariants(kernels=True, k_only=True, k_plain_fp8=True, topk="split")
        monkeypatch.setattr(pro_attn, "variants", lambda: v)
        monkeypatch.setattr(m, "_use_kernels", lambda v: True, raising=False)
        monkeypatch.setattr(
            m, "_day0_decode", lambda v, n, reqs, fp8: True, raising=False
        )
        monkeypatch.setattr(K, "day0_decode_available", lambda: True)
        md = SimpleNamespace(
            slot_mapping=torch.zeros(n, dtype=torch.long),
            block_tables=torch.zeros(6, 16, dtype=torch.int32),
            cu_seqlens_q=torch.arange(0, 25, 4, dtype=torch.int32),
        )
        layer = SimpleNamespace(k_cache=torch.zeros(1, 1, 1, PG, D, dtype=FP8))
        fwd = SimpleNamespace(
            context=SimpleNamespace(is_dummy_run=False, is_prefill=False),
            attn_metadata=md,
            kv_cache_data={"layer_1": layer},
        )
        monkeypatch.setattr(pro_attn, "get_forward_context", lambda: fwd)

        def fake_kernel(parts, pos, seq, slots, bt, layer, ic, nb, v, out, **kw):
            assert out.shape[0] == n
            out.view(torch.uint8).fill_(0x38)  # 1.0 in e4m3 (bf16 bytes otherwise)
            if kw["mx_scale"] is not None:
                kw["mx_scale"].fill_(127)

        monkeypatch.setattr(m, "_kernel_attention", fake_kernel, raising=False)
        q = torch.zeros(n_tok, H * D)
        out = m.forward(q, q, q, position=torch.zeros(n_tok, dtype=torch.long), qkv=q)
        assert out.shape == (n_tok, H * VD)
        if not mx_out:
            assert out.dtype == q.dtype and m.take_mx_scale() is None
            return
        assert out.dtype == FP8
        assert (out[:n].float() == 1.0).all() and (out[n:].view(torch.uint8) == 0).all()
        scale = m.take_mx_scale()
        assert scale.dtype == torch.float8_e8m0fnu and scale.shape == (
            n_tok,
            H * VD // 32,
        )
        su = scale.view(torch.uint8)
        assert (su[:n] == 127).all() and (su[n:] == 0).all()
        assert m.take_mx_scale() is None  # taken once

    @pytest.mark.parametrize("day0", [True, False])
    def test_forward_warms_the_decode_up_before_it_runs(self, monkeypatch, day0):
        # the batch the day0 decode takes first compiles its kernel (page and output of
        # this cache and o_proj), ahead of the attention; other batches warm nothing
        n = 24
        m = self._impl(True)
        v = ProVariants(kernels=True, k_only=True, k_plain_fp8=True, topk="split")
        monkeypatch.setattr(pro_attn, "variants", lambda: v)
        monkeypatch.setattr(m, "_use_kernels", lambda v: True, raising=False)
        monkeypatch.setattr(
            m, "_day0_decode", lambda v, n, reqs, fp8: day0, raising=False
        )
        monkeypatch.setattr(K, "day0_decode_available", lambda: day0)
        events = []
        monkeypatch.setattr(
            K, "day0_decode_warm_up", lambda *a: events.append(("warm", *a))
        )
        md = SimpleNamespace(
            slot_mapping=torch.zeros(n, dtype=torch.long),
            block_tables=torch.zeros(6, 16, dtype=torch.int32),
            cu_seqlens_q=torch.arange(0, 25, 4, dtype=torch.int32),
        )
        layer = SimpleNamespace(k_cache=torch.zeros(1, 1, 1, 32, D, dtype=FP8))
        fwd = SimpleNamespace(
            context=SimpleNamespace(is_dummy_run=False, is_prefill=False),
            attn_metadata=md,
            kv_cache_data={"layer_1": layer},
        )
        monkeypatch.setattr(pro_attn, "get_forward_context", lambda: fwd)
        monkeypatch.setattr(
            m,
            "_kernel_attention",
            lambda *a, **kw: events.append(("attend", kw["day0_lens"])),
            raising=False,
        )
        q = torch.zeros(n, H * D)
        pos = torch.arange(n) + 100
        m.forward(q, q, q, position=pos, qkv=q)
        m.forward(q, q, q, position=pos, qkv=q)  # the next layer, same forward
        if not day0:
            assert events == [("attend", None)] * 2
            return
        w, a1, _, a2 = events
        assert w == ("warm", q.device, 32, True)
        # 6 requests of 4 rows: each request's last position + 1, once for both layers
        assert a1[1].tolist() == [104 + 4 * r for r in range(6)] and a2[1] is a1[1]
        assert a1[1].dtype == torch.int32
