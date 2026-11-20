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

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

pytest.importorskip("triton")
pytest.importorskip("aiter", reason="pro_attn imports the ATOM attention base")

from types import SimpleNamespace

import triton
import triton.language as tl

from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.minimax_m3 import pro_kernels as K
from atom.model_ops.minimax_m3.pro_attn import (
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
        self, seq_lens, topk=4, init=1, local=2, seed=0, kv_dtype=None, ic_dtype=None
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
        self.x = 16 if self.fp8 else X  # ATOM's shuffle: 16 bytes
        self.kc4 = torch.zeros(total, D // self.x, PG, self.x, dtype=kvdt, device=DEV)
        self.vc4 = torch.zeros(total, PG // self.x, D, self.x, dtype=kvdt, device=DEV)
        scales = lambda: torch.zeros(total, 1, PG, device=DEV) if self.fp8 else None
        self.ks, self.vs = scales(), scales()
        self.layer = SimpleNamespace(
            k_cache=self.kc4.unsqueeze(1),
            v_cache=self.vc4.unsqueeze(1),
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

    @pytest.mark.parametrize("width,k", [(40, 8), (1024, 128), (100, 128)])
    def test_against_topk_and_the_rows_rule(self, width, k):
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
        rows, ctx = K.select_rows(score, bt, seq, pos, k)
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

    def test_ties_go_to_the_lower_block(self):
        score = torch.zeros(1, 16, device=DEV)
        score[0, 15] = 1.0
        bt = torch.arange(16, dtype=torch.int32, device=DEV)[None] + 100
        pos = torch.tensor([15 * PG], device=DEV)
        rows, ctx = K.select_rows(
            score, bt, torch.zeros(1, dtype=torch.long, device=DEV), pos, 4
        )
        assert sorted(rows[0, :3].tolist()) == [100, 101, 102] and rows[0, 3] == 115
        assert ctx.tolist() == [3 * PG + 1]


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
        v, seen, out = self._run(
            c, variant, {"topk": "fused", "attn": "aiter"}, monkeypatch
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
        ],
    )
    def test_static_decode_equals_the_dynamic_path(self, attn, topk, monkeypatch):
        # decode as a graph runs it: the last token of each sequence plus a padding row
        # (slot -1), over a block table wider than any sequence, selection throughout
        c = Case([150, 97], topk=8 if topk == "fused" else 6, seed=3)
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

    @pytest.mark.skipif(not GPU, reason="CUDA graphs need a GPU")
    @pytest.mark.parametrize(
        "attn,topk",
        [
            ("triton", "torch"),
            ("aiter", "torch"),
            ("aiter", "aiter"),
            ("aiter", "fused"),
        ],
    )
    def test_decode_graph_replays_another_batch(self, attn, topk):
        # the static path captured over fixed buffers (a graph's decode), replayed for
        # other sequences at other positions plus a padding row; the token -> sequence
        # map comes from the metadata as in ATOM, whose capture reuses its warmup's
        c = Case([1500, 970, 610, 200], topk=64 if topk == "fused" else 40, seed=5)
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
