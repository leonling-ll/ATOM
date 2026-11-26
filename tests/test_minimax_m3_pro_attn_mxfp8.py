# SPDX-License-Identifier: MIT
"""``fuse_mxfp8_quant`` at o_proj: the decode attention writes the o_proj input as
MXFP8, and that is ``quantize_fp8`` of the bf16 output it writes otherwise, bit for bit
(E4M3 values and E8M0 codes) -- for every kernel that can end a Pro decode: the Triton
attention (split + combine, or one program a token), aiter's gluon decode one-shot, and
its PS reduce (the HIP one and the Triton one), at decode / DSpark-verify row counts over
short and long contexts, on an fp8 K-only cache and (aiter) on the same tokens as K and V
caches. GPU only.
"""

from __future__ import annotations

import itertools
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("MXFP8 attention output: GPU kernels", allow_module_level=True)
pytest.importorskip("aiter")

from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.minimax_m3 import pro_attn
from atom.model_ops.minimax_m3 import pro_kernels as K
from atom.model_ops.minimax_m3.pro_attn import ProVariants

DEV = "cuda"
FP8 = torch.float8_e4m3fn
H, D, DV, PG, TOPK = 16, 256, 192, 16, K.TOPK
SC = D**-0.5
WIDTH = 131072 // PG  # the block table of --max-model-len 131072

needs_aiter_mx = pytest.mark.skipif(
    not K.aiter_mxfp8_out_available(), reason="this aiter's decode writes bf16 only"
)


def _case(b, qlen, ctx, plain=True, seed=0, pool=65536):
    """``b`` sequences at context ``ctx``, the last ``qlen`` tokens of each a row (a
    decode at 1, a DSpark verify at k + 1), over random blocks of an fp8 K-only cache
    (``plain``: k_plain_fp8's unit scales)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    used = -(-ctx // PG)
    n = min(b * used, pool) + 1
    bt = torch.zeros(b, WIDTH, dtype=torch.int32, device=DEV)
    bt[:, :used] = torch.randint(
        0, n - 1, (b, used), generator=g, device=DEV, dtype=torch.int32
    )
    t = b * qlen
    seq = torch.arange(b, device=DEV).repeat_interleave(qlen)
    pos = (ctx - qlen + torch.arange(qlen, device=DEV)).repeat(b)
    j = torch.arange(WIDTH, device=DEV)
    score = torch.randn(t, WIDTH, generator=g, device=DEV)
    score = torch.where(j[None] <= (pos // PG)[:, None], score, float("-inf"))
    q = (torch.randn(t, H, D, generator=g, device=DEV) * 0.5).to(torch.bfloat16)
    kr = (torch.randn(n, 1, 1, PG, D, generator=g, device=DEV) * 0.7).to(FP8)
    if plain:
        ks = torch.ones(n, 1, PG, device=DEV)
    else:
        ks = torch.rand(n, 1, PG, generator=g, device=DEV) * 0.04 + 0.01
    return SimpleNamespace(
        t=t,
        bt=bt,
        seq=seq,
        pos=pos,
        q=q,
        kr=kr,
        ks=ks,
        rows=K.select_rows(score, bt, seq, pos, TOPK),
        sel=K.select_logical(score, pos, TOPK),
    )


def _kv_caches(kr):
    """The K-only rows ``[n, 1, 1, page, 256]`` as the K cache ``[n, 1, 256/x, page, x]``
    and the V cache ``[n, 1, page/x, 256, x]`` (V, the rows' last 192 dims, in its first
    192 rows) of the V-cache layout."""
    n, x = kr.shape[0], 16 // kr.element_size()
    rows = kr[:, 0, 0]
    kc = rows.view(n, PG, D // x, x).permute(0, 2, 1, 3).unsqueeze(1).contiguous()
    vc = torch.zeros(n, 1, PG // x, D, x, dtype=kr.dtype, device=kr.device)
    vc[:, 0, :, :DV] = rows[..., D - DV :].reshape(n, PG // x, x, DV).transpose(2, 3)
    return kc, vc


def _mx_out(t):
    return (
        torch.full((t, H * DV), 0x7F, dtype=torch.uint8, device=DEV).view(FP8),
        torch.full((t, H * DV // 32), 0xFF, dtype=torch.uint8, device=DEV),
    )


def _assert_quantized(ref_bf16, xq, xs):
    rq, rs = quantize_fp8(ref_bf16)
    assert torch.isfinite(ref_bf16.float()).all()
    bad_q = (xq.view(torch.uint8) != rq.view(torch.uint8)).sum().item()
    bad_s = (xs.view(torch.uint8) != rs.view(torch.uint8)).sum().item()
    assert bad_q == 0 and bad_s == 0, f"{bad_q} values, {bad_s} codes differ"


def _splits(t):
    return 32 if t <= 16 else 16 if t <= 32 else 8 if t <= 64 else 4 if t <= 128 else 2


# (sequences, query rows each, context): decode bs 1..64 (and the 129+ rows of a wide
# verify), DSpark verify q-len 4 / 8, contexts from inside the first selection to past 64k
SHAPES = [
    (1, 1, 300),
    (1, 1, 1024),
    (2, 1, 8192),
    (4, 1, 70000),
    (8, 1, 131072 - 16),
    (16, 1, 3000),
    (17, 1, 1024),
    (32, 1, 8192),
    (64, 1, 1024),
    (64, 1, 70000),
    (1, 4, 1024),
    (4, 4, 70000),
    (16, 4, 8192),
    (40, 4, 2000),
    (1, 8, 1024),
    (2, 8, 70000),
    (16, 8, 8192),
    (20, 8, 3000),
]


class TestTriton:
    @pytest.mark.parametrize("b,qlen,ctx", SHAPES)
    @pytest.mark.parametrize("plain", [True, False])
    def test_split_and_combine(self, b, qlen, ctx, plain):
        c = _case(b, qlen, ctx, plain, seed=b * 7 + qlen)
        self._check(c, _splits(c.t))

    @pytest.mark.parametrize("b,qlen,ctx", [(1, 1, 1024), (4, 4, 70000), (40, 4, 2000)])
    def test_one_program_a_token(self, b, qlen, ctx):
        self._check(_case(b, qlen, ctx), 1)

    @pytest.mark.parametrize("splits", [1, 3])
    def test_dense(self, splits):
        c = _case(3, 4, 1500)
        ref = torch.empty(c.t, H * DV, dtype=torch.bfloat16, device=DEV)
        nb = -(-1500 // PG)
        kw = {"nblk": 4, "n_blocks": nb, "k_scale": c.ks}
        K.attention(c.q, c.kr[:, 0], c.bt, c.seq, c.pos, None, ref, SC, splits, **kw)
        xq, xs = _mx_out(c.t)
        K.attention(
            c.q,
            c.kr[:, 0],
            c.bt,
            c.seq,
            c.pos,
            None,
            xq,
            SC,
            splits,
            out_scale=xs,
            **kw,
        )
        _assert_quantized(ref, xq, xs)

    @staticmethod
    def _check(c, splits):
        kw = {"nblk": 4, "k_scale": c.ks, "valid_first": True}
        ref = torch.empty(c.t, H * DV, dtype=torch.bfloat16, device=DEV)
        K.attention(c.q, c.kr[:, 0], c.bt, c.seq, c.pos, c.sel, ref, SC, splits, **kw)
        xq, xs = _mx_out(c.t)
        K.attention(
            c.q,
            c.kr[:, 0],
            c.bt,
            c.seq,
            c.pos,
            c.sel,
            xq,
            SC,
            splits,
            out_scale=xs,
            **kw,
        )
        _assert_quantized(ref, xq, xs)


@needs_aiter_mx
@pytest.mark.parametrize("k_rows", [True, False], ids=["k-rows", "v-cache"])
class TestAiter:
    @pytest.mark.parametrize("b,qlen,ctx", SHAPES)
    @pytest.mark.parametrize("reduce", ["hip", "triton"])
    def test_default_partitions(self, b, qlen, ctx, reduce, k_rows, monkeypatch):
        """Production's partition count: split + PS reduce up to 128 rows, one-shot past."""
        self._reduce(reduce, monkeypatch)
        self._check(_case(b, qlen, ctx, seed=b * 5 + qlen), None, k_rows)

    @pytest.mark.parametrize(
        "b,qlen,ctx", [(1, 1, 1024), (8, 1, 131072 - 16), (16, 8, 8192), (64, 4, 3000)]
    )
    @pytest.mark.parametrize("partitions", [1, 2, 8])
    def test_partitions(self, b, qlen, ctx, partitions, k_rows):
        """One-shot (the gluon kernel's own epilogue) and split (the reduce) at any size."""
        c = _case(b, qlen, ctx, plain=False, seed=partitions)
        self._check(c, partitions, k_rows)

    def test_one_shot_and_split_rows_differ_only_in_rounding(self, k_rows):
        # sanity: both paths compute the same attention (not a test of bits)
        c = _case(4, 1, 8192)
        outs = []
        for p in (1, 8):
            o = torch.empty(c.t, H * DV, dtype=torch.bfloat16, device=DEV)
            self._run(c, o, None, p, k_rows)
            outs.append(o.float())
        # the split path rounds every partition's output to bf16 before the reduce
        assert (outs[0] - outs[1]).abs().max() < 5e-2 * outs[1].abs().max()

    @staticmethod
    def _reduce(which, monkeypatch):
        """Pick the PS reduce of the ``pa_decode_gluon`` that ``torch.ops.aiter`` runs
        (its module globals: the op holds that function, whichever module object it
        came from), for the MXFP8 call and its bf16 reference alike."""
        import aiter.ops.attention as A

        g = A.pa_decode_gluon.__globals__
        if which == "triton":
            monkeypatch.setitem(g, "CXX_PS_REDUCE_AVAILABLE", False)
            monkeypatch.setitem(g, "launch_pa_decode_ps_reduce_mxfp8_cxx", None)
        else:
            assert g["CXX_PS_REDUCE_AVAILABLE"]
            assert g["launch_pa_decode_ps_reduce_mxfp8_cxx"] is not None

    @staticmethod
    def _run(c, out, out_scale, partitions, k_rows):
        kc, vc, vs = (c.kr, None, None) if k_rows else (*_kv_caches(c.kr), c.ks)
        K.attention_aiter(
            c.q,
            kc,
            vc,
            c.rows[0],
            c.rows[1],
            out,
            SC,
            c.ks,
            vs,
            partitions=partitions,
            k_rows=k_rows,
            out_scale=out_scale,
        )

    def _check(self, c, partitions, k_rows):
        ref = torch.empty(c.t, H * DV, dtype=torch.bfloat16, device=DEV)
        self._run(c, ref, None, partitions, k_rows)
        xq, xs = _mx_out(c.t)
        self._run(c, xq, xs, partitions, k_rows)
        _assert_quantized(ref, xq, xs)


class TestImpl:
    """The gate and the glue: which decode rows write o_proj's MXFP8 input and through which
    kernel (the day0 decode from ``day0_decode_rows``, aiter's and the Triton attention
    below it), ``_kernel_attention``'s MXFP8 out against its bf16 out quantised, and the
    forward's hand-off to o_proj (E4M3 rows, scales through ``take_mx_scale``).
    """

    @staticmethod
    def _impl():
        from test_minimax_m3_pro_kernels import Case, TestImplGlue

        return Case, TestImplGlue

    def test_gate(self):
        Case, _ = self._impl()
        m = Case([20]).impl
        m.k_only = True
        on = ProVariants(kernels=True, attn="aiter", fuse_mxfp8_quant=True)
        assert not m._mxfp8_out(on, 1)  # the o_proj has not said it is native group32
        assert not m._mxfp8_out(on, 64, day0_dec=True)
        m.mx_out = True
        assert m._mxfp8_out(on, 1)
        assert m._mxfp8_out(on, 20) == K.aiter_mxfp8_out_available()
        off = ProVariants(kernels=True, attn="aiter", fuse_mxfp8_quant=False)
        assert not m._mxfp8_out(off, 1)
        assert ProVariants().fuse_mxfp8_quant  # on by default
        # the day0 decode's epilogue: its own switch, whatever this aiter's decode writes
        assert m._mxfp8_out(on, 64, day0_dec=True) and m._mxfp8_out(off, 64, True)
        assert not m._mxfp8_out(replace(on, day0_decode_mxfp8=False), 64, True)
        # aiter over a V cache (every row count)
        m.k_only = False
        assert m._mxfp8_out(on, 64) == K.aiter_mxfp8_out_available()
        assert m._mxfp8_out(on, 1) == K.aiter_mxfp8_out_available()
        tri = ProVariants(kernels=True, attn="triton", fuse_mxfp8_quant=True)
        assert m._mxfp8_out(tri, 1) and m._mxfp8_out(tri, 64)
        # off under decode context parallelism, the day0 decode included
        m.dcp_world = 2
        assert not m._mxfp8_out(tri, 1) and not m._mxfp8_out(tri, 64)
        assert not m._mxfp8_out(on, 64, day0_dec=True)

    @staticmethod
    def _decode_case(n_seqs, k_only, **kw):
        """A static decode of ``n_seqs`` sequences (one row each), the impl and variants."""
        Case, Glue = TestImpl._impl()
        lens = [150 + 120 * i for i in range(n_seqs)]  # past 2048 tokens from i = 16 on
        c = Case(lens, topk=TOPK, seed=11, kv_dtype=FP8, ic_dtype=FP8, k_only=k_only)
        c.k_unit = k_only
        v = ProVariants(
            kernels=True,
            attn="aiter",
            topk="split",
            k_only=k_only,
            k_plain_fp8=k_only,
            fuse_mxfp8_quant=True,
            **kw,
        )
        m = Glue()._impl_with_rope(c, v)
        m.k_only, m.mx_out = k_only, True
        ends = torch.tensor([sum(lens[: s + 1]) - 1 for s in range(c.S)], device=DEV)
        wide = torch.cat([c.bt, torch.zeros_like(c.bt)], dim=1)
        seq = torch.arange(c.S, device=DEV)
        args = (c.qkv[ends], c.pos[ends], seq, c.slot[ends], wide, c.layer, c.ic)
        return c, v, m, ends, args, wide.shape[1]

    @pytest.mark.parametrize(
        "n_seqs,k_only", [(3, True), (20, True), (3, False), (20, False)]
    )
    def test_static_decode(self, n_seqs, k_only):
        """Below the day0 decode. K-only: Triton to 16 rows, aiter (K rows) past; V cache:
        aiter at both."""
        if (n_seqs >= 17 or not k_only) and not K.aiter_mxfp8_out_available():
            pytest.skip("this aiter's decode writes bf16 only")
        c, v, m, ends, args, nb = self._decode_case(n_seqs, k_only)
        assert not m._day0_decode(v, c.S, c.S, True) and m._mxfp8_out(v, c.S)
        full = torch.zeros(c.T, H * DV, dtype=torch.bfloat16, device=DEV)
        m._kernel_attention(
            c.qkv, c.pos, c.seq, c.slot, c.bt, c.layer, c.ic, c.bt.shape[1], v, full
        )
        ref = torch.zeros(c.S, H * DV, dtype=torch.bfloat16, device=DEV)
        m._kernel_attention(*args, nb, v, ref, static=True)
        xq, xs = _mx_out(c.S)
        m._kernel_attention(*args, nb, v, xq, static=True, mx_scale=xs)
        _assert_quantized(ref, xq, xs)
        assert (ref.float() - full[ends].float()).abs().max() < 0.1

    @pytest.mark.parametrize(
        "n_seqs,kernel", [(3, "triton"), (20, "aiter"), (24, "day0"), (24, "aiter")]
    )
    def test_rows_route_the_scales(self, monkeypatch, n_seqs, kernel):
        """Each row count hands ``mx_scale`` to the one kernel that attends it: the day0
        decode from ``day0_decode_rows`` rows, aiter (K rows) from 17 below it, Triton to
        16."""
        monkeypatch.setattr(K, "aiter_value_rows_available", lambda: True)
        monkeypatch.setattr(K, "aiter_mxfp8_out_available", lambda: True)
        monkeypatch.setattr(K, "day0_decode_available", lambda: True)
        got = []

        def fake(name, i):
            def f(*a, **kw):
                got.append((name, kw.get("out_scale")))
                return a[i]

            return f

        monkeypatch.setattr(K, "attention", fake("triton", 6))
        monkeypatch.setattr(K, "attention_aiter", fake("aiter", 5))
        monkeypatch.setattr(K, "attention_day0_decode", fake("day0", 6))
        rows = 24 if kernel == "day0" else 0
        c, v, m, _, args, nb = self._decode_case(n_seqs, True, day0_decode_rows=rows)
        day0 = m._day0_decode(v, c.S, c.S, True)
        assert day0 == (kernel == "day0") and m._mxfp8_out(v, c.S, day0)
        xq, xs = _mx_out(c.S)
        m._kernel_attention(*args, nb, v, xq, static=True, mx_scale=xs)
        assert len(got) == 1 and got[0][0] == kernel and got[0][1] is xs

    @pytest.mark.parametrize("fuse", [True, False])
    def test_forward_hands_triton_rows_to_o_proj(self, monkeypatch, fuse):
        """Below the day0 decode, the forward returns E4M3 rows and keeps their scales for
        ``take_mx_scale`` under ``fuse_mxfp8_quant``; bf16 without."""
        n_tok, n = 8, 3  # padding rows past the batch
        m = object.__new__(pro_attn.ProSparseAttentionImpl)
        m.num_heads, m.head_dim, m.value_head_dim, m.layer_num = H, D, DV, 1
        m.k_only, m.mx_out, m.index_cache = True, True, None
        v = ProVariants(kernels=True, attn="triton", k_only=True, fuse_mxfp8_quant=fuse)
        monkeypatch.setattr(pro_attn, "variants", lambda: v)
        monkeypatch.setattr(m, "_use_kernels", lambda v: True, raising=False)
        md = SimpleNamespace(
            slot_mapping=torch.zeros(n, dtype=torch.long),
            block_tables=torch.zeros(n, 16, dtype=torch.int32),
            cu_seqlens_q=torch.arange(0, n + 1, dtype=torch.int32),
        )
        layer = SimpleNamespace(k_cache=torch.zeros(1, 1, 1, PG, D, dtype=FP8))
        fwd = SimpleNamespace(
            context=SimpleNamespace(is_dummy_run=False, is_prefill=False),
            attn_metadata=md,
            kv_cache_data={"layer_1": layer},
        )
        monkeypatch.setattr(pro_attn, "get_forward_context", lambda: fwd)
        assert not m._day0_decode(v, n, n, True)

        def fake_kernel(parts, pos, seq, slots, bt, layer, ic, nb, v, out, **kw):
            assert out.shape[0] == n
            out.view(torch.uint8).fill_(0x38)  # 1.0 in e4m3 (bf16 bytes otherwise)
            if kw["mx_scale"] is not None:
                kw["mx_scale"].fill_(127)

        monkeypatch.setattr(m, "_kernel_attention", fake_kernel, raising=False)
        q = torch.zeros(n_tok, H * D, dtype=torch.bfloat16)
        out = m.forward(q, q, q, position=torch.zeros(n_tok, dtype=torch.long), qkv=q)
        assert out.shape == (n_tok, H * DV)
        if not fuse:
            assert out.dtype == q.dtype and m.take_mx_scale() is None
            return
        assert out.dtype == FP8 and (out[:n].float() == 1.0).all()
        assert (out[n:].view(torch.uint8) == 0).all()
        su = m.take_mx_scale().view(torch.uint8)
        assert su.shape == (n_tok, H * DV // 32)
        assert (su[:n] == 127).all() and (su[n:] == 0).all()


@pytest.mark.parametrize(
    "b,qlen,ctx", list(itertools.product([1, 32], [1, 4], [1024, 70000]))
)
def test_triton_mx_attention_matches_unfused_values(b, qlen, ctx):
    """The MXFP8 output dequantised is the bf16 output to within E4M3's step."""
    c = _case(b, qlen, ctx, seed=3)
    ref = torch.empty(c.t, H * DV, dtype=torch.bfloat16, device=DEV)
    kw = {"nblk": 4, "k_scale": c.ks, "valid_first": True}
    K.attention(c.q, c.kr[:, 0], c.bt, c.seq, c.pos, c.sel, ref, SC, _splits(c.t), **kw)
    xq, xs = _mx_out(c.t)
    K.attention(
        c.q,
        c.kr[:, 0],
        c.bt,
        c.seq,
        c.pos,
        c.sel,
        xq,
        SC,
        _splits(c.t),
        out_scale=xs,
        **kw,
    )
    deq = xq.float().view(c.t, -1, 32) * torch.exp2(xs.float() - 127)[..., None]
    err = (deq.view(c.t, -1) - ref.float()).abs()
    assert (err <= ref.float().abs() * 2**-3 + 2**-9 * deq.abs().max()).all()


needs_day0_decode = pytest.mark.skipif(
    not K.day0_decode_available(),
    reason="this aiter has no minimax_m3_fp8_sparse_decode",
)

# (sequences, query rows each, context): from the day0 decode's 24 rows, the CONC 32 k=3
# verify (8 to 22 requests of 4 rows at 55k to 64k), decode and wider verify rows
DAY0_SHAPES = [
    (24, 1, 1024),
    (64, 1, 70000),
    (8, 4, 56000),
    (22, 4, 64000),
    (16, 4, 8192),
    (1, 4, 300),
    (4, 8, 3000),
]


@needs_day0_decode
class TestDay0Decode:
    """aiter's FlyDSL sparse decode (``day0_decode_rows``): its bf16 output against the
    Triton attention on the same picks and the fp8-cast q, its MXFP8 epilogue against
    ``quantize_fp8`` of its own bf16 output, and the impl's dispatch against the batch
    with it off. No ``amd_kernels`` anywhere."""

    @staticmethod
    def _run(c, qlen, out, out_scale=None, slots=None):
        if slots is None:
            slots = torch.zeros(c.t, dtype=torch.long, device=DEV)
        return K.attention_day0_decode(
            c.q, c.kr[:, 0], c.bt, c.pos, slots, c.sel, out, SC, qlen, out_scale
        )

    @pytest.mark.parametrize("b,qlen,ctx", DAY0_SHAPES)
    def test_matches_the_triton_attention(self, b, qlen, ctx):
        c = _case(b, qlen, ctx, seed=b * 3 + qlen)
        ref = torch.empty(c.t, H * DV, dtype=torch.bfloat16, device=DEV)
        kw = {"nblk": 4, "k_scale": c.ks, "valid_first": True}
        q8 = c.q.to(FP8).to(torch.bfloat16)  # the day0 decode reads q as fp8
        K.attention(q8, c.kr[:, 0], c.bt, c.seq, c.pos, c.sel, ref, SC, 4, **kw)
        got = torch.full_like(ref, float("nan"))
        self._run(c, qlen, got)
        assert torch.allclose(got.float(), ref.float(), atol=3e-2, rtol=1e-2)

    @pytest.mark.parametrize("b,qlen,ctx", DAY0_SHAPES)
    def test_mxfp8_epilogue_is_quantize_fp8_of_its_bf16(self, b, qlen, ctx):
        c = _case(b, qlen, ctx, seed=b * 3 + qlen)
        ref = torch.empty(c.t, H * DV, dtype=torch.bfloat16, device=DEV)
        self._run(c, qlen, ref)
        xq, xs = _mx_out(c.t)
        self._run(c, qlen, xq, xs)
        _assert_quantized(ref, xq, xs)

    def test_padding_requests_write_zeros(self):
        c = _case(6, 4, 8192)
        slots = torch.zeros(c.t, dtype=torch.long, device=DEV)
        slots[-8:] = -1  # the last two requests: a graph's padding
        got = torch.full((c.t, H * DV), float("nan"), dtype=torch.bfloat16, device=DEV)
        self._run(c, 4, got, slots=slots)
        assert torch.isfinite(got[:-8].float()).all() and (got[-8:] == 0).all()

    @pytest.mark.parametrize("n_seqs", [24, 40, 64])
    @pytest.mark.parametrize("mx", [False, True])
    def test_prep_q8_gives_the_same_bytes(self, n_seqs, mx, monkeypatch):
        # prep_q8: aiter's prep writes the decode's E4M3 q itself instead of the
        # decode casting the bf16 q; the impl's output (bf16, or the MXFP8
        # epilogue) is bit-equal
        if not (K.aiter_prep_available(k_unit=True) and K.aiter_prep_q8_available()):
            pytest.skip("this aiter's prep has no kv_unit_scale / E4M3 q_out")
        c, v, m, _, args, nb = TestImpl._decode_case(n_seqs, True, prep="aiter")
        m.__dict__["index_cache"] = c.ic
        assert m._day0_decode(v, c.S, c.S, True) and m._aiter_prep(v)
        seen, real = [], K.prep_aiter

        def spy(*a, **kw):
            seen.append(kw.get("q8", False))
            return real(*a, **kw)

        monkeypatch.setattr(K, "prep_aiter", spy)
        outs = []
        for q8 in (False, True):
            if mx:
                o, sc = _mx_out(c.S)
            else:
                o, sc = torch.zeros(c.S, H * DV, dtype=torch.bfloat16, device=DEV), None
            m._kernel_attention(
                *args, nb, replace(v, prep_q8=q8), o, static=True, mx_scale=sc
            )
            outs.append((o, sc))
        assert seen == [False, True]
        raw = lambda t: t.view(torch.uint8)
        assert torch.equal(raw(outs[0][0]), raw(outs[1][0]))
        if mx:
            assert torch.equal(outs[0][1], outs[1][1])

    @pytest.mark.parametrize("n_seqs", [24, 40])
    def test_impl_matches_the_attention_below_it(self, n_seqs):
        # the impl's static decode: the day0 decode at day0_decode_rows, against the same
        # batch with it off (aiter's K rows where this aiter reads them, else Triton)
        c, v, m, _, args, nb = TestImpl._decode_case(n_seqs, True)
        assert m._day0_decode(v, c.S, c.S, True)
        got = torch.zeros(c.S, H * DV, dtype=torch.bfloat16, device=DEV)
        m._kernel_attention(*args, nb, v, got, static=True)
        ref = torch.zeros_like(got)
        m._kernel_attention(*args, nb, replace(v, day0_decode_rows=0), ref, static=True)
        assert torch.allclose(got.float(), ref.float(), atol=3e-2, rtol=1e-2)
