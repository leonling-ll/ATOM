# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Triton kernels for MiniMax-M3-Pro's attention (see ``pro_attn`` for the definition).

Four launches replace the torch reference's hundred-odd small ops per layer and need no
host sync:

* ``prep``       gemma-norm + partial rope of q / k / index-q / index-k, K, V (K's 192
                 un-roped dims) and index-K written straight into the paged caches (K
                 ``[block, 1, D/x, page, x]``, V ``[block, 1, page/x, 256, x]`` of which the
                 first 192 rows are used); an fp8 KV cache is quantized per token, the
                 dequant scales written to ``[block, 1, page]`` planes;
* ``scores``     the lightning indexer: per query token and 16-token block, the score of
                 the 16 index heads (the max of the block's 4 softmax-pooled centroids, or
                 max or log-sum-exp over its token dots; a few blocks per dot), reduced over
                 heads, with the forced init / local blocks pinned above every real score and
                 future blocks at -inf; ``torch.topk`` over these picks the blocks (or
                 ``select_blocks``, aiter's per-row top-k over each token's causal blocks);
* ``attention``  flash-style decode/prefill attention of a token's 16 local heads over its
                 selected blocks (MQA: one 256-wide K head; V is the K tile without its 64
                 rope dims, read from the same cache), optionally split over the blocks
                 for small batches, followed by ``combine``.

``prep_aiter`` is the alternative to ``prep``: aiter's ``fused_qk_norm_rope_cache_mqa``, the
same math (and rounding points) in one HIP launch of a 32-lane group per head row, any
number of local heads.

``attention_aiter`` is the alternative to ``attention``: ``sparse_rows`` turns every
token's selected blocks into a page table of its own (its own block last, so the context
length masks the part of it past the token), and aiter's paged-attention decode runs over
those tables, reading V from the V cache. ``select_rows`` selects and builds those tables in
one launch.

Shapes are fixed by the model: head_dim 256, rotary 64, value 192, index 16 x 128, page 16.
"""

import torch
import triton
import triton.language as tl

PAGE = 16
HEAD_DIM = 256
ROT = 64
VALUE_DIM = 192
IDX_HEADS = 16
IDX_DIM = 128
TOPK = 128

BLOCK_TOKMAX = 0
BLOCK_POOL4MAX = 1
BLOCK_POOL4MEAN = 2
BLOCK_TOKLSE = 3
HEAD_SUM = 0
HEAD_MAX = 1
HEAD_LSE = 2

# jit'ed code may only read module globals that are constexpr
D_HEAD = tl.constexpr(HEAD_DIM)
D_IDX = tl.constexpr(IDX_DIM)
# smallest amax an fp8 row is scaled for (an all-zero row stores zeros, not NaN)
FP8_AMAX_FLOOR = 1e-6
F8_FLOOR = tl.constexpr(FP8_AMAX_FLOOR)


def quant_fp8_rows(x: torch.Tensor, dtype: torch.dtype, plain: bool = False):
    """Per-row fp8 quantization as ``prep`` does it: ``(q, scale)`` with
    ``x ~= q * scale[:, None]``, ``scale = max(amax, floor) / fp8_max`` (fp32); ``plain``
    the clamped cast, every scale 1.0 (``prep``'s ``k_unit``)."""
    fmax = torch.finfo(dtype).max
    xf = x.float()
    scale = xf.abs().amax(dim=-1).clamp(min=FP8_AMAX_FLOOR) / fmax
    if plain:
        scale = torch.ones_like(scale)
    q = (xf / scale.unsqueeze(-1)).clamp(-fmax, fmax).to(dtype)
    return q, scale


@triton.jit
def _to_fp8(x, scale, fp8_max, DT: tl.constexpr):
    return tl.minimum(tl.maximum(x / scale, -fp8_max), fp8_max).to(DT)


# An NVFP4 index row (``--index-cache-dtype nvfp4``): 128 e2m1 (64 B, the low nibble the
# even dim) and 8 e4m3 scales, one per 16 dims, no global scale; a page of PAGE tokens is
# [PAGE x 64 B data | PAGE x 8 B scales]. Quantized as vLLM's ``_quant4``: the scale
# e4m3(min(amax * RN(1/6), 448)), the elements x / that scale (decoded), RNE to e2m1.
NV_DATA_BYTES = IDX_DIM // 2
NV_SCALE_BYTES = IDX_DIM // 16
NV_ROW_BYTES = NV_DATA_BYTES + NV_SCALE_BYTES
NV_RCP6 = tl.constexpr(0.1666666716337204)  # RN(1/6) in fp32 (0x3E2AAAAB)


@triton.jit
def _nvfp4_store(x, G: tl.constexpr, data_ptr, scale_ptr):
    """``x`` ``[1, 16 G]`` (fp32 of bf16 values): its G groups' 8 G packed e2m1 bytes at
    ``data_ptr``, their G e4m3 scale bytes at ``scale_ptr``."""
    xg = tl.reshape(x, (G, 16))
    sf = tl.minimum(tl.max(tl.abs(xg), axis=1) * NV_RCP6, 448.0)
    s8 = sf.to(tl.float8e4nv)
    sv = s8.to(tl.float32)
    y = xg * tl.where(sv != 0.0, 1.0 / sv, 0.0)[:, None]
    a = tl.abs(y)
    # RNE onto 0, .5, 1, 1.5, 2, 3, 4, 6: a tie goes to the even code
    c = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    c = tl.where(y < 0, c + 8, c)
    ev, od = tl.split(tl.reshape(c, (G * 8, 2)))
    tl.store(data_ptr + tl.arange(0, G * 8), (ev | (od << 4)).to(tl.uint8))
    tl.store(scale_ptr + tl.arange(0, G), s8.to(tl.uint8, bitcast=True))


@triton.jit
def _nvfp4_rows(dat, sc, R: tl.constexpr):
    """``[R, 128]`` fp32 index keys from their ``[R, 64]`` e2m1 bytes and ``[R, 8]`` e4m3
    scale bytes."""
    c = tl.reshape(tl.join((dat & 15).to(tl.int32), (dat >> 4).to(tl.int32)), (R, 128))
    e = (c >> 1) & 3
    m = (c & 1).to(tl.float32)
    mag = tl.where(e == 0, 0.5 * m, (1.0 + 0.5 * m) * tl.exp2((e - 1).to(tl.float32)))
    v = tl.where((c & 8) != 0, -mag, mag)
    s = sc.to(tl.float8e4nv, bitcast=True).to(tl.float32)
    return tl.reshape(tl.reshape(v, (R, 8, 16)) * s[:, :, None], (R, 128))


def nvfp4_rows(x: torch.Tensor) -> torch.Tensor:
    """The torch reference of an NVFP4 index row: ``[..., 128]`` -> the ``[..., 72]`` uint8
    row ``prep`` writes (64 data bytes, 8 scale bytes)."""
    xg = x.float().reshape(*x.shape[:-1], 8, 16)
    sf = (xg.abs().amax(-1) * 0.1666666716337204).clamp(max=448.0)
    s8 = sf.to(torch.float8_e4m3fn)
    sv = s8.float()
    y = xg * torch.where(sv != 0, 1.0 / sv, torch.zeros_like(sv)).unsqueeze(-1)
    a = y.abs()
    c = sum(
        (op(a, t)).int()
        for op, t in (
            (torch.gt, 0.25),
            (torch.ge, 0.75),
            (torch.gt, 1.25),
            (torch.ge, 1.75),
            (torch.gt, 2.5),
            (torch.ge, 3.5),
            (torch.gt, 5.0),
        )
    )
    c = torch.where(y < 0, c + 8, c).reshape(*x.shape[:-1], 64, 2)
    data = (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)
    return torch.cat([data, s8.view(torch.uint8)], dim=-1)


def nvfp4_dequant_rows(rows: torch.Tensor) -> torch.Tensor:
    """``[..., 72]`` uint8 NVFP4 rows -> ``[..., 128]`` fp32."""
    dat, sc = rows[..., :64].int(), rows[..., 64:]
    c = torch.stack([dat & 15, dat >> 4], dim=-1).reshape(*rows.shape[:-1], 128)
    e, m = (c >> 1) & 3, (c & 1).float()
    mag = torch.where(e == 0, 0.5 * m, (1 + 0.5 * m) * torch.exp2((e - 1).float()))
    v = torch.where((c & 8) != 0, -mag, mag).reshape(*rows.shape[:-1], 8, 16)
    s = sc.view(torch.float8_e4m3fn).float().unsqueeze(-1)
    return (v * s).reshape(*rows.shape[:-1], 128)


# --------------------------------------------------------------------------- prep --
@triton.jit
def _norm_rope_rows(
    lo,
    hi,
    rest0,
    rest1,
    w_lo,
    w_hi,
    w_r0,
    w_r1,
    c,
    s,
    eps,
    DIM: tl.constexpr,
    OUT_DT: tl.constexpr,
):
    """Gemma RMSNorm (``x * rsqrt(mean(x^2) + eps) * (1 + w)``, rounded to the activation
    dtype) of rows given as ``[R, 32]`` ``lo`` / ``hi`` (the rope pairs), ``[R, W0]``
    ``rest0`` and ``[R, W1]`` ``rest1``, then NeoX rope on ``lo`` / ``hi``."""
    ss = tl.sum(lo * lo, axis=1) + tl.sum(hi * hi, axis=1)
    ss += tl.sum(rest0 * rest0, axis=1) + tl.sum(rest1 * rest1, axis=1)
    r = tl.rsqrt(ss / DIM + eps)[:, None]
    lo = (lo * r * w_lo[None, :]).to(OUT_DT).to(tl.float32)
    hi = (hi * r * w_hi[None, :]).to(OUT_DT).to(tl.float32)
    rest0 = (rest0 * r * w_r0[None, :]).to(OUT_DT)
    rest1 = (rest1 * r * w_r1[None, :]).to(OUT_DT)
    new_lo = (lo * c[None, :] - hi * s[None, :]).to(OUT_DT)
    new_hi = (hi * c[None, :] + lo * s[None, :]).to(OUT_DT)
    return new_lo, new_hi, rest0, rest1


@triton.jit
def _prep_kernel(
    q_ptr,
    sq,
    k_ptr,
    sk,
    iq_ptr,
    siq,
    ik_ptr,
    sik,
    pos_ptr,
    slot_ptr,
    cos_ptr,
    sin_ptr,
    stride_cs,
    wq_ptr,
    wk_ptr,
    wiq_ptr,
    wik_ptr,
    q_out_ptr,
    iq_out_ptr,
    kc_ptr,
    k_sb,
    k_sg,
    k_so,
    vc_ptr,
    v_sb,
    v_sg,
    ks_ptr,
    vs_ptr,
    s_sb,
    ic_ptr,
    ic_sb,
    ic_so,
    ratio,
    eps,
    fp8_max,
    ic_max,
    H: tl.constexpr,
    NI: tl.constexpr,
    PG: tl.constexpr,
    XK: tl.constexpr,
    XV: tl.constexpr,
    WRITE_V: tl.constexpr,
    KV_FP8: tl.constexpr,
    IC_FP8: tl.constexpr,
    K_UNIT: tl.constexpr,
    IC_NV: tl.constexpr,
):
    t = tl.program_id(0)
    OUT_DT = q_out_ptr.dtype.element_ty
    t64 = t.to(tl.int64)
    pos = tl.load(pos_ptr + t)
    c = tl.load(cos_ptr + pos * stride_cs + tl.arange(0, 32)).to(tl.float32) * ratio
    s = tl.load(sin_ptr + pos * stride_cs + tl.arange(0, 32)).to(tl.float32) * ratio

    a32 = tl.arange(0, 32)
    a64 = tl.arange(0, 64)
    a128 = tl.arange(0, 128)

    # ---- q: H heads x 256 ------------------------------------------------------------
    wl = 1.0 + tl.load(wq_ptr + a32).to(tl.float32)
    wh = 1.0 + tl.load(wq_ptr + 32 + a32).to(tl.float32)
    w0 = 1.0 + tl.load(wq_ptr + 64 + a64).to(tl.float32)
    w1 = 1.0 + tl.load(wq_ptr + 128 + a128).to(tl.float32)
    hh = tl.arange(0, H)[:, None]
    rowq = q_ptr + t64 * sq + hh * D_HEAD
    lo = tl.load(rowq + a32[None, :]).to(tl.float32)
    hi = tl.load(rowq + 32 + a32[None, :]).to(tl.float32)
    r0 = tl.load(rowq + 64 + a64[None, :]).to(tl.float32)
    r1 = tl.load(rowq + 128 + a128[None, :]).to(tl.float32)
    lo, hi, r0, r1 = _norm_rope_rows(
        lo, hi, r0, r1, wl, wh, w0, w1, c, s, eps, D_HEAD, OUT_DT
    )
    qo = q_out_ptr + t.to(tl.int64) * (H * D_HEAD) + hh * D_HEAD
    tl.store(qo + a32[None, :], lo)
    tl.store(qo + 32 + a32[None, :], hi)
    tl.store(qo + 64 + a64[None, :], r0)
    tl.store(qo + 128 + a128[None, :], r1)

    # ---- k: one head x 256, straight into the K cache ----------------------------------
    wl = 1.0 + tl.load(wk_ptr + a32).to(tl.float32)
    wh = 1.0 + tl.load(wk_ptr + 32 + a32).to(tl.float32)
    w0 = 1.0 + tl.load(wk_ptr + 64 + a64).to(tl.float32)
    w1 = 1.0 + tl.load(wk_ptr + 128 + a128).to(tl.float32)
    rowk = k_ptr + t64 * sk
    lo = tl.load(rowk + a32[None, :]).to(tl.float32)
    hi = tl.load(rowk + 32 + a32[None, :]).to(tl.float32)
    r0 = tl.load(rowk + 64 + a64[None, :]).to(tl.float32)
    r1 = tl.load(rowk + 128 + a128[None, :]).to(tl.float32)
    lo, hi, r0, r1 = _norm_rope_rows(
        lo, hi, r0, r1, wl, wh, w0, w1, c, s, eps, D_HEAD, OUT_DT
    )
    slot = tl.load(slot_ptr + t).to(tl.int64)
    blk = slot // PG
    off = slot % PG
    if slot >= 0:
        KV_DT = kc_ptr.dtype.element_ty
        lo_k, hi_k, r0_k, r1_k = lo, hi, r0, r1
        r0_v, r1_v = r0, r1
        if KV_FP8:
            lo_f, hi_f = lo.to(tl.float32), hi.to(tl.float32)
            r0_f, r1_f = r0.to(tl.float32), r1.to(tl.float32)
            vmax = tl.maximum(tl.max(tl.abs(r0_f)), tl.max(tl.abs(r1_f)))
            kmax = tl.maximum(
                tl.maximum(tl.max(tl.abs(lo_f)), tl.max(tl.abs(hi_f))), vmax
            )
            ksc = tl.maximum(kmax, F8_FLOOR) / fp8_max
            if K_UNIT:  # a plain cast: what moo's / day0's kernels read
                ksc = ksc * 0.0 + 1.0
            lo_k = _to_fp8(lo_f, ksc, fp8_max, KV_DT)
            hi_k = _to_fp8(hi_f, ksc, fp8_max, KV_DT)
            r0_k = _to_fp8(r0_f, ksc, fp8_max, KV_DT)
            r1_k = _to_fp8(r1_f, ksc, fp8_max, KV_DT)
            tl.store(ks_ptr + blk * s_sb + off, ksc)
            if WRITE_V:
                vsc = tl.maximum(vmax, F8_FLOOR) / fp8_max
                if K_UNIT:
                    vsc = vsc * 0.0 + 1.0
                r0_v = _to_fp8(r0_f, vsc, fp8_max, KV_DT)
                r1_v = _to_fp8(r1_f, vsc, fp8_max, KV_DT)
                tl.store(vs_ptr + blk * s_sb + off, vsc)
        kb = kc_ptr + blk * k_sb + off * k_so
        tl.store(kb + (a32[None, :] // XK) * k_sg + (a32[None, :] % XK), lo_k)
        tl.store(
            kb + ((32 + a32[None, :]) // XK) * k_sg + ((32 + a32[None, :]) % XK), hi_k
        )
        tl.store(
            kb + ((64 + a64[None, :]) // XK) * k_sg + ((64 + a64[None, :]) % XK), r0_k
        )
        tl.store(
            kb + ((128 + a128[None, :]) // XK) * k_sg + ((128 + a128[None, :]) % XK),
            r1_k,
        )
        if WRITE_V:
            # V row d of token `off` sits at [off // XV, d, off % XV] of the block
            vb = vc_ptr + blk * v_sb + (off // XV) * v_sg + (off % XV)
            tl.store(vb + a64[None, :] * XV, r0_v)
            tl.store(vb + (64 + a128[None, :]) * XV, r1_v)

    # ---- index q: NI heads x 128 (two rope halves and a 64-wide rest) --------------------
    wl = 1.0 + tl.load(wiq_ptr + a32).to(tl.float32)
    wh = 1.0 + tl.load(wiq_ptr + 32 + a32).to(tl.float32)
    w0 = 1.0 + tl.load(wiq_ptr + 64 + a64).to(tl.float32)
    ii = tl.arange(0, NI)[:, None]
    rowi = iq_ptr + t64 * siq + ii * D_IDX
    lo = tl.load(rowi + a32[None, :]).to(tl.float32)
    hi = tl.load(rowi + 32 + a32[None, :]).to(tl.float32)
    r0 = tl.load(rowi + 64 + a64[None, :]).to(tl.float32)
    z32 = tl.zeros([NI, 32], tl.float32)
    lo, hi, r0, z32 = _norm_rope_rows(
        lo,
        hi,
        r0,
        z32,
        wl,
        wh,
        w0,
        tl.zeros([32], tl.float32),
        c,
        s,
        eps,
        D_IDX,
        OUT_DT,
    )
    io = iq_out_ptr + t.to(tl.int64) * (NI * D_IDX) + ii * D_IDX
    tl.store(io + a32[None, :], lo)
    tl.store(io + 32 + a32[None, :], hi)
    tl.store(io + 64 + a64[None, :], r0)

    # ---- index k: one head x 128, into the index cache ---------------------------------
    wl = 1.0 + tl.load(wik_ptr + a32).to(tl.float32)
    wh = 1.0 + tl.load(wik_ptr + 32 + a32).to(tl.float32)
    w0 = 1.0 + tl.load(wik_ptr + 64 + a64).to(tl.float32)
    rowj = ik_ptr + t64 * sik
    lo = tl.load(rowj + a32[None, :]).to(tl.float32)
    hi = tl.load(rowj + 32 + a32[None, :]).to(tl.float32)
    r0 = tl.load(rowj + 64 + a64[None, :]).to(tl.float32)
    z1 = tl.zeros([1, 32], tl.float32)
    lo, hi, r0, z1 = _norm_rope_rows(
        lo, hi, r0, z1, wl, wh, w0, tl.zeros([32], tl.float32), c, s, eps, D_IDX, OUT_DT
    )
    if IC_NV:  # [page x 64 B e2m1 | page x 8 B e4m3 scales] a block
        if slot >= 0:
            nd = ic_ptr + blk * ic_sb + off * 64
            ns = ic_ptr + blk * ic_sb + PG * 64 + off * 8
            _nvfp4_store(lo.to(tl.float32), 2, nd, ns)
            _nvfp4_store(hi.to(tl.float32), 2, nd + 16, ns + 2)
            _nvfp4_store(r0.to(tl.float32), 4, nd + 32, ns + 4)
    elif slot >= 0:
        lo_i, hi_i, r0_i = lo, hi, r0
        if IC_FP8:  # a plain cast, as M3's fp8 index cache: the keys are normed
            IC_DT = ic_ptr.dtype.element_ty
            lo_i = _to_fp8(lo.to(tl.float32), 1.0, ic_max, IC_DT)
            hi_i = _to_fp8(hi.to(tl.float32), 1.0, ic_max, IC_DT)
            r0_i = _to_fp8(r0.to(tl.float32), 1.0, ic_max, IC_DT)
        ib = ic_ptr + blk * ic_sb + off * ic_so
        tl.store(ib + a32[None, :], lo_i)
        tl.store(ib + 32 + a32[None, :], hi_i)
        tl.store(ib + 64 + a64[None, :], r0_i)


def _split_rows(qkv, heads):
    """``(q, k, index q, index k)`` from the packed rows or the tuple of the four."""
    if isinstance(qkv, torch.Tensor):
        o = heads * HEAD_DIM
        qkv = (
            qkv,
            qkv[:, o:],
            qkv[:, o + HEAD_DIM :],
            qkv[:, o + HEAD_DIM + IDX_HEADS * IDX_DIM :],
        )
    return qkv


def prep(
    qkv,
    positions,
    slots,
    cos,
    sin,
    ratio,
    wq,
    wk,
    wiq,
    wik,
    eps,
    kc4,
    ic,
    heads,
    vc4=None,
    k_scale=None,
    v_scale=None,
    k_unit=False,
):
    """Norm + rope the raw projections and write K / V / index-K to the caches.

    ``qkv`` is the packed rows ``[q | k | index q | index k]`` or those four as a tuple
    (each with its own row stride). Returns ``q`` ``[T, heads, 256]`` and ``iq``
    ``[T, 16, 128]`` (activation dtype).
    ``kc4`` is the K cache ``[N, D/x, page, x]``, ``vc4`` (optional) the V cache
    ``[N, page/x, 256, x]``, ``ic`` the index cache ``[N, page, 128]``. An fp8 ``kc4``
    takes the per-token dequant scales ``k_scale`` / ``v_scale`` ``[N, 1, page]``; an fp8
    ``ic`` holds the keys cast (clamped to the fp8 range), unscaled. ``k_unit``: an fp8 K /
    V cache a plain cast too, its scales written 1.0.
    """
    qkv = _split_rows(qkv, heads)
    q, k, iq, ik = qkv
    t = q.shape[0]
    q_out = torch.empty(t, heads, HEAD_DIM, dtype=q.dtype, device=q.device)
    iq_out = torch.empty(t, IDX_HEADS, IDX_DIM, dtype=q.dtype, device=q.device)
    assert (
        kc4.stride(-1) == 1
        and ic.stride(-1) == 1
        and all(p.stride(-1) == 1 for p in qkv)
    )
    assert q.shape[-1] >= heads * HEAD_DIM and k.shape[-1] >= HEAD_DIM
    assert iq.shape[-1] >= IDX_HEADS * IDX_DIM and ik.shape[-1] >= IDX_DIM
    fp8 = kc4.element_size() == 1
    ic_nv = ic.dtype == torch.uint8  # NVFP4 rows, [N, page * 72]
    ic_fp8 = ic.element_size() == 1 and not ic_nv
    write_v = vc4 is not None
    if write_v:
        assert vc4.dtype == kc4.dtype and vc4.stride(-1) == 1
        assert vc4.stride(-2) == vc4.shape[-1] and vc4.shape[-2] >= VALUE_DIM
    if fp8:
        assert k_scale is not None and k_scale.stride(-1) == 1
        assert (
            v_scale is not None and v_scale.stride(0) == k_scale.stride(0)
        ) or not write_v
    ks = k_scale if fp8 else kc4
    vs = (v_scale if write_v else k_scale) if fp8 else kc4
    _prep_kernel[(t,)](
        q,
        q.stride(0),
        k,
        k.stride(0),
        iq,
        iq.stride(0),
        ik,
        ik.stride(0),
        positions,
        slots,
        cos,
        sin,
        cos.stride(0),
        wq,
        wk,
        wiq,
        wik,
        q_out,
        iq_out,
        kc4,
        kc4.stride(0),
        kc4.stride(1),
        kc4.stride(2),
        vc4 if write_v else kc4,
        vc4.stride(0) if write_v else 0,
        vc4.stride(1) if write_v else 0,
        ks,
        vs,
        ks.stride(0) if fp8 else 0,
        ic,
        ic.stride(0),
        ic.stride(1),
        ratio,
        eps,
        torch.finfo(kc4.dtype).max if fp8 else 1.0,
        torch.finfo(ic.dtype).max if ic_fp8 else 1.0,
        H=heads,
        NI=IDX_HEADS,
        PG=PAGE,
        XK=kc4.shape[-1],
        XV=vc4.shape[-1] if write_v else 1,
        WRITE_V=write_v,
        KV_FP8=fp8,
        IC_FP8=ic_fp8,
        K_UNIT=k_unit,
        IC_NV=ic_nv,
    )
    return q_out, iq_out


_fused_prep = (
    None  # aiter's fused_qk_norm_rope_cache_mqa (tests swap it); False: missing
)


def _aiter_prep():
    global _fused_prep
    if _fused_prep is None:
        try:
            from aiter.ops.fused_qk_norm_rope_cache_mqa import (
                fused_qk_norm_rope_cache_mqa,
            )
        except ImportError:
            _fused_prep = False
        else:
            _fused_prep = fused_qk_norm_rope_cache_mqa
    return _fused_prep


def aiter_prep_available() -> bool:
    """Whether this aiter has ``fused_qk_norm_rope_cache_mqa`` (``prep_aiter``)."""
    return bool(_aiter_prep())


def prep_aiter_fits(qkv, heads) -> bool:
    """Whether ``prep_aiter`` takes these rows: 2-D, rows 16-byte (q, k) / 8-byte (index)
    aligned (its vector loads)."""

    def ok(x, vec):
        return (
            x.dim() == 2
            and x.stride(-1) == 1
            and x.stride(0) % vec == 0
            and x.data_ptr() % (vec * x.element_size()) == 0
        )

    q, k, iq, ik = _split_rows(qkv, heads)
    return ok(q, 8) and ok(k, 8) and ok(iq, 4) and ok(ik, 4)


def prep_aiter(
    qkv,
    positions,
    slots,
    cos,
    sin,
    ratio,
    wq,
    wk,
    wiq,
    wik,
    eps,
    kc4,
    ic,
    heads,
    vc4=None,
    k_scale=None,
    v_scale=None,
):
    """``prep`` in one launch of aiter's ``fused_qk_norm_rope_cache_mqa``: the same arguments
    and results (rows as ``prep_aiter_fits`` takes them), any number of heads. The op takes
    the norm weights and cos / sin in the activation dtype; others go to ``prep``."""
    q, k, iq, ik = _split_rows(qkv, heads)
    if any(t.dtype != q.dtype for t in (wq, wk, wiq, wik, cos, sin)):
        return prep(
            qkv,
            positions,
            slots,
            cos,
            sin,
            ratio,
            wq,
            wk,
            wiq,
            wik,
            eps,
            kc4,
            ic,
            heads,
            vc4=vc4,
            k_scale=k_scale,
            v_scale=v_scale,
        )
    fp8 = kc4.element_size() == 1
    return _aiter_prep()(
        q,
        k,
        iq,
        ik,
        positions,
        slots,
        cos,
        sin,
        wq,
        wk,
        wiq,
        wik,
        kc4,
        ic,
        heads,
        eps,
        rope_scale=ratio,
        v_cache=vc4,
        k_scale=k_scale if fp8 else None,
        v_scale=v_scale if fp8 and vc4 is not None else None,
        num_index_heads=IDX_HEADS,
        kv_fp8_max=torch.finfo(kc4.dtype).max if fp8 else None,
        index_fp8_max=torch.finfo(ic.dtype).max if ic.element_size() == 1 else None,
    )


# ------------------------------------------------------------------------- scores --
@triton.jit
def _score_kernel(
    iq_ptr,
    ic_ptr,
    ic_sb,
    ic_so,
    bt_ptr,
    stride_bt,
    seq_ptr,
    pos_ptr,
    cq_ptr,
    out_ptr,
    stride_out,
    sc,
    pool_temp,
    init_blocks,
    local_blocks,
    NB,
    BB: tl.constexpr,
    NI: tl.constexpr,
    DI: tl.constexpr,
    PG: tl.constexpr,
    BLOCK_MODE: tl.constexpr,
    HEAD_MODE: tl.constexpr,
):
    t = tl.program_id(0)
    jt = tl.program_id(1)
    seq = tl.load(seq_ptr + t)
    qpos = tl.load(pos_ptr + t)
    qb = qpos // PG
    ar_i = tl.arange(0, NI)
    ar_d = tl.arange(0, DI)
    ar_p = tl.arange(0, PG)
    iq = tl.load(
        iq_ptr + t.to(tl.int64) * (NI * DI) + ar_i[:, None] * DI + ar_d[None, :]
    )  # [NI, DI]
    cq = tl.load(cq_ptr + ar_d).to(
        tl.float32
    )  # (read for every mode; unused by tokmax)
    neg = tl.full([1], float("-inf"), tl.float32)
    one = tl.arange(0, 1)
    obase = out_ptr + t.to(tl.int64) * stride_out
    for jj in range(BB):
        j = jt * BB + jj
        if j < NB:
            if j <= qb:
                phys = tl.load(bt_ptr + seq * stride_bt + j).to(tl.int64)
                ik = tl.load(
                    ic_ptr + phys * ic_sb + ar_p[:, None] * ic_so + ar_d[None, :]
                )  # [PG, DI]
                ik = ik.to(iq.dtype)  # an fp8 index cache
                dots = tl.dot(iq, tl.trans(ik)) * sc  # [NI, PG]
                if BLOCK_MODE == 0 or BLOCK_MODE == 3:
                    vis = (j * PG + ar_p) <= qpos
                    dm = tl.where(vis[None, :], dots, float("-inf"))  # [NI, PG]
                    bs = tl.max(dm, axis=1)  # [NI]
                    if (
                        BLOCK_MODE == 3
                    ):  # log sum exp over the visible tokens (>= 1: j <= qb)
                        bs = tl.log(tl.sum(tl.exp(dm - bs[:, None]), axis=1)) + bs
                else:
                    logits = (
                        tl.sum(ik.to(tl.float32) * cq[None, :], axis=1) * pool_temp
                    )  # [PG]
                    lg = tl.reshape(logits, (PG // 4, 4))
                    e = tl.exp(lg - tl.max(lg, axis=1)[:, None])
                    w = tl.reshape(e / tl.sum(e, axis=1)[:, None], (PG,))
                    wd = tl.reshape(dots * w[None, :], (NI, PG // 4, 4))
                    cent = tl.sum(wd, axis=2)  # [NI, 4]: q . centroid
                    if BLOCK_MODE == 1:
                        bs = tl.max(cent, axis=1)
                    else:
                        bs = tl.sum(cent, axis=1) / (PG // 4)
                if HEAD_MODE == 0:
                    score = tl.sum(bs, axis=0)
                elif HEAD_MODE == 1:
                    score = tl.max(bs, axis=0)
                else:
                    mx = tl.max(bs, axis=0)
                    score = tl.log(tl.sum(tl.exp(bs - mx), axis=0)) + mx
                score = tl.where(j < init_blocks, 1e30, score)
                score = tl.where(j >= qb + 1 - local_blocks, 1e29, score)
                tl.store(obase + j + one, score + tl.zeros([1], tl.float32))
            else:
                tl.store(obase + j + one, neg)


@triton.jit
def _score_tile_kernel(
    iq_ptr,
    ic_ptr,
    ic_sb,
    ic_so,
    bt_ptr,
    stride_bt,
    seq_ptr,
    pos_ptr,
    cq_ptr,
    out_ptr,
    stride_out,
    sc,
    pool_temp,
    init_blocks,
    local_blocks,
    NB,
    BB: tl.constexpr,
    NI: tl.constexpr,
    DI: tl.constexpr,
    PG: tl.constexpr,
    BLOCK_MODE: tl.constexpr,
    HEAD_MODE: tl.constexpr,
    IC_NV: tl.constexpr,
):
    """_score_kernel's scores: BB blocks' keys in one [BB*PG, DI] tile (``IC_NV``: from
    NVFP4 pages)."""
    t = tl.program_id(0)
    j0 = tl.program_id(1) * BB
    qpos = tl.load(pos_ptr + t)
    qb = qpos // PG
    jb = j0 + tl.arange(0, BB)
    obase = out_ptr + t.to(tl.int64) * stride_out
    if j0 > qb:
        tl.store(obase + jb, tl.full([BB], float("-inf"), tl.float32), mask=jb < NB)
    else:
        seq = tl.load(seq_ptr + t)
        r = tl.arange(0, BB * PG)
        jr = j0 + r // PG
        live = (jr <= qb) & (jr < NB)
        phys = tl.load(bt_ptr + seq * stride_bt + jr, mask=live, other=0).to(tl.int64)
        ar_d = tl.arange(0, DI)
        ar_i = tl.arange(0, NI)
        if IC_NV:
            pg = ic_ptr + phys[:, None] * ic_sb
            dat = tl.load(
                pg + (r % PG)[:, None] * 64 + tl.arange(0, 64)[None, :],
                mask=live[:, None],
                other=0,
            )
            scb = tl.load(
                pg + PG * 64 + (r % PG)[:, None] * 8 + tl.arange(0, 8)[None, :],
                mask=live[:, None],
                other=0,
            )
            ik = _nvfp4_rows(dat, scb, BB * PG)
        else:
            ik = tl.load(
                ic_ptr
                + phys[:, None] * ic_sb
                + (r % PG)[:, None] * ic_so
                + ar_d[None, :],
                mask=live[:, None],
                other=0.0,
            )
        iq = tl.load(
            iq_ptr + t.to(tl.int64) * (NI * DI) + ar_i[:, None] * DI + ar_d[None, :]
        )
        dots = tl.dot(iq, tl.trans(ik.to(iq.dtype))) * sc  # [NI, BB*PG]
        if BLOCK_MODE == 0 or BLOCK_MODE == 3:
            vis = (j0 * PG + r) <= qpos
            # a block past the query scores 0 here (no NaN on the way) and -inf below
            hide = tl.where(live, float("-inf"), 0.0)
            dm = tl.reshape(tl.where(vis[None, :], dots, hide[None, :]), (NI, BB, PG))
            bs = tl.max(dm, axis=2)  # [NI, BB]
            if BLOCK_MODE == 3:
                bs = tl.log(tl.sum(tl.exp(dm - bs[:, :, None]), axis=2)) + bs
        else:
            # no token mask: only the query's own block holds keys past it, and it is pinned
            cq = tl.load(cq_ptr + ar_d).to(tl.float32)
            lg = tl.sum(ik.to(tl.float32) * cq[None, :], axis=1) * pool_temp  # [BB*PG]
            lg = tl.reshape(lg, (BB * PG // 4, 4))
            e = tl.exp(lg - tl.max(lg, axis=1)[:, None])
            w = tl.reshape(e / tl.sum(e, axis=1)[:, None], (BB * PG,))
            cent = tl.sum(tl.reshape(dots * w[None, :], (NI, BB * PG // 4, 4)), axis=2)
            cent = tl.reshape(cent, (NI, BB, PG // 4))  # q . centroid
            if BLOCK_MODE == 1:
                bs = tl.max(cent, axis=2)
            else:
                bs = tl.sum(cent, axis=2) / (PG // 4)
        if HEAD_MODE == 0:
            score = tl.sum(bs, axis=0)
        elif HEAD_MODE == 1:
            score = tl.max(bs, axis=0)
        else:
            mx = tl.max(bs, axis=0)
            score = tl.log(tl.sum(tl.exp(bs - mx[None, :]), axis=0)) + mx
        score = tl.where(jb < init_blocks, 1e30, score)
        score = tl.where(jb >= qb + 1 - local_blocks, 1e29, score)
        score = tl.where(jb <= qb, score, float("-inf"))
        tl.store(obase + jb, score, mask=jb < NB)


def scores(
    iq,
    ic,
    block_tables,
    seq_id,
    positions,
    nb,
    cq,
    init,
    local,
    sc,
    block_mode=BLOCK_TOKMAX,
    head_mode=HEAD_SUM,
    pool_temp=1.0,
    bb=None,
    tiled=True,
):
    """Indexer block scores ``[T, nb]`` (fp32): pins and ``-inf`` for future blocks applied.

    ``tiled``: score ``bb`` blocks per dot (_score_tile_kernel; by default 4 for up to 16
    tokens, else 8) instead of one (_score_kernel)."""
    t = iq.shape[0]
    out = torch.empty(t, nb, dtype=torch.float32, device=iq.device)
    assert ic.stride(-1) == 1
    nv = ic.dtype == torch.uint8
    assert tiled or not nv, "NVFP4 index pages: the tiled scorer only"
    if bb is None:
        bb = 4 if tiled and t <= 16 else 8
    grid = (t, triton.cdiv(nb, bb))
    kw = {"IC_NV": nv} if tiled else {}
    (_score_tile_kernel if tiled else _score_kernel)[grid](
        iq,
        ic,
        ic.stride(0),
        ic.stride(1),
        block_tables,
        block_tables.stride(0),
        seq_id,
        positions,
        cq if cq is not None else iq,
        out,
        out.stride(0),
        sc,
        pool_temp,
        init,
        local,
        nb,
        BB=bb,
        NI=IDX_HEADS,
        DI=IDX_DIM,
        PG=PAGE,
        BLOCK_MODE=block_mode,
        HEAD_MODE=head_mode,
        **kw,
    )
    return out


# ----------------------------------------------------------------------- attention --
@triton.jit
def _attn_kernel(
    q_ptr,
    stride_q_t,
    stride_q_h,
    kc_ptr,
    k_sb,
    k_sg,
    k_so,
    ks_ptr,
    s_sb,
    bt_ptr,
    stride_bt,
    seq_ptr,
    pos_ptr,
    sel_ptr,
    stride_sel,
    out_ptr,
    stride_out_t,
    ws_m_ptr,
    ws_l_ptr,
    ws_acc_ptr,
    scale,
    n_entries,
    H,
    SPLITS: tl.constexpr,
    CHUNK: tl.constexpr,
    DENSE: tl.constexpr,
    PG: tl.constexpr,
    NBLK: tl.constexpr,
    XK: tl.constexpr,
    KV_FP8: tl.constexpr,
    VALID_FIRST: tl.constexpr = False,
    EP2: tl.constexpr = 1,
):
    t = tl.program_id(0)
    sp = tl.program_id(1)
    ht = tl.program_id(2)
    seq = tl.load(seq_ptr + t)
    qpos = tl.load(pos_ptr + t)
    qb = qpos // PG
    hbase = ht * 16
    TN: tl.constexpr = PG * NBLK  # keys per step: NBLK whole blocks
    ar_h = tl.arange(0, 16)
    ar_c = tl.arange(0, 64)
    ar_u = tl.arange(0, TN)
    ar_b = ar_u // PG  # which block of the step a key column belongs to
    ar_o = ar_u % PG  # its offset in the block
    qrow = q_ptr + t.to(tl.int64) * stride_q_t + (hbase + ar_h)[:, None] * stride_q_h
    q0 = tl.load(qrow + ar_c[None, :])
    q1 = tl.load(qrow + 64 + ar_c[None, :])
    q2 = tl.load(qrow + 128 + ar_c[None, :])
    q3 = tl.load(qrow + 192 + ar_c[None, :])

    m_i = tl.full([16], float("-inf"), tl.float32)
    l_i = tl.zeros([16], tl.float32)
    acc1 = tl.zeros([16, 64], tl.float32)
    acc2 = tl.zeros([16, 64], tl.float32)
    acc3 = tl.zeros([16, 64], tl.float32)

    if VALID_FIRST:
        # the selection's valid blocks lead (select_logical): the splits share those,
        # not the fixed width (a short context leaves most of it -1)
        ep = tl.arange(0, EP2)
        ids = tl.load(
            sel_ptr + t.to(tl.int64) * stride_sel + ep, mask=ep < n_entries, other=-1
        )
        n_valid = tl.sum((ids >= 0).to(tl.int32), axis=0)
        chunk = tl.cdiv(tl.cdiv(n_valid, SPLITS), NBLK) * NBLK
        lo_i = sp * chunk
        hi_i = tl.minimum(lo_i + chunk, n_valid)
    else:
        lo_i = sp * CHUNK
        hi_i = tl.minimum(lo_i + CHUNK, n_entries)
    d = ar_c[:, None]
    for i in range(lo_i, hi_i, NBLK):
        e = i + ar_b
        in_range = e < hi_i
        if DENSE:
            j = e
        else:
            j = tl.load(
                sel_ptr + t.to(tl.int64) * stride_sel + e, mask=in_range, other=-1
            )
        ok = in_range & (j >= 0) & (j <= qb)  # [TN]: a real, causally reachable block
        phys = tl.load(bt_ptr + seq * stride_bt + j, mask=ok, other=0).to(tl.int64)
        # keys past the query (the rest of its own block) are never read: a cache slot
        # nobody wrote may hold anything, and p = 0 times a NaN would still be NaN
        vis = ok & ((j * PG + ar_o) <= qpos)
        kb = kc_ptr + phys[None, :] * k_sb + ar_o[None, :] * k_so
        k0 = tl.load(
            kb + (d // XK) * k_sg + (d % XK), mask=vis[None, :], other=0.0
        )  # [64, TN]
        k1 = tl.load(
            kb + ((64 + d) // XK) * k_sg + ((64 + d) % XK), mask=vis[None, :], other=0.0
        )
        k2 = tl.load(
            kb + ((128 + d) // XK) * k_sg + ((128 + d) % XK),
            mask=vis[None, :],
            other=0.0,
        )
        k3 = tl.load(
            kb + ((192 + d) // XK) * k_sg + ((192 + d) % XK),
            mask=vis[None, :],
            other=0.0,
        )
        if KV_FP8:  # per-token dequant; V is K's tail, so it shares K's scale here
            ksc = tl.load(ks_ptr + phys * s_sb + ar_o, mask=vis, other=0.0)[None, :]
            k0 = (k0.to(tl.float32) * ksc).to(q0.dtype)
            k1 = (k1.to(tl.float32) * ksc).to(q0.dtype)
            k2 = (k2.to(tl.float32) * ksc).to(q0.dtype)
            k3 = (k3.to(tl.float32) * ksc).to(q0.dtype)
        sc = (
            tl.dot(q0, k0) + tl.dot(q1, k1) + tl.dot(q2, k2) + tl.dot(q3, k3)
        )  # [16, TN]
        sc = sc * scale
        sc = tl.where(vis[None, :], sc, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(sc, axis=1))
        m_use = tl.where(
            m_new == float("-inf"), 0.0, m_new
        )  # a step with nothing visible
        alpha = tl.exp(m_i - m_use)
        p = tl.exp(sc - m_use[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        pb = p.to(k1.dtype)
        acc1 = acc1 * alpha[:, None] + tl.dot(pb, tl.trans(k1))
        acc2 = acc2 * alpha[:, None] + tl.dot(pb, tl.trans(k2))
        acc3 = acc3 * alpha[:, None] + tl.dot(pb, tl.trans(k3))
        m_i = m_new

    if SPLITS == 1:
        inv = 1.0 / l_i
        ob = out_ptr + t.to(tl.int64) * stride_out_t + (hbase + ar_h)[:, None] * 192
        tl.store(ob + ar_c[None, :], (acc1 * inv[:, None]).to(out_ptr.dtype.element_ty))
        tl.store(
            ob + 64 + ar_c[None, :], (acc2 * inv[:, None]).to(out_ptr.dtype.element_ty)
        )
        tl.store(
            ob + 128 + ar_c[None, :], (acc3 * inv[:, None]).to(out_ptr.dtype.element_ty)
        )
    else:
        wbase = (t.to(tl.int64) * SPLITS + sp) * H + hbase + ar_h
        tl.store(ws_m_ptr + wbase, m_i)
        tl.store(ws_l_ptr + wbase, l_i)
        ab = ws_acc_ptr + wbase[:, None] * 192
        tl.store(ab + ar_c[None, :], acc1)
        tl.store(ab + 64 + ar_c[None, :], acc2)
        tl.store(ab + 128 + ar_c[None, :], acc3)


@triton.jit
def _combine_kernel(
    ws_m_ptr,
    ws_l_ptr,
    ws_acc_ptr,
    out_ptr,
    stride_out_t,
    H,
    SPLITS: tl.constexpr,
    SP2: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    ar = tl.arange(0, SP2)
    ok = ar < SPLITS
    wb = (t.to(tl.int64) * SPLITS + ar) * H + h
    m = tl.load(ws_m_ptr + wb, mask=ok, other=float("-inf"))
    l = tl.load(ws_l_ptr + wb, mask=ok, other=0.0)
    mx = tl.max(m, axis=0)
    w = tl.exp(m - mx)
    ltot = tl.sum(l * w, axis=0)
    a192 = tl.arange(0, 256)
    dm = a192 < 192
    acc = tl.load(
        ws_acc_ptr + wb[:, None] * 192 + a192[None, :],
        mask=ok[:, None] & dm[None, :],
        other=0.0,
    )
    o = tl.sum(acc * w[:, None], axis=0) / ltot
    tl.store(
        out_ptr + t.to(tl.int64) * stride_out_t + h * 192 + a192,
        o.to(out_ptr.dtype.element_ty),
        mask=dm,
    )


def attention(
    q,
    kc4,
    block_tables,
    seq_id,
    positions,
    sel,
    out,
    scale,
    splits=1,
    nblk=4,
    n_blocks=None,
    k_scale=None,
    valid_first=False,
):
    """Attention of every token's local heads over its selected blocks.

    ``q`` ``[T, H, 256]`` (post norm / rope), ``sel`` ``[T, K]`` int32 block indices
    (``None``: dense, every block up to the token's own, ``n_blocks`` of them at most),
    ``out`` ``[T, H * 192]``. ``splits`` cuts the blocks over that many programs per token
    (for few tokens), ``nblk`` blocks are processed per step. An fp8 ``kc4`` takes its
    per-token scales ``k_scale`` ``[N, 1, page]``. ``valid_first``: ``sel``'s valid blocks
    lead each row (``select_logical``), and the splits divide those.
    """
    t, h, _ = q.shape
    assert h % 16 == 0 and kc4.stride(-1) == 1 and q.stride(-1) == 1
    fp8 = kc4.element_size() == 1
    assert k_scale is not None or not fp8
    dense = sel is None
    n_entries = (n_blocks if n_blocks else TOPK) if dense else sel.shape[1]
    chunk = triton.cdiv(triton.cdiv(n_entries, splits), nblk) * nblk
    splits = triton.cdiv(n_entries, chunk)  # no empty trailing splits
    if splits > 1:
        ws_m = torch.empty(t, splits, h, dtype=torch.float32, device=q.device)
        ws_l = torch.empty_like(ws_m)
        ws_acc = torch.empty(
            t, splits, h, VALUE_DIM, dtype=torch.float32, device=q.device
        )
    else:
        ws_m = ws_l = ws_acc = out
    _attn_kernel[(t, splits, h // 16)](
        q,
        q.stride(0),
        q.stride(1),
        kc4,
        kc4.stride(0),
        kc4.stride(1),
        kc4.stride(2),
        k_scale if fp8 else kc4,
        k_scale.stride(0) if fp8 else 0,
        block_tables,
        block_tables.stride(0),
        seq_id,
        positions,
        sel if not dense else q,
        sel.stride(0) if not dense else 0,
        out,
        out.stride(0),
        ws_m,
        ws_l,
        ws_acc,
        scale,
        n_entries,
        h,
        SPLITS=splits,
        CHUNK=chunk,
        DENSE=dense,
        PG=PAGE,
        NBLK=nblk,
        XK=kc4.shape[-1],
        KV_FP8=fp8,
        VALID_FIRST=valid_first and not dense,
        EP2=triton.next_power_of_2(n_entries),
    )
    if splits > 1:
        _combine_kernel[(t, h)](
            ws_m,
            ws_l,
            ws_acc,
            out,
            out.stride(0),
            h,
            SPLITS=splits,
            SP2=triton.next_power_of_2(splits),
        )
    return out


# --------------------------------------------------------- aiter paged attention --
@triton.jit
def _rows_kernel(
    sel_ptr,
    stride_sel,
    bt_ptr,
    stride_bt,
    seq_ptr,
    pos_ptr,
    rows_ptr,
    stride_rows,
    ctx_ptr,
    n_entries,
    PG: tl.constexpr,
    BW: tl.constexpr,
    DENSE: tl.constexpr,
):
    t = tl.program_id(0)
    seq = tl.load(seq_ptr + t)
    qpos = tl.load(pos_ptr + t)
    qb = qpos // PG
    e = tl.arange(0, BW)
    inr = e < n_entries
    if DENSE:
        j = tl.where(inr, e, -1)
    else:
        j = tl.load(sel_ptr + t.to(tl.int64) * stride_sel + e, mask=inr, other=-1)
    valid = (j >= 0) & (j <= qb)
    full = (valid & (j < qb)).to(tl.int32)
    n_full = tl.sum(full, axis=0)
    has_tail = tl.sum((valid & (j == qb)).to(tl.int32), axis=0)
    # full blocks keep their order and the token's own block goes last, where the
    # context length cuts off the part of it past the token
    slot = tl.where(full > 0, tl.cumsum(full, axis=0) - full, n_full)
    phys = tl.load(bt_ptr + seq * stride_bt + j, mask=valid, other=0)
    rbase = rows_ptr + t.to(tl.int64) * stride_rows
    tl.store(rbase + slot, phys, mask=valid)
    tl.store(rbase + e, tl.zeros_like(e), mask=inr & (e >= n_full + has_tail))
    tl.store(ctx_ptr + t, n_full * PG + has_tail * (qpos - qb * PG + 1))


def sparse_rows(sel, block_tables, seq_id, positions, n_blocks=None):
    """Every token's page table for ``attention_aiter``: ``(rows [T, W], ctx [T])`` int32.

    ``sel`` ``[T, K]`` int32 block indices (``None``: dense, blocks ``0..own``, ``n_blocks``
    at most); negative entries and blocks past the token's own are dropped, unused slots
    hold page 0 (in bounds, past the context length).
    """
    dense = sel is None
    t = positions.shape[0]
    w = n_blocks if dense else sel.shape[1]
    rows = torch.empty(t, w, dtype=torch.int32, device=positions.device)
    ctx = torch.empty(t, dtype=torch.int32, device=positions.device)
    _rows_kernel[(t,)](
        rows if dense else sel,
        0 if dense else sel.stride(0),
        block_tables,
        block_tables.stride(0),
        seq_id,
        positions,
        rows,
        rows.stride(0),
        ctx,
        w,
        PG=PAGE,
        BW=triton.next_power_of_2(w),
        DENSE=dense,
    )
    return rows, ctx


_topk_rows = None  # aiter's per-row top-k (tests swap it)


def select_blocks(score, positions, k):
    """The ``k`` best blocks of every row of ``scores`` among its causal ones (``j <=
    position // PAGE``) with aiter's per-row top-k, which reads only those; a row with
    fewer pads with -1. ``[T, k]`` int32, unordered."""
    global _topk_rows
    if _topk_rows is None:
        from aiter.ops.topk import top_k_per_row_decode

        _topk_rows = top_k_per_row_decode
    t = score.shape[0]
    sel = torch.empty(t, k, dtype=torch.int32, device=score.device)
    lens = (positions // PAGE + 1).to(torch.int32)
    # next_n 1: row r is a request of its own, its length lens[r]
    _topk_rows(score, 1, lens, sel, t, score.stride(0), score.stride(1), k)
    return sel


_NEG_INF_KEY = tl.constexpr(-2139095041)  # _select_rows_kernel's key of a -inf score


@triton.jit
def _select_rows_kernel(
    score_ptr,
    stride_s,
    bt_ptr,
    stride_bt,
    seq_ptr,
    pos_ptr,
    rows_ptr,
    stride_rows,
    ctx_ptr,
    NB,
    K: tl.constexpr,
    BW: tl.constexpr,
    PG: tl.constexpr,
    LOGICAL: tl.constexpr,
):
    t = tl.program_id(0)
    qpos = tl.load(pos_ptr + t)
    qb = qpos // PG
    j = tl.arange(0, BW)
    s = tl.load(
        score_ptr + t.to(tl.int64) * stride_s + j,
        mask=(j < NB) & (j <= qb),
        other=float("-inf"),
    )
    # fp32 bits as an int ordered like the floats, the block index (reversed: the lower
    # wins a tie) below them: K distinct keys
    b = s.to(tl.int32, bitcast=True)
    key = (b ^ ((b >> 31) & 0x7FFFFFFF)).to(tl.int64)
    top = tl.topk((key << 32) | (BW - 1 - j).to(tl.int64), K)
    jj = BW - 1 - top.to(tl.int32)
    valid = (
        top >> 32
    ) > _NEG_INF_KEY  # a causal block: fewer than K leave -inf ones in
    rbase = rows_ptr + t.to(tl.int64) * stride_rows
    e = tl.arange(0, K)
    if LOGICAL:
        # the block ids themselves, best first: the valid ones lead, -1 after
        tl.store(rbase + e, tl.where(valid, jj, -1))
        return
    seq = tl.load(seq_ptr + t)
    full = (valid & (jj < qb)).to(tl.int32)
    n_full = tl.sum(full, axis=0)
    has_tail = tl.sum((valid & (jj == qb)).to(tl.int32), axis=0)
    # as _rows_kernel: the token's own block last, cut by the context length
    slot = tl.where(full > 0, tl.cumsum(full, axis=0) - full, n_full)
    phys = tl.load(bt_ptr + seq * stride_bt + jj, mask=valid, other=0)
    e = tl.arange(0, K)
    rbase = rows_ptr + t.to(tl.int64) * stride_rows
    tl.store(rbase + slot, phys, mask=valid)
    tl.store(rbase + e, tl.zeros_like(e), mask=e >= n_full + has_tail)
    tl.store(ctx_ptr + t, n_full * PG + has_tail * (qpos - qb * PG + 1))


def select_rows(score, block_tables, seq_id, positions, k, num_warps=8):
    """``select_blocks`` then ``sparse_rows`` in one launch: every token's page table for
    ``attention_aiter`` over the ``k`` (a power of two) best of its causal blocks (ties: the
    lower block). ``(rows [T, k], ctx [T])`` int32."""
    assert k & (k - 1) == 0, k
    t, nb = score.shape
    rows = torch.empty(t, k, dtype=torch.int32, device=score.device)
    ctx = torch.empty(t, dtype=torch.int32, device=score.device)
    _select_rows_kernel[(t,)](
        score,
        score.stride(0),
        block_tables,
        block_tables.stride(0),
        seq_id,
        positions,
        rows,
        rows.stride(0),
        ctx,
        nb,
        K=k,
        BW=max(triton.next_power_of_2(nb), k),
        PG=PAGE,
        LOGICAL=False,
        num_warps=num_warps,
    )
    return rows, ctx


def select_logical(score, positions, k, num_warps=8):
    """``select_rows``' selection as block ids, for the Triton attention: ``[T, k]`` int32,
    best first, the token's valid (causal) blocks leading and -1 after."""
    assert k & (k - 1) == 0, k
    t, nb = score.shape
    sel = torch.empty(t, k, dtype=torch.int32, device=score.device)
    _select_rows_kernel[(t,)](
        score,
        score.stride(0),
        sel,
        0,
        positions,
        positions,
        sel,
        sel.stride(0),
        sel,
        nb,
        K=k,
        BW=max(triton.next_power_of_2(nb), k),
        PG=PAGE,
        LOGICAL=True,
        num_warps=num_warps,
    )
    return sel


_pa_decode = None  # aiter's paged-attention decode as ATOM wraps it (tests swap it)


def _pa():
    global _pa_decode
    if _pa_decode is None:
        from atom.model_ops.base_attention import run_pa_decode

        _pa_decode = run_pa_decode
    return _pa_decode


def _default_partitions(t: int) -> int:
    from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits

    # at most one program per CU: aiter plans two, too many for programs of 16 heads
    units = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    return min(get_recommended_splits(t, 1), max(1, units // t))


def attention_aiter(
    q,
    k_cache,
    v_cache,
    rows,
    ctx,
    out,
    scale,
    k_scale=None,
    v_scale=None,
    fp8_compute=True,
    partitions=None,
    value_dim=VALUE_DIM,
):
    """``attention`` on aiter's paged-attention decode (gluon, persistent dispatch), every
    token a sequence of its own over its ``sparse_rows`` table.

    ``k_cache`` ``[N, 1, 256/x, page, x]`` and ``v_cache`` ``[N, 1, page/x, 256, x]`` are the
    5-D caches (V in the first ``value_dim`` rows), ``out`` ``[T, H * value_dim]``. An fp8
    cache takes its per-token scales ``[N, 1, page]``; ``fp8_compute`` runs the dots in fp8
    as well.
    """
    t, h, _ = q.shape
    fp8 = k_cache.element_size() == 1
    assert (k_scale is not None and v_scale is not None) or not fp8
    if partitions is None:
        partitions = _default_partitions(t)
    partitions = max(1, min(partitions, triton.cdiv(rows.shape[1] * PAGE, 256)))
    _pa()(
        out.view(t, h, value_dim),
        q,
        k_cache,
        v_cache[:, :, :, :value_dim, :],
        ctx,
        rows,
        scale,
        1,
        partitions,
        256,
        k_cache.dtype if (fp8 and fp8_compute) else q.dtype,
        None,
        k_scale.unsqueeze(-1) if fp8 else None,
        v_scale.unsqueeze(-1) if fp8 else None,
        ps=True,
    )
    return out


_day0_prefill = None  # amd-kernels-day0's fp8_sparse_prefill; False: not installed


def day0_prefill_available() -> bool:
    """Whether amd-kernels-day0's FlyDSL sparse prefill (``attention_day0``) imports."""
    global _day0_prefill
    if _day0_prefill is None:
        try:
            from amd_kernels.attention.fp8_sparse_prefill_gqa16_d256_v192_b16_top128.host import (
                fp8_sparse_prefill,
            )
        except ImportError:
            _day0_prefill = False
        else:
            _day0_prefill = fp8_sparse_prefill
    return bool(_day0_prefill)


def attention_day0(q, kc4, block_tables, seq_id, positions, sel, out, scale):
    """``attention`` on amd-kernels-day0's FlyDSL fp8 sparse prefill, over the K-only cache
    as a plain fp8 cast (``kc4`` ``[N, 1, page, 256]``): ``q`` ``[T, 16, 256]`` is cast to
    fp8 as well. ``sel`` ``[T, 128]`` block ids, valid first and -1 after (``select_logical``),
    or None: every causal block (at most 128)."""
    t = q.shape[0]
    if sel is None:
        j = torch.arange(TOPK, device=q.device, dtype=torch.int32)
        sel = torch.where(j[None, :] <= (positions // PAGE)[:, None], j[None, :], -1)
        sel = sel.to(torch.int32)
    _day0_prefill(
        q.to(torch.float8_e4m3fn),
        kc4[:, 0],
        sel,
        block_tables,
        seq_id.to(torch.int32),
        positions.to(torch.int32),
        scale,
        out.view(t, IDX_HEADS, VALUE_DIM),
    )
    return out
