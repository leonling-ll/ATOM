# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""4-bit (NVFP4) lightning-indexer kernels for MiniMax-M3.1-Pro.

The index cache holds one NVFP4 page per 16-token block: e2m1 values with one e4m3 scale per
16 elements, laid out ``[R x 64 B codes (low nibble = even element)][R x 8 B scales]``.
``R = 16`` stores every token's key (1152 B per block); ``R = 4`` stores the block's 4
softmax-pooled centroids, computed once when the block is written (288 B per block).
The scorers also read the raw caches of the requirements' format list -- ``bf16`` / ``fp8``
``[N, 16, 128]`` and ``bf16_centroid`` ``[N, 4, 128]`` -- with bf16 MFMAs, so the fused
selection does not depend on the 4-bit format.

* ``write``      quantizes full blocks of bf16 index keys into their pages (per-token or
                 centroid pages, or bf16 centroids);
* ``score``      the block scores ``[T, nb]`` fp32 of ``pro_kernels.scores`` (pool4 / max,
                 heads max; pins and ``-inf`` applied) read from a 4-bit cache;
* ``score_lse``  the lse / lse reading (per-token pages only);
* ``select``     the scores fused with the top-k: each token's ``k`` best blocks, without
                 the ``[T, nb]`` score table.

q4k4 (requirement 4.1, the checkpoint's ``sparse_index_q4k4``): ``score``, ``score_lse``
and ``select`` quantize the index q to NVFP4 in the kernel, by the keys' rule (UT:
``PRO_NEW_INDEX_Q4=1``). The ``*_bf16q`` variants use q as given -- bf16, or q the prep
already wrote in 4-bit (UT: ``PRO_NEW_INDEX_Q4`` unset).

MI350 has no NVFP4 MFMA (its FP4 MFMA takes MXFP4: an E8M0 scale per 32 elements), so the
kernels dequantize to 16-bit values and run 16-bit MFMAs with fp32 accumulation. Every
e2m1 value times its e4m3 scale has at most 6 significant bits and lies in [2^-10, 2688],
so this is exact. On gfx950 ``v_cvt_scalef32_pk_f16_fp4`` converts a byte to 2 fp16 at
scale 1 (it applies only a scale's exponent) and ``v_pk_mul_f16`` applies the e4m3 scale;
elsewhere (``hw=False``) integer bit tricks produce bf16, the even and odd columns of q
dotted with the low and high nibbles.

Signatures follow the kernel requirements (``PRO_NEW_INDEX_SCORE`` / ``_WRITE`` /
``_SELECT``):

    score(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local, pool_temp)
    write(keys, pages, cq, pool_temp, index_cache)
    select(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local, pool_temp, k)
"""

import torch
import triton
import triton.language as tl

PAGE = 16
GROUP = 4  # tokens per centroid
CENTROIDS = PAGE // GROUP
IDX_HEADS = 16
IDX_DIM = 128
NVFP4_GROUP = 16
CODE_BYTES = IDX_DIM // 2  # per row
SCALE_BYTES = IDX_DIM // NVFP4_GROUP  # per row
# jit'ed code may only read module globals that are constexpr
CB = tl.constexpr(CODE_BYTES)
SB = tl.constexpr(SCALE_BYTES)

MODE_POOL4MAX = 0  # per-token pages: softmax-pool 4 groups of 4, max over them
MODE_CENTROID = 1  # centroid pages: max over the block's 4 rows
MODE_LSE = 2  # per-token pages: log-sum-exp over visible tokens, then over heads


def page_bytes(rows: int) -> int:
    return rows * (CODE_BYTES + SCALE_BYTES)


RAW_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)


def _layout(ic: torch.Tensor):
    """``(rows per block, raw)`` of an index cache: NVFP4 pages ``[N, 1152]`` (per-token) /
    ``[N, 288]`` (centroids) uint8, or raw rows ``[N, 16 | 4, 128]`` bf16 / fp8 (the
    ``bf16`` / ``fp8`` / ``bf16_centroid`` formats)."""
    if ic.dtype == torch.uint8 and ic.dim() == 2:
        if ic.shape[1] == page_bytes(PAGE):
            return PAGE, False
        if ic.shape[1] == page_bytes(CENTROIDS):
            return CENTROIDS, False
    if (
        ic.dtype in RAW_DTYPES
        and ic.dim() == 3
        and ic.shape[1] in (PAGE, CENTROIDS)
        and ic.shape[2] == IDX_DIM
        and ic.stride(2) == 1
        and ic.stride(1) == IDX_DIM
    ):
        return ic.shape[1], True
    raise ValueError(f"not an index cache: {tuple(ic.shape)} {ic.dtype}")


def _rows_of(ic: torch.Tensor) -> int:
    return _layout(ic)[0]


# ---------------------------------------------------------------------------- helpers --
@triton.jit
def _nvfp4_scale(amax):
    """E4M3(clamp(amax / 6, 2^-9, 448)) as fp8 (NaN stays NaN)."""
    s = amax / 6.0
    s = tl.where(s < 0.001953125, 0.001953125, s)
    s = tl.where(s > 448.0, 448.0, s)
    return s.to(tl.float8e4nv)


@triton.jit
def _e2m1_code(a, s):
    """Index of the e2m1 magnitude nearest ``a / s`` (ties to the smaller), from the
    midpoints between magnitudes, as the reference counts them."""
    c = (a > 0.25 * s).to(tl.int32)
    c += (a > 0.75 * s).to(tl.int32)
    c += (a > 1.25 * s).to(tl.int32)
    c += (a > 1.75 * s).to(tl.int32)
    c += (a > 2.5 * s).to(tl.int32)
    c += (a > 3.5 * s).to(tl.int32)
    c += (a > 5.0 * s).to(tl.int32)
    return c


@triton.jit
def _fake_nvfp4(x, ROWS: tl.constexpr, D: tl.constexpr):
    """fp32 ``[ROWS, D]`` through NVFP4 and back (exact scale x e2m1)."""
    g = tl.reshape(x, (ROWS, D // 16, 16))
    a = tl.abs(g)
    s = _nvfp4_scale(tl.max(a, axis=2)).to(tl.float32)[:, :, None]
    c = _e2m1_code(a, s).to(tl.float32)
    mag = tl.where(c < 4.0, c * 0.5, tl.where(c < 6.0, c - 2.0, c * 2.0 - 8.0))
    v = mag * s
    return tl.reshape(tl.where(g < 0, -v, v), (ROWS, D))


@triton.jit
def _e2m1_raw(nib):
    """e2m1 nibble (int32) -> its value / 2^14, exactly, through fp16 bits: the 3 magnitude
    bits land on fp16's low exponent bits and top mantissa bit (0.5 becomes the subnormal
    2^-15), the sign on fp16's sign."""
    bits = ((nib & 8) << 12) | ((nib & 7) << 9)
    return bits.to(tl.int16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _dequant_halves(codes, scales, R: tl.constexpr):
    """codes ``[R, 64]`` uint8 and scales ``[R, 8]`` uint8 -> the even and the odd elements
    ``[R, 64]`` bf16 (exact). Byte ``b`` of a row holds elements ``2b`` (low nibble) and
    ``2b + 1``, both in scale group ``b // 8``."""
    c = codes.to(tl.int32)
    s = scales.to(tl.float8e4nv, bitcast=True).to(tl.float32) * 16384.0
    s = tl.reshape(tl.broadcast_to(s[:, :, None], (R, 8, 8)), (R, 64))
    lo = (_e2m1_raw(c & 15) * s).to(tl.bfloat16)
    hi = (_e2m1_raw(c >> 4) * s).to(tl.bfloat16)
    return lo, hi


@triton.jit
def _cvt_byte(word, sc2, SEL: tl.constexpr):
    """gfx950: byte ``SEL`` of ``word`` (2 e2m1) -> 2 fp16 times the fp16 scale held twice
    in ``sc2``. ``v_cvt_scalef32_pk_f16_fp4`` applies only the exponent of its scale, so it
    converts at scale 1 and ``v_pk_mul_f16`` applies the e4m3 scale (exact in fp16)."""
    if SEL == 0:
        return tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_f16_fp4 $0, $1, 1.0\nv_pk_mul_f16 $0, $0, $2",
            "=&v,v,v", [word, sc2], dtype=tl.int32, is_pure=True, pack=1)  # fmt: skip
    elif SEL == 1:
        return tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_f16_fp4 $0, $1, 1.0 op_sel:[1,0,0]\nv_pk_mul_f16 $0, $0, $2",
            "=&v,v,v", [word, sc2], dtype=tl.int32, is_pure=True, pack=1)  # fmt: skip
    elif SEL == 2:
        return tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_f16_fp4 $0, $1, 1.0 op_sel:[0,1,0]\nv_pk_mul_f16 $0, $0, $2",
            "=&v,v,v", [word, sc2], dtype=tl.int32, is_pure=True, pack=1)  # fmt: skip
    else:
        return tl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_f16_fp4 $0, $1, 1.0 op_sel:[1,1,0]\nv_pk_mul_f16 $0, $0, $2",
            "=&v,v,v", [word, sc2], dtype=tl.int32, is_pure=True, pack=1)  # fmt: skip


@triton.jit
def _lo16(p):
    return p.to(tl.int16).to(tl.float16, bitcast=True)


@triton.jit
def _hi16(p):
    return (p >> 16).to(tl.int16).to(tl.float16, bitcast=True)


@triton.jit
def _dequant_hw(words, scales, R: tl.constexpr):
    """gfx950: codes ``[R, 16]`` int32 (4 bytes each) and scales ``[R, 8]`` uint8 -> the
    keys ``[R, 128]`` fp16 in natural order (exact). Word ``w`` byte ``b`` holds elements
    ``8w + 2b`` (low nibble) and ``8w + 2b + 1``; scale group ``w // 2``."""
    s = scales.to(tl.float8e4nv, bitcast=True).to(tl.float16)
    sb = s.to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF
    sc2 = tl.reshape(tl.broadcast_to((sb | (sb << 16))[:, :, None], (R, 8, 2)), (R, 16))
    p0 = _cvt_byte(words, sc2, 0)
    p1 = _cvt_byte(words, sc2, 1)
    p2 = _cvt_byte(words, sc2, 2)
    p3 = _cvt_byte(words, sc2, 3)
    # [R, 16, i, j] with byte b = 2i + j, then the nibble h last: element 8w + 2b + h
    lo = tl.join(tl.join(_lo16(p0), _lo16(p2)), tl.join(_lo16(p1), _lo16(p3)))
    hi = tl.join(tl.join(_hi16(p0), _hi16(p2)), tl.join(_hi16(p1), _hi16(p3)))
    return tl.reshape(tl.join(lo, hi), (R, 128))


# ----------------------------------------------------------------------------- scores --
@triton.jit
def _prep_q(
    iq_ptr,
    s_iq_t,
    s_iq_h,
    cq_ptr,
    t0,
    T,
    QT: tl.constexpr,
    NI: tl.constexpr,
    DI: tl.constexpr,
    MODE: tl.constexpr,
    Q4: tl.constexpr,
    HW: tl.constexpr,
    RAW: tl.constexpr,
):
    """The q tile's MFMA operands ``(qa, qb, ca, cb)``: bf16 (raw caches, ``RAW``) or fp16
    (NVFP4 on gfx950, ``HW``) q ``[QT*16, DI]`` and ``index_compress_query`` as row 0 of a
    16-row operand (``qb`` / ``cb`` unused); otherwise bf16 even / odd columns of each.
    q4k4 (``Q4``): q through NVFP4 first."""
    M: tl.constexpr = QT * NI
    HALF: tl.constexpr = DI // 2
    rm = tl.arange(0, M)
    tm = t0 + rm // NI
    d = tl.arange(0, DI)
    q = tl.load(
        iq_ptr
        + tm[:, None].to(tl.int64) * s_iq_t
        + (rm % NI)[:, None] * s_iq_h
        + d[None, :],
        mask=(tm < T)[:, None],
        other=0.0,
    ).to(tl.float32)
    if Q4:
        q = _fake_nvfp4(q, M, DI)
    r16 = tl.arange(0, 16)[:, None]
    if RAW:
        qa = q.to(tl.bfloat16)
        qb = qa
        ca = qa
        if MODE == 0:
            cq = tl.load(cq_ptr + d).to(tl.float32)
            ca = tl.where(r16 == 0, cq[None, :], 0.0).to(tl.bfloat16)
        cb = ca
    elif HW:
        qa = q.to(tl.float16)
        qb = qa
        ca = qa
        if MODE == 0:
            cq = tl.load(cq_ptr + d).to(tl.float32)
            ca = tl.where(r16 == 0, cq[None, :], 0.0).to(tl.float16)
        cb = ca
    else:
        qe, qo = tl.split(tl.reshape(q, (M, HALF, 2)))
        qa = qe.to(tl.bfloat16)
        qb = qo.to(tl.bfloat16)
        ca = qa
        cb = qb
        if MODE == 0:
            cq = tl.reshape(tl.load(cq_ptr + d).to(tl.float32), (HALF, 2))
            cqe, cqo = tl.split(cq)
            ca = tl.where(r16 == 0, cqe[None, :], 0.0).to(tl.bfloat16)
            cb = tl.where(r16 == 0, cqo[None, :], 0.0).to(tl.bfloat16)
    return qa, qb, ca, cb


@triton.jit
def _tile_score(
    j0,
    seq,
    qbmax,
    qa,
    qb,
    ca,
    cb,
    ic_ptr,
    s_ic,
    bt_ptr,
    s_bt,
    NB,
    pos_t,
    qb_t,
    sc,
    pool_temp,
    init_blocks,
    local_blocks,
    QT: tl.constexpr,
    BB: tl.constexpr,
    ROWS: tl.constexpr,
    MODE: tl.constexpr,
    HW: tl.constexpr,
    RAW: tl.constexpr,
    NI: tl.constexpr,
    DI: tl.constexpr,
):
    """Scores ``[QT, BB]`` of blocks ``j0 .. j0 + BB - 1`` of sequence ``seq`` for the q tile:
    pins applied, ``-inf`` past each token's own block (blocks past ``qbmax`` are not read)."""
    M: tl.constexpr = QT * NI
    R: tl.constexpr = BB * ROWS
    jb = j0 + tl.arange(0, BB)
    if j0 <= qbmax:
        rr = tl.arange(0, R)
        jr = j0 + rr // ROWS
        live = (jr <= qbmax) & (jr < NB)
        phys = tl.load(bt_ptr + seq * s_bt + jr, mask=live, other=0)
        if RAW:  # bf16 / fp8 rows (fp8 -> bf16 is exact)
            k = tl.load(
                ic_ptr
                + phys.to(tl.int64)[:, None] * s_ic
                + (rr % ROWS)[:, None] * DI
                + tl.arange(0, DI)[None, :],
                mask=live[:, None],
                other=0.0,
            ).to(tl.bfloat16)
            dots = tl.dot(qa, tl.trans(k))
            if MODE == 0:
                lg = tl.dot(ca, tl.trans(k))
        elif HW:
            row = ic_ptr + phys.to(tl.int64) * s_ic + (rr % ROWS) * CB
            srow = ic_ptr + phys.to(tl.int64) * s_ic + ROWS * CB + (rr % ROWS) * SB
            scales = tl.load(srow[:, None] + tl.arange(0, SB)[None, :], mask=live[:, None], other=0)
            words = tl.load(
                row.to(tl.pointer_type(tl.int32))[:, None] + tl.arange(0, CB // 4)[None, :],
                mask=live[:, None],
                other=0,
            )
            k16 = _dequant_hw(words, scales, R)
            dots = tl.dot(qa, tl.trans(k16))
            if MODE == 0:
                lg = tl.dot(ca, tl.trans(k16))
        else:
            row = ic_ptr + phys.to(tl.int64) * s_ic + (rr % ROWS) * CB
            srow = ic_ptr + phys.to(tl.int64) * s_ic + ROWS * CB + (rr % ROWS) * SB
            scales = tl.load(srow[:, None] + tl.arange(0, SB)[None, :], mask=live[:, None], other=0)
            codes = tl.load(
                row[:, None] + tl.arange(0, CB)[None, :], mask=live[:, None], other=0
            )
            ke, ko = _dequant_halves(codes, scales, R)
            dots = tl.dot(qa, tl.trans(ke)) + tl.dot(qb, tl.trans(ko))
            if MODE == 0:
                lg = tl.dot(ca, tl.trans(ke)) + tl.dot(cb, tl.trans(ko))
        if MODE == 0:
            # pool weights: softmax over each group of 4 (exp2 and a fast reciprocal: the
            # IEEE divide costs more than the dequant here)
            r16 = tl.arange(0, 16)[:, None]
            lg = tl.sum(tl.where(r16 == 0, lg, 0.0), axis=0) * (pool_temp * 1.4426950408889634)
            lg = tl.reshape(lg, (R // 4, 4))
            e = tl.math.exp2(lg - tl.max(lg, axis=1)[:, None])
            rs = tl.fdiv(1.0, tl.sum(e, axis=1), ieee_rounding=False)
            w = tl.reshape(e * rs[:, None], (R,))
            cent = tl.sum(tl.reshape(dots * w[None, :], (M, R // 4, 4)), axis=2)
            bs = tl.max(tl.reshape(cent, (M, BB, 4)), axis=2)
            score = tl.max(tl.reshape(bs, (QT, NI, BB)), axis=1) * sc
        elif MODE == 1:
            bs = tl.max(tl.reshape(dots, (M, BB, ROWS)), axis=2)
            score = tl.max(tl.reshape(bs, (QT, NI, BB)), axis=1) * sc
        else:
            tokpos = jr * 16 + rr % 16
            dm = tl.reshape(dots * sc, (QT, NI, R))
            vis = tokpos[None, :] <= pos_t[:, None]
            dm = tl.where(vis[:, None, :], dm, float("-inf"))
            dm = tl.reshape(dm, (QT, NI, BB, ROWS))
            mx = tl.max(dm, axis=3)
            mx_s = tl.where(mx == float("-inf"), 0.0, mx)
            bs = tl.log(tl.sum(tl.exp(dm - mx_s[:, :, :, None]), axis=3)) + mx_s
            hm = tl.max(bs, axis=1)
            hm_s = tl.where(hm == float("-inf"), 0.0, hm)
            score = tl.log(tl.sum(tl.exp(bs - hm_s[:, None, :]), axis=1)) + hm_s
    else:
        score = tl.full((QT, BB), float("-inf"), tl.float32)
    score = tl.where(jb[None, :] < init_blocks, 1e30, score)
    score = tl.where(jb[None, :] >= qb_t[:, None] + 1 - local_blocks, 1e29, score)
    return tl.where(jb[None, :] <= qb_t[:, None], score, float("-inf"))


@triton.jit
def _score_kernel(
    iq_ptr,
    s_iq_t,
    s_iq_h,
    ic_ptr,
    s_ic,
    bt_ptr,
    s_bt,
    seq_ptr,
    pos_ptr,
    cq_ptr,
    out_ptr,
    s_out,
    T,
    NB,
    FILL,
    sc,
    pool_temp,
    init_blocks,
    local_blocks,
    CAUSAL: tl.constexpr,
    QT: tl.constexpr,
    BB: tl.constexpr,
    ROWS: tl.constexpr,
    MODE: tl.constexpr,
    Q4: tl.constexpr,
    HW: tl.constexpr,
    RAW: tl.constexpr,
    NI: tl.constexpr,
    DI: tl.constexpr,
):
    """Program (q tile, ``pid_s`` of ``NS``): ``QT`` consecutive query tokens, ``BB`` blocks
    (``BB * ROWS`` cache rows) per dot. The program strides over the tiles that hold the
    tokens' causal blocks (tiles ``pid_s, pid_s + NS, ...``), so the work follows the
    context, not ``NB`` (a CUDA graph's block-table width). Unless ``CAUSAL``, it then fills
    its share ``[pid_s * FILL, +FILL)`` of the columns past those tiles with ``-inf``, in
    wide stores. Tokens of the tile that belong to different sequences are handled one run
    of equal ``seq_id`` at a time (tokens of a sequence are contiguous)."""
    FW: tl.constexpr = 256  # -inf columns a store
    t0 = tl.program_id(0) * QT
    pid_s = tl.program_id(1)
    ns = tl.num_programs(1)
    qa, qb, ca, cb = _prep_q(iq_ptr, s_iq_t, s_iq_h, cq_ptr, t0, T, QT, NI, DI, MODE, Q4, HW, RAW)
    tq = t0 + tl.arange(0, QT)
    tval = tq < T
    pos_t = tl.load(pos_ptr + tq, mask=tval, other=0)
    seq_t = tl.load(seq_ptr + tq, mask=tval, other=-1)
    qb_t = pos_t // 16
    jt = tl.arange(0, BB)
    for sub in range(QT):
        t_sub = t0 + sub
        if t_sub < T:
            s_sub = tl.load(seq_ptr + t_sub)
            prev = tl.load(seq_ptr + t_sub - 1, mask=sub > 0, other=-1)
            if (sub == 0) | (s_sub != prev):
                mine = tval & (seq_t == s_sub)
                qbmax = tl.max(tl.where(mine, qb_t, -1), axis=0)
                ntile = qbmax // BB + 1  # tiles holding the run's causal blocks
                orow = out_ptr + tq[:, None].to(tl.int64) * s_out
                for ti in range(pid_s, ntile, ns):
                    j0 = ti * BB
                    score = _tile_score(
                        j0, s_sub, qbmax, qa, qb, ca, cb, ic_ptr, s_ic, bt_ptr, s_bt, NB,
                        pos_t, qb_t, sc, pool_temp, init_blocks, local_blocks,
                        QT, BB, ROWS, MODE, HW, RAW, NI, DI,
                    )  # fmt: skip
                    jb = j0 + jt
                    tl.store(orow + jb[None, :], score, mask=mine[:, None] & (jb < NB)[None, :])
                if not CAUSAL:
                    f1 = tl.minimum(NB, (pid_s + 1) * FILL)
                    neg = tl.full((QT, FW), float("-inf"), tl.float32)
                    for c0 in range(tl.maximum(ntile * BB, pid_s * FILL), f1, FW):
                        cols = c0 + tl.arange(0, FW)
                        tl.store(orow + cols[None, :], neg, mask=mine[:, None] & (cols < f1)[None, :])


# --------------------------------------------------------------- scores fused with top-k --
KEY_NONE = tl.constexpr(-(1 << 63))
KEY_NEGINF = tl.constexpr(-2139095041)  # _key's high word for a -inf score


@triton.jit
def _key(score, j):
    """int64 keys ordering (score, -block): the fp32 bits made order-preserving as int32 in
    the high word, ``2^31 - 1 - j`` in the low word, so equal scores rank the lower block
    first."""
    b = score.to(tl.int32, bitcast=True)
    k32 = b ^ ((b >> 31) & 0x7FFFFFFF)
    return (k32.to(tl.int64) << 32) | (0x7FFFFFFF - j).to(tl.int64)


@triton.jit
def _select_kernel(
    iq_ptr,
    s_iq_t,
    s_iq_h,
    ic_ptr,
    s_ic,
    bt_ptr,
    s_bt,
    seq_ptr,
    pos_ptr,
    cq_ptr,
    out_ptr,
    s_out_t,
    T,
    NB,
    sc,
    pool_temp,
    init_blocks,
    local_blocks,
    FINAL: tl.constexpr,
    K: tl.constexpr,
    QT: tl.constexpr,
    BB: tl.constexpr,
    ROWS: tl.constexpr,
    MODE: tl.constexpr,
    Q4: tl.constexpr,
    HW: tl.constexpr,
    RAW: tl.constexpr,
    NI: tl.constexpr,
    DI: tl.constexpr,
):
    """Program (q tile, ``pid_s`` of ``NS``): strides over the tiles holding the tokens'
    causal blocks (``pid_s, pid_s + NS, ...``; the work follows the context, not ``NB``) and
    keeps each token's top-``K`` keys in registers -- ``K`` blocks are scored into a buffer,
    then the buffer and the running top are reduced to the top ``K`` again. ``FINAL``
    (``NS`` = 1): writes logical blocks ``[T, K]`` (-1 = unused); otherwise the keys
    ``[T, NS, K]`` for ``_merge_kernel``. The ``[T, nb]`` scores never reach memory."""
    CHT: tl.constexpr = K // BB  # tiles per chunk of K blocks
    t0 = tl.program_id(0) * QT
    pid_s = tl.program_id(1)
    ns = tl.num_programs(1)
    qa, qb, ca, cb = _prep_q(iq_ptr, s_iq_t, s_iq_h, cq_ptr, t0, T, QT, NI, DI, MODE, Q4, HW, RAW)
    tq = t0 + tl.arange(0, QT)
    tval = tq < T
    pos_t = tl.load(pos_ptr + tq, mask=tval, other=0)
    seq_t = tl.load(seq_ptr + tq, mask=tval, other=-1)
    qb_t = pos_t // 16
    jt = tl.arange(0, BB)
    slot = tl.arange(0, CHT)[None, :, None]
    for sub in range(QT):
        t_sub = t0 + sub
        if t_sub < T:
            s_sub = tl.load(seq_ptr + t_sub)
            prev = tl.load(seq_ptr + t_sub - 1, mask=sub > 0, other=-1)
            if (sub == 0) | (s_sub != prev):
                mine = tval & (seq_t == s_sub)
                qbmax = tl.max(tl.where(mine, qb_t, -1), axis=0)
                ntile = qbmax // BB + 1  # tiles holding the run's causal blocks
                top = tl.full((QT, K), KEY_NONE, tl.int64)
                for c in range(pid_s, ntile, ns * CHT):
                    buf = tl.full((QT, CHT, BB), KEY_NONE, tl.int64)
                    for ti in range(CHT):
                        # past the causal tiles _tile_score reads nothing and gives -inf
                        j0 = (c + ti * ns) * BB
                        score = _tile_score(
                            j0, s_sub, qbmax, qa, qb, ca, cb, ic_ptr, s_ic, bt_ptr, s_bt, NB,
                            pos_t, qb_t, sc, pool_temp, init_blocks, local_blocks,
                            QT, BB, ROWS, MODE, HW, RAW, NI, DI,
                        )  # fmt: skip
                        jb = j0 + jt
                        key = _key(score, jb)
                        buf = tl.where(slot == ti, key[:, None, :], buf)
                    cand = tl.reshape(tl.join(top, tl.reshape(buf, (QT, K))), (QT, 2 * K))
                    top = tl.topk(cand, K)
                kk = tl.arange(0, K)
                if FINAL:
                    out = _decode(top)
                else:
                    out = top
                tl.store(
                    out_ptr + tq[:, None].to(tl.int64) * s_out_t + pid_s * K + kk[None, :],
                    out,
                    mask=mine[:, None],
                )


@triton.jit
def _decode(top):
    """Keys -> logical blocks (int32), -1 for an empty slot or a -inf score."""
    hi = (top >> 32).to(tl.int32)
    j = 0x7FFFFFFF - (top & 0x7FFFFFFF).to(tl.int32)
    return tl.where(hi > KEY_NEGINF, j, -1)


@triton.jit
def _merge_kernel(cand_ptr, sel_ptr, NS, K: tl.constexpr, NS2: tl.constexpr):
    """Per token: the top ``K`` of its ``NS`` ranges' keys ``[NS, K]`` -> logical blocks."""
    t = tl.program_id(0)
    r = tl.arange(0, NS2)[:, None]
    kk = tl.arange(0, K)[None, :]
    keys = tl.load(
        cand_ptr + t.to(tl.int64) * NS * K + r * K + kk, mask=r < NS, other=KEY_NONE
    )
    top = tl.topk(tl.reshape(keys, (NS2 * K,)), K)
    tl.store(sel_ptr + t.to(tl.int64) * K + tl.arange(0, K), _decode(top))


_DEV = {}


def _dev_info(device):
    """(CU count, gfx950?) of ``device``, cached."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _DEV:
        p = torch.cuda.get_device_properties(idx)
        _DEV[idx] = (p.multi_processor_count, getattr(p, "gcnArchName", "").startswith("gfx950"))
    return _DEV[idx]


def _score(
    iq,
    ic,
    block_tables,
    seq_id,
    positions,
    nb,
    cq,
    init,
    local,
    pool_temp,
    mode,
    q4,
    qt=None,
    bb=None,
    waves=None,
    num_warps=4,
    hw=None,
    causal=False,
    index_scale=1.0,
):
    """``causal``: leave the columns past each token's own block unwritten (a selection
    that reads only the causal blocks needs no more). ``index_scale``: the scores' scale
    (``ProVariants.index_scale``)."""
    rows, raw = _layout(ic)
    if mode == MODE_CENTROID:
        assert rows == CENTROIDS, "centroid scores need a centroid cache"
    else:
        assert rows == PAGE, "this reading needs per-token pages"
    t = iq.shape[0]
    assert iq.shape[1:] == (IDX_HEADS, IDX_DIM) and iq.stride(-1) == 1
    out = torch.empty(t, nb, dtype=torch.float32, device=iq.device)
    if t == 0 or nb == 0:
        return out
    if qt is None:
        qt = 1 if t <= 64 else 8
    if bb is None:
        # per-token pages: 8 blocks a dot for q tiles of up to 4 rows (decode, k3 verify),
        # 4 for 8-row tiles (k7 verify, prefill) -- measured on MI350
        bb = (8 if qt <= 4 else 4) if rows == PAGE else 16
    if waves is None:
        # programs a CU: a multi-row q tile (verify, prefill) costs more to set up, so give
        # each program more tiles (measured on #67's verify points, MI350)
        waves = 2 if qt > 1 else 4
    nq = triton.cdiv(t, qt)
    tiles = triton.cdiv(nb, bb)
    cus, gfx950 = _dev_info(iq.device)
    if hw is None:
        hw = gfx950
    nsplit = max(1, min(tiles, triton.cdiv(waves * cus, nq)))
    _score_kernel[(nq, nsplit)](
        iq,
        iq.stride(0),
        iq.stride(1),
        ic,
        ic.stride(0),
        block_tables,
        block_tables.stride(0),
        seq_id,
        positions,
        cq if cq is not None else iq,
        out,
        out.stride(0),
        t,
        nb,
        triton.cdiv(nb, nsplit),
        index_scale * (IDX_DIM**-0.5 if mode == MODE_LSE else 1.0),
        pool_temp,
        init,
        local,
        CAUSAL=causal,
        QT=qt,
        BB=bb,
        ROWS=rows,
        MODE=mode,
        Q4=q4,
        HW=hw,
        RAW=raw,
        NI=IDX_HEADS,
        DI=IDX_DIM,
        num_warps=num_warps,
    )
    return out


def _mode(ic):
    return MODE_POOL4MAX if _rows_of(ic) == PAGE else MODE_CENTROID


def score(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local, pool_temp, **kw):
    """Block scores ``[T, nb]`` fp32 from an NVFP4 cache, q4k4 (the checkpoint's
    ``sparse_index_q4k4``): the index q goes through NVFP4 in the kernel, by the same rule
    as the keys. pool4 / max from per-token pages, or max over the 4 centroids of centroid
    pages; heads max; pins and ``-inf`` applied. UT: ``PRO_NEW_INDEX_Q4=1``."""
    return _score(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local,
                  pool_temp, _mode(index_cache), True, **kw)  # fmt: skip


def score_bf16q(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local, pool_temp, **kw):
    """``score`` with the index q used as given (bf16, or already 4-bit from the prep).
    UT: ``PRO_NEW_INDEX_Q4`` unset."""
    return _score(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local,
                  pool_temp, _mode(index_cache), False, **kw)  # fmt: skip


def score_lse(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local, pool_temp, **kw):
    """The lse / lse reading from per-token pages, q4k4: log-sum-exp of a block's visible
    token dots / sqrt(128), then over the heads. UT: ``PRO_NEW_INDEX_Q4=1``."""
    return _score(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local,
                  pool_temp, MODE_LSE, True, **kw)  # fmt: skip


def score_lse_bf16q(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local, pool_temp, **kw):
    """``score_lse`` with the index q used as given. UT: ``PRO_NEW_INDEX_Q4`` unset."""
    return _score(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local,
                  pool_temp, MODE_LSE, False, **kw)  # fmt: skip


def _select(iq, ic, block_tables, seq_id, positions, nb, cq, init, local, pool_temp, k, q4,
            qt=None, bb=None, splits=None, waves=2, num_warps=4, hw=None,
            index_scale=1.0):  # fmt: skip
    rows, raw = _layout(ic)
    mode = MODE_POOL4MAX if rows == PAGE else MODE_CENTROID
    t = iq.shape[0]
    assert iq.shape[1:] == (IDX_HEADS, IDX_DIM) and iq.stride(-1) == 1
    assert k & (k - 1) == 0, "k must be a power of two"
    sel = torch.empty(t, k, dtype=torch.int32, device=iq.device)
    if t == 0:
        return sel
    if qt is None:
        qt = 1 if t <= 64 else 8
    if bb is None:
        bb = 4 if rows == PAGE else 16
    bb = min(bb, k)
    cus, gfx950 = _dev_info(iq.device)
    if hw is None:
        hw = gfx950
    nq = triton.cdiv(t, qt)
    if splits is None:  # enough programs for the GPU, at most 16 strided parts to merge
        splits = max(1, min(16, triton.cdiv(waves * cus, nq), triton.cdiv(nb, k)))
    final = splits == 1
    out = sel if final else torch.empty(t, splits, k, dtype=torch.int64, device=iq.device)
    _select_kernel[(nq, splits)](
        iq,
        iq.stride(0),
        iq.stride(1),
        ic,
        ic.stride(0),
        block_tables,
        block_tables.stride(0),
        seq_id,
        positions,
        cq if cq is not None else iq,
        out,
        out.stride(0),
        t,
        nb,
        index_scale,
        pool_temp,
        init,
        local,
        FINAL=final,
        K=k,
        QT=qt,
        BB=bb,
        ROWS=rows,
        MODE=mode,
        Q4=q4,
        HW=hw,
        RAW=raw,
        NI=IDX_HEADS,
        DI=IDX_DIM,
        num_warps=num_warps,
    )
    if not final:
        _merge_kernel[(t,)](out, sel, splits, K=k, NS2=triton.next_power_of_2(splits))
    return sel


def select(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local, pool_temp, k, **kw):
    """Scores fused with the top-k, q4k4: each token's ``k`` best causal blocks as logical
    block indices ``[T, k]`` int32 (-1 = unused; equal scores: the lower block), without
    writing the ``[T, nb]`` scores. Programs split the blocks into at most 16 ranges when the
    q tiles alone do not fill the GPU; ``_merge_kernel`` then reduces their
    ``[T, ranges, k]`` keys. UT: ``PRO_NEW_INDEX_Q4=1``."""
    return _select(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local,
                   pool_temp, k, True, **kw)  # fmt: skip


def select_bf16q(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local, pool_temp, k, **kw):
    """``select`` with the index q used as given. UT: ``PRO_NEW_INDEX_Q4`` unset."""
    return _select(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local,
                   pool_temp, k, False, **kw)  # fmt: skip


# ------------------------------------------------------------------------------ write --
@triton.jit
def _write_kernel(
    keys_ptr,
    pages_ptr,
    cq_ptr,
    ic_ptr,
    s_ic,
    pool_temp,
    ROWS: tl.constexpr,
    BF16_OUT: tl.constexpr,
    DI: tl.constexpr,
):
    """One full block: its 16 keys -> per-token NVFP4 rows, or its 4 softmax-pooled
    centroids -> NVFP4 rows or bf16."""
    p = tl.program_id(0)
    page = tl.load(pages_ptr + p).to(tl.int64)
    r = tl.arange(0, 16)
    d = tl.arange(0, DI)
    k = tl.load(keys_ptr + p.to(tl.int64) * (16 * DI) + r[:, None] * DI + d[None, :])
    k = k.to(tl.float32)
    if ROWS == 4:
        cq = tl.load(cq_ptr + d).to(tl.float32)
        lg = tl.reshape(tl.sum(k * cq[None, :], axis=1) * pool_temp, (4, 4))
        e = tl.exp(lg - tl.max(lg, axis=1)[:, None])
        w = e / tl.sum(e, axis=1)[:, None]
        x = tl.sum(tl.reshape(k, (4, 4, DI)) * w[:, :, None], axis=1)
    else:
        x = k
    base = ic_ptr + page * s_ic
    ro = tl.arange(0, ROWS)
    if BF16_OUT:
        tl.store(base + ro[:, None] * DI + d[None, :], x.to(tl.bfloat16))
    else:
        g = tl.reshape(x, (ROWS, DI // 16, 16))
        a = tl.abs(g)
        s8 = _nvfp4_scale(tl.max(a, axis=2))
        c = _e2m1_code(a, s8.to(tl.float32)[:, :, None])
        nib = tl.reshape(c | ((g < 0).to(tl.int32) << 3), (ROWS, DI // 2, 2))
        lo, hi = tl.split(nib)
        cb = tl.arange(0, DI // 2)
        tl.store(base + ro[:, None] * (DI // 2) + cb[None, :], (lo | (hi << 4)).to(tl.uint8))
        gb = tl.arange(0, DI // 16)
        tl.store(
            base + ROWS * (DI // 2) + ro[:, None] * (DI // 16) + gb[None, :],
            s8.to(tl.uint8, bitcast=True),
        )


def write(keys, pages, cq, pool_temp, index_cache):
    """Write full blocks: ``keys [P, 16, 128]`` bf16 into physical pages ``pages [P]`` of
    ``index_cache``: NVFP4 per-token pages ``[N, 1152]``, NVFP4 centroid pages
    ``[N, 288]``, bf16 centroids ``[N, 4, 128]``, or the keys themselves into a bf16 / fp8
    ``[N, 16, 128]`` cache (fp8: clamped to its range, no scale)."""
    if index_cache.dtype in RAW_DTYPES and index_cache.shape[1] == PAGE:
        v = keys.float()
        if index_cache.dtype != torch.bfloat16:
            fmax = torch.finfo(index_cache.dtype).max
            v = v.clamp(-fmax, fmax)
        index_cache[pages.long()] = v.to(index_cache.dtype)
        return
    if index_cache.dtype == torch.bfloat16:
        assert index_cache.shape[1:] == (CENTROIDS, IDX_DIM)
        rows, bf16_out = CENTROIDS, True
    else:
        rows, bf16_out = _rows_of(index_cache), False
    p = keys.shape[0]
    if p == 0:
        return
    keys = keys.contiguous()
    _write_kernel[(p,)](
        keys,
        pages,
        cq,
        index_cache,
        index_cache.stride(0),
        pool_temp,
        ROWS=rows,
        BF16_OUT=bf16_out,
        DI=IDX_DIM,
        num_warps=4,
    )
