# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon block scorer for MiniMax-M3.1-Pro's NVFP4 index cache on gfx950.

Packed pages contain E2M1 codes and one E4M3 scale per 16 elements. Per-token
pages hold 16 rows (1152 bytes); centroid pages hold four rows (288 bytes).
``score`` quantizes bf16 queries to the same group-16 NVFP4 values;
``score_bf16q`` uses bf16 queries directly.
``tq`` is the number of query-token rows grouped per program; each query row
contains 16 index heads. ``tq=0`` selects single-row scoring, like ``tq=1``.

Per-token scoring softmax-pools each group of four keys using ``cq`` and
``pool_temp``, then takes the maximum across groups and query heads. Centroid
scoring takes the maximum across the four stored rows and query heads. Output
is contiguous ``[T, nb]`` fp32: initial blocks are pinned to 1e30, local blocks
to 1e29, and future blocks to -inf. Causal mode leaves future columns unwritten.

Keys load directly into MFMA B-operand registers. Query and key dimensions are
permuted together: MFMA K index ``32*s + 8*c + e`` holds dimension
``32*c + 8*s + e``, giving each lane a contiguous 16-byte key load. E2M1 codes
are decoded at scale 1 by ``v_cvt_scalef32_pk_f16_fp4``; ``v_pk_mul_f16`` applies
the E4M3 scale exactly. FP16 MFMA accumulates in fp32.

Queries are loaded and quantized once per program. Single-tile decode programs
issue their first key loads during query quantization. The register pipeline
prefetches the next tile's keys/scales and the following tile's page-table
entries while scoring the current tile. Multi-query tiles stage queries once
in padded shared memory before retaining their MFMA operands in registers.
Native maxima use non-IEEE mode; denormal handling remains IEEE.
"""

from functools import lru_cache

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

PAGE = 16
CENTROIDS = 4
IDX_HEADS = 16
IDX_DIM = 128
NUM_WARPS = 4


def page_bytes(rows):
    """Bytes of a packed NVFP4 page, including group-16 E4M3 scales."""
    return rows * (IDX_DIM // 2 + IDX_DIM // 16)


@lru_cache(maxsize=16)
def _device_info(index):
    props = torch.cuda.get_device_properties(index)
    return props.multi_processor_count, getattr(props, "gcnArchName", "").split(":")[0]


def is_supported(device=None):
    """Whether ``device`` is a gfx950 GPU."""
    device = torch.device("cuda" if device is None else device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    index = torch.cuda.current_device() if device.index is None else device.index
    return _device_info(index)[1] == "gfx950"


def _rows_of(ic):
    if ic.dtype == torch.uint8 and ic.ndim == 2 and ic.stride(1) == 1:
        for rows in (PAGE, CENTROIDS):
            if ic.shape[1] == page_bytes(rows) and ic.stride(0) == page_bytes(rows):
                return rows
    raise ValueError(
        "index_cache must be contiguous NVFP4 uint8 pages [N,1152] or [N,288]"
    )


# ------------------------------------ helpers -------------------------------------------
@gluon.jit
def _cvt_byte(word, sc2, SEL: gl.constexpr):
    """Byte ``SEL`` of ``word`` (2 e2m1) -> 2 fp16 times the fp16 scale held twice in
    ``sc2``, as an int32."""
    if SEL == 0:
        return gl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_f16_fp4 $0, $1, 1.0\nv_pk_mul_f16 $0, $0, $2",
            "=&v,v,v", [word, sc2], dtype=gl.int32, is_pure=True, pack=1)  # fmt: skip
    elif SEL == 1:
        return gl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_f16_fp4 $0, $1, 1.0 op_sel:[1,0,0]\nv_pk_mul_f16 $0, $0, $2",
            "=&v,v,v", [word, sc2], dtype=gl.int32, is_pure=True, pack=1)  # fmt: skip
    elif SEL == 2:
        return gl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_f16_fp4 $0, $1, 1.0 op_sel:[0,1,0]\nv_pk_mul_f16 $0, $0, $2",
            "=&v,v,v", [word, sc2], dtype=gl.int32, is_pure=True, pack=1)  # fmt: skip
    else:
        return gl.inline_asm_elementwise(
            "v_cvt_scalef32_pk_f16_fp4 $0, $1, 1.0 op_sel:[1,1,0]\nv_pk_mul_f16 $0, $0, $2",
            "=&v,v,v", [word, sc2], dtype=gl.int32, is_pure=True, pack=1)  # fmt: skip


@gluon.jit
def _lo16(p):
    return p.to(gl.int16).to(gl.float16, bitcast=True)


@gluon.jit
def _hi16(p):
    return (p >> 16).to(gl.int16).to(gl.float16, bitcast=True)


@gluon.jit
def _nvfp4_scale(amax):
    s = amax / 6.0
    s = gl.where(s < 0.001953125, 0.001953125, s)
    s = gl.where(s > 448.0, 448.0, s)
    return s.to(gl.float8e4nv)


@gluon.jit
def _fake_nvfp4(x, ROWS: gl.constexpr, D: gl.constexpr):
    """Quantize bf16 queries promoted to fp32 and return restored fp16 values."""
    g = gl.reshape(x, [ROWS, D // 16, 16])
    a = gl.abs(g)
    s = gl.expand_dims(_nvfp4_scale(gl.max(a, axis=2)).to(gl.float32), 2)
    # BF16 midpoint products are exact and adjacent magnitudes differ by at
    # least 1/256. With a reciprocal within two ULPs, an eight-ULP bias toward
    # zero keeps midpoint ties on the smaller code without crossing non-ties.
    inv_scale = gl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [s], dtype=gl.float32, is_pure=True, pack=1
    )
    normalized = a * inv_scale
    normalized_bits = normalized.to(gl.uint32, bitcast=True)
    normalized_bits = gl.maximum(normalized_bits, 8) - 8
    normalized = normalized_bits.to(gl.float32, bitcast=True)
    even, odd = gl.split(gl.reshape(normalized, [ROWS, D // 16, 8, 2]))
    packed = gl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        "=v,v,v,v",
        [even, odd, 1.0],
        dtype=gl.int32,
        is_pure=True,
        pack=1,
    )
    neg_even, neg_odd = gl.split(
        gl.reshape((g < 0).to(gl.int32), [ROWS, D // 16, 8, 2])
    )
    packed = packed | (neg_even << 3) | (neg_odd << 7)
    sb = s.to(gl.float16).to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
    sc2 = gl.convert_layout(sb | (sb << 16), packed.type.layout)
    sc2 = sc2 + gl.full(packed.shape, 0, gl.int32, layout=packed.type.layout)
    restored = _cvt_byte(packed, sc2, 0)
    return gl.reshape(gl.join(_lo16(restored), _hi16(restored)), [ROWS, D])


@gluon.jit
def _k_permute(x, ROWS: gl.constexpr):
    """``[ROWS, 128]`` in dim order -> MFMA K order (K index ``32 s + 8 c + e`` holds dim
    ``32 c + 8 s + e``)."""
    x = gl.reshape(x, [ROWS, 4, 4, 8])  # [rows, c, s, e]
    x = gl.permute(x, [0, 2, 1, 3])  # [rows, s, c, e]
    return gl.reshape(x, [ROWS, 128])


@gluon.jit
def _phys_rows(bt_ptr, s_bt, seq, j0, qbmax, NB, ROWS: gl.constexpr, R: gl.constexpr,
               RL: gl.constexpr):  # fmt: skip
    """Physical page of each of the tile's ``R`` cache rows (0 for rows not read) and the
    live mask, in row layout ``RL``."""
    rr = gl.arange(0, R, layout=RL)
    jr = j0 + rr // ROWS
    live = (jr <= qbmax) & (jr < NB)
    phys = gl.amd.cdna4.buffer_load(bt_ptr, seq * s_bt + jr, mask=live, other=0)
    return phys, live


@gluon.jit
def _load_tile(ic_ptr, icw_ptr, phys, live, ROWS: gl.constexpr, R: gl.constexpr,
               WL3: gl.constexpr, BIG_IC: gl.constexpr):  # fmt: skip
    """The tile's words ``[R, 4 (c), 4 (s)]`` int32 (word ``4 c + s`` of each row) and the
    matching e4m3 scale bytes, both in ``WL3``."""
    rr = gl.arange(0, R, layout=gl.SliceLayout(1, gl.SliceLayout(2, WL3)))
    cc = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, WL3)))
    ss = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, WL3)))
    r3 = gl.expand_dims(gl.expand_dims(rr % ROWS, 1), 2)
    c3 = gl.expand_dims(gl.expand_dims(cc, 0), 2)
    s3 = gl.expand_dims(gl.expand_dims(ss, 0), 1)
    p3 = gl.expand_dims(
        gl.expand_dims(
            gl.convert_layout(phys, gl.SliceLayout(1, gl.SliceLayout(2, WL3))), 1
        ),
        2,
    )
    l3 = gl.expand_dims(
        gl.expand_dims(
            gl.convert_layout(live, gl.SliceLayout(1, gl.SliceLayout(2, WL3))), 1
        ),
        2,
    )
    m3 = l3 & (c3 >= 0) & (s3 >= 0)
    S_IC: gl.constexpr = ROWS * 72  # bytes a page
    if BIG_IC:
        p64 = p3.to(gl.int64)
        woff = p64 * (S_IC // 4) + r3 * 16 + c3 * 4 + s3
        words = gl.load(icw_ptr + woff, mask=m3, other=0)
        soff = p64 * S_IC + ROWS * 64 + r3 * 8 + c3 * 2 + s3 // 2
        scales = gl.load(ic_ptr + soff, mask=m3, other=0)
    else:
        woff = p3 * (S_IC // 4) + r3 * 16 + c3 * 4 + s3
        words = gl.amd.cdna4.buffer_load(icw_ptr, woff, mask=m3, other=0)
        soff = p3 * S_IC + ROWS * 64 + r3 * 8 + c3 * 2 + s3 // 2
        scales = gl.amd.cdna4.buffer_load(ic_ptr, soff, mask=m3, other=0)
    return words, scales


@gluon.jit
def _keys_b(words, scales, R: gl.constexpr, BL: gl.constexpr):
    """Words + scales ``[R, 4, 4]`` -> the keys as the MFMA B operand ``[128, R]`` fp16 (K in
    MFMA order), without moving data between lanes."""
    s16 = scales.to(gl.float8e4nv, bitcast=True).to(gl.float16)
    sb = s16.to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
    sc2 = sb | (sb << 16)
    w2 = gl.reshape(gl.permute(words, [0, 2, 1]), [R, 16])  # word slot 4 s + c
    s2 = gl.reshape(gl.permute(sc2, [0, 2, 1]), [R, 16])
    p0 = _cvt_byte(w2, s2, 0)
    p1 = _cvt_byte(w2, s2, 1)
    p2 = _cvt_byte(w2, s2, 2)
    p3 = _cvt_byte(w2, s2, 3)
    lo = gl.join(gl.join(_lo16(p0), _lo16(p2)), gl.join(_lo16(p1), _lo16(p3)))
    hi = gl.join(gl.join(_hi16(p0), _hi16(p2)), gl.join(_hi16(p1), _hi16(p3)))
    k = gl.reshape(gl.join(lo, hi), [R, 128])  # [row, 8 (4 s + c) + 2 b + h]
    return gl.convert_layout(gl.permute(k, [1, 0]), BL, assert_trivial=True)


# ----------------------------------------------------------------------------- kernel --
@gluon.jit
def _score_kernel(
    iq_ptr,
    ic_ptr,
    bt_ptr,
    s_bt: gl.constexpr,
    seq_ptr,
    pos_ptr,
    cq_ptr,
    out_ptr,
    T: gl.constexpr,
    NB: gl.constexpr,
    FILL: gl.constexpr,
    sc,
    pool_temp,
    INIT: gl.constexpr,
    LOCAL: gl.constexpr,
    CAUSAL: gl.constexpr,
    TQ: gl.constexpr,
    BB: gl.constexpr,
    ROWS: gl.constexpr,
    POOL: gl.constexpr,
    Q4: gl.constexpr,
    WL3: gl.constexpr,
    WM: gl.constexpr,
    PIPE: gl.constexpr,
    SINGLE_SEQ: gl.constexpr,
    NSPLIT: gl.constexpr,
    BIG_IC: gl.constexpr,
    FLAT_MAX: gl.constexpr,
    FIRST_TILE: gl.constexpr,
):
    NI: gl.constexpr = 16
    DI: gl.constexpr = 128
    M: gl.constexpr = TQ * NI
    R: gl.constexpr = BB * ROWS
    FW: gl.constexpr = 256
    MFMA: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[WM, 4 // WM],
    )
    AL: gl.constexpr = gl.DotOperandLayout(operand_index=0, parent=MFMA, k_width=8)
    BL: gl.constexpr = gl.DotOperandLayout(operand_index=1, parent=MFMA, k_width=8)
    QL: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    SL: gl.constexpr = gl.BlockedLayout(
        [1, 1], [1, 64], [1, 4], [1, 0]
    )  # [TQ, BB] scores
    FL: gl.constexpr = gl.BlockedLayout(
        [1, 4], [1, 64], [1, 4], [1, 0]
    )  # [TQ, FW] -inf fill
    RL: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, WL3))

    t0 = gl.program_id(0) * TQ
    pid_s = gl.program_id(1)
    ns = NSPLIT
    icw_ptr = ic_ptr.to(gl.pointer_type(gl.int32), bitcast=True)

    if FIRST_TILE:
        q_rows = t0 + gl.arange(0, TQ, layout=gl.SliceLayout(1, SL))
        tval = q_rows < T
        pos_t = gl.load(pos_ptr + q_rows, mask=tval, other=0)
        if SINGLE_SEQ:
            # A one-row table can only be indexed by sequence zero. Specializing this
            # metadata removes the sequence scan and its live scalar state for bs1.
            seq_t = gl.full([TQ], 0, gl.int32, layout=gl.SliceLayout(1, SL))
        else:
            seq_t = gl.load(seq_ptr + q_rows, mask=tval, other=-1)
        qb_t = pos_t // 16

    # q tile -> fp32 (q4k4: NVFP4 by 16 consecutive dims) -> MFMA K order -> fp16 A operand.
    # Loaded once in a blocked layout and redistributed: the A operand is replicated over the
    # 4 warps (they split N), so reading it in place would load q 4 times.
    rm = gl.arange(0, M, layout=gl.SliceLayout(1, QL))
    dd = gl.arange(0, DI, layout=gl.SliceLayout(0, QL))
    tm = t0 + rm // NI
    qmask = gl.expand_dims(tm < T, 1) & gl.expand_dims(dd >= 0, 0)
    qoff = gl.expand_dims(
        tm.to(gl.int64) * (NI * DI) + (rm % NI) * DI, 1
    ) + gl.expand_dims(dd, 0)
    if FIRST_TILE:
        # Issue Q first so its wait leaves the later key loads outstanding.
        q_raw = gl.load(iq_ptr + qoff, mask=qmask, other=0.0)
        qb_first = gl.max(gl.where(tval, qb_t, -1), axis=0)
        ph_first, lv_first = _phys_rows(
            bt_ptr, s_bt, 0, pid_s * BB, qb_first, NB, ROWS, R, RL
        )
        w_first, s_first = _load_tile(
            ic_ptr, icw_ptr, ph_first, lv_first, ROWS, R, WL3, BIG_IC
        )
        gl.amd.hint.sched_barrier()
        q = q_raw.to(gl.float32)
    else:
        q = gl.load(iq_ptr + qoff, mask=qmask, other=0.0).to(gl.float32)
    if Q4:
        q = _fake_nvfp4(q, M, DI)
    if TQ > 1:
        SHP_AUTO: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
            AL, [M, DI], gl.float16
        )
        if SHP_AUTO is None:
            SHP: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
                [[128, 8]], [M, DI], [1, 0]
            )
        else:
            SHP: gl.constexpr = SHP_AUTO
        qp = gl.allocate_shared_memory(gl.float16, [M, DI], SHP)
        qp.store(_k_permute(q, M).to(gl.float16))
        qa = qp.load(AL)
    else:
        qa = gl.convert_layout(_k_permute(q, M).to(gl.float16), AL)
    # MFMA K index k = 32 s + 8 c + e holds dim 32 c + 8 s + e
    ka = gl.arange(0, DI, layout=gl.SliceLayout(0, AL))
    dperm = gl.max_contiguous(
        gl.multiple_of(32 * ((ka // 8) % 4) + 8 * (ka // 32) + ka % 8, 8), 8
    )
    if POOL:
        # index_compress_query in all 16 rows: every row of the logits dot is the same, so
        # each lane holds its key rows' logits without a cross-lane extraction
        r16 = gl.arange(0, 16, layout=gl.SliceLayout(1, AL))
        ca = gl.load(cq_ptr + gl.expand_dims(r16 * 0, 1) + gl.expand_dims(dperm, 0)).to(
            gl.float16
        )

    if not FIRST_TILE:
        q_rows = t0 + gl.arange(0, TQ, layout=gl.SliceLayout(1, SL))
        tval = q_rows < T
        pos_t = gl.load(pos_ptr + q_rows, mask=tval, other=0)
        if SINGLE_SEQ:
            # A one-row table can only be indexed by sequence zero. Specializing this
            # metadata removes the sequence scan and its live scalar state for bs1.
            seq_t = gl.full([TQ], 0, gl.int32, layout=gl.SliceLayout(1, SL))
        else:
            seq_t = gl.load(seq_ptr + q_rows, mask=tval, other=-1)
        qb_t = pos_t // 16
    jt = gl.arange(0, BB, layout=gl.SliceLayout(0, SL))
    orow = out_ptr + gl.expand_dims(q_rows.to(gl.int64) * NB, 1)
    zero_acc = gl.zeros([M, R], gl.float32, layout=MFMA)

    # Visit the tile's distinct sequence ids once, including non-contiguous ids. With
    # SINGLE_SEQ this becomes one pass; otherwise the scan depends on device data and
    # remains valid when seq_id changes during CUDA-graph replay.
    big = 2147483647
    s_lo = gl.min(gl.where(tval, seq_t, big), axis=0)
    s_hi = gl.max(gl.where(tval, seq_t, -1), axis=0)
    s_sub = s_lo
    while s_sub <= s_hi:
        mine = tval & (seq_t == s_sub)
        qbmax = gl.max(gl.where(mine, qb_t, -1), axis=0)
        ntile = qbmax // BB + 1
        # PIPE: tile i scored while tile i+1's keys and tile i+2's pages load
        if PIPE:
            if FIRST_TILE:
                w_c, s_c = w_first, s_first
            else:
                ph, lv = _phys_rows(
                    bt_ptr, s_bt, s_sub, pid_s * BB, qbmax, NB, ROWS, R, RL
                )
                w_c, s_c = _load_tile(ic_ptr, icw_ptr, ph, lv, ROWS, R, WL3, BIG_IC)
            ph_n, lv_n = _phys_rows(
                bt_ptr, s_bt, s_sub, (pid_s + ns) * BB, qbmax, NB, ROWS, R, RL
            )
        for ti in range(pid_s, ntile, ns):
            if PIPE:
                w_n, s_n = _load_tile(ic_ptr, icw_ptr, ph_n, lv_n, ROWS, R, WL3, BIG_IC)
                ph_n, lv_n = _phys_rows(
                    bt_ptr, s_bt, s_sub, (ti + 2 * ns) * BB, qbmax, NB, ROWS, R, RL
                )
            else:
                if FIRST_TILE:
                    w_c, s_c = w_first, s_first
                else:
                    ph, lv = _phys_rows(
                        bt_ptr, s_bt, s_sub, ti * BB, qbmax, NB, ROWS, R, RL
                    )
                    w_c, s_c = _load_tile(ic_ptr, icw_ptr, ph, lv, ROWS, R, WL3, BIG_IC)
            kb = _keys_b(w_c, s_c, R, BL)
            dots = gl.amd.cdna4.mfma(qa, kb, zero_acc)  # [M, R]
            if POOL:
                lg = gl.amd.cdna4.mfma(
                    ca, kb, gl.zeros([16, R], gl.float32, layout=MFMA)
                )
                lg = gl.reshape(lg * (pool_temp * 1.4426950408889634), [16, R // 4, 4])
                e = gl.exp2(lg - gl.expand_dims(gl.max(lg, axis=2), 2))
                den = gl.sum(e, axis=2)
                rs = gl.fdiv(gl.full(den.shape, 1.0, gl.float32, layout=den.type.layout), den,
                             ieee_rounding=False)  # fmt: skip
                w = gl.reshape(e * gl.expand_dims(rs, 2), [16, R])  # same in every row
                if TQ == 1:
                    dw = dots * gl.convert_layout(w, MFMA)
                else:
                    d3 = gl.reshape(dots, [TQ, NI, R])
                    w3 = gl.convert_layout(w, gl.SliceLayout(0, d3.type.layout))
                    dw = gl.reshape(d3 * gl.expand_dims(w3, 0), [M, R])
                cent = gl.sum(gl.reshape(dw, [M, R // 4, 4]), axis=2)
                if FLAT_MAX:
                    all_cent = gl.permute(
                        gl.reshape(cent, [TQ, NI, BB, 4]), [0, 2, 3, 1]
                    )
                    block_score = gl.max(gl.reshape(all_cent, [TQ, BB, 4 * NI]), axis=2)
                else:
                    bs = gl.max(gl.reshape(cent, [M, BB, 4]), axis=2)
            else:
                if FLAT_MAX:
                    all_rows = gl.permute(
                        gl.reshape(dots, [TQ, NI, BB, ROWS]), [0, 2, 3, 1]
                    )
                    block_score = gl.max(
                        gl.reshape(all_rows, [TQ, BB, ROWS * NI]), axis=2
                    )
                else:
                    bs = gl.max(gl.reshape(dots, [M, BB, ROWS]), axis=2)
            if FLAT_MAX:
                score = gl.convert_layout(block_score, SL) * sc
            else:
                score = (
                    gl.convert_layout(gl.max(gl.reshape(bs, [TQ, NI, BB]), axis=1), SL)
                    * sc
                )
            jb = ti * BB + jt
            jb2 = gl.expand_dims(jb, 0)
            qb2 = gl.expand_dims(qb_t, 1)
            score = gl.where(jb2 < INIT, 1e30, score)
            score = gl.where(jb2 >= qb2 + 1 - LOCAL, 1e29, score)
            score = gl.where(jb2 <= qb2, score, float("-inf"))
            gl.store(orow + jb2, score, mask=gl.expand_dims(mine, 1) & (jb2 < NB))
            if PIPE:
                w_c = w_n
                s_c = s_n
        if not CAUSAL:
            f1 = gl.minimum(NB, (pid_s + 1) * FILL)
            neg = gl.full([TQ, FW], float("-inf"), gl.float32, layout=FL)
            q_fill_rows = t0 + gl.arange(0, TQ, layout=gl.SliceLayout(1, FL))
            minef = gl.convert_layout(mine, gl.SliceLayout(1, FL))
            orowf = out_ptr + gl.expand_dims(q_fill_rows.to(gl.int64) * NB, 1)
            for c0 in range(gl.maximum(ntile * BB, pid_s * FILL), f1, FW):
                cols = gl.expand_dims(
                    c0 + gl.arange(0, FW, layout=gl.SliceLayout(0, FL)), 0
                )
                gl.store(orowf + cols, neg, mask=gl.expand_dims(minef, 1) & (cols < f1))
        if SINGLE_SEQ:
            s_sub = big
        else:
            s_sub = gl.min(gl.where(tval & (seq_t > s_sub), seq_t, big), axis=0)


# ------------------------------------------------------------------------------- host --
@lru_cache(maxsize=16)
def _words_layout(r, wm=1):
    """``[R, 4 (c), 4 (s)]`` words as the MFMA B operand holds them (``wm`` x ``4 // wm``
    warps, M x N): rows over lanes 0-3, then the N warps, then registers; c over lanes
    4-5; s over registers (16 contiguous bytes a lane). Warps along M hold copies.

    Cache this shape metadata: constructing a DistributedLinearLayout unwraps all its
    nested bases and otherwise adds about 13 us to every eager scorer call.
    """
    assert r in (64, 128) and wm in (1, 2, 4), (r, wm)
    wn = 4 // wm
    row_bits = [
        16 << i for i in range((r // 16).bit_length() - 1)
    ]  # rows past lanes 0-3
    n_warp = row_bits[: wn.bit_length() - 1]
    warp = [[b, 0, 0] for b in n_warp] + [[0, 0, 0]] * (wm.bit_length() - 1)
    reg = [[0, 0, 1], [0, 0, 2]] + [[b, 0, 0] for b in row_bits[len(n_warp) :]]
    return gl.DistributedLinearLayout(
        reg_bases=reg,
        lane_bases=[[1, 0, 0], [2, 0, 0], [4, 0, 0], [8, 0, 0], [0, 1, 0], [0, 2, 0]],
        warp_bases=warp,
        block_bases=[],
        shape=[r, 4, 4],
    )


def _score(iq, ic, block_tables, seq_id, positions, nb, cq, init, local, pool_temp, q4,
           tq=None, bb=None, waves=None, causal=False, index_scale=1.0, wm=1,
           pipe=None):  # fmt: skip
    if not is_supported(iq.device):
        raise RuntimeError("NVFP4 index scorer requires gfx950")
    index = torch.cuda.current_device() if iq.device.index is None else iq.device.index
    cus = _device_info(index)[0]
    rows = _rows_of(ic)
    t = iq.shape[0]
    if (
        iq.dtype != torch.bfloat16
        or iq.shape[1:] != (IDX_HEADS, IDX_DIM)
        or not iq.is_contiguous()
    ):
        raise ValueError("iq must be contiguous bf16 [T,16,128]")
    if nb < 0 or block_tables.ndim != 2 or block_tables.shape[1] < nb:
        raise ValueError("block_tables must have shape [B, >= nb], with nb >= 0")
    if block_tables.dtype != torch.int32 or block_tables.stride(1) != 1:
        raise ValueError("block_tables must be int32 with contiguous columns")
    for name, value in (("seq_id", seq_id), ("positions", positions)):
        if (
            value.dtype != torch.int32
            or value.shape != (t,)
            or not value.is_contiguous()
        ):
            raise ValueError(f"{name} must be contiguous int32 [T]")
    if rows == PAGE and (
        cq is None or cq.shape != (IDX_DIM,) or not cq.is_contiguous()
    ):
        raise ValueError("per-token pages require contiguous cq [128]")
    for value in (ic, block_tables, seq_id, positions, cq):
        if value is not None and value.device != iq.device:
            raise ValueError("all scorer tensors must be on the same device")
    out = torch.empty(t, nb, dtype=torch.float32, device=iq.device)
    if t == 0 or nb == 0:
        return out
    if tq is None:
        tq = 1 if t <= 64 else 8
    elif tq == 0:
        tq = 1
    if tq not in (1, 2, 4, 8):
        raise ValueError("tq must be one of 0, 1, 2, 4, 8")
    if wm not in (1, 2, 4):
        raise ValueError("wm must be one of 1, 2, 4")
    if bb is None:
        bb = (8 if tq <= 4 else 4) if rows == PAGE else 16
        if rows == PAGE and t == 1 and tq == 1 and nb <= 1024:
            # Short bs1 decode benefits from more, smaller programs.
            bb = 4
    if bb * rows not in (64, 128):
        raise ValueError("bb must form a tile of 64 or 128 cache rows")
    if waves is None:
        waves = 2 if tq > 1 else 4
    nq = triton.cdiv(t, tq)
    if not isinstance(waves, int) or waves <= 0:
        raise ValueError("waves must be a positive integer")
    nsplit = max(1, min(triton.cdiv(nb, bb), triton.cdiv(waves * cus, nq)))
    if pipe is None:
        # Each program can consume at most one tile in this launch, so preloading
        # subsequent tiles only adds masked loads and loop state.
        pipe = nsplit * bb < nb
    _score_kernel[(nq, nsplit)](
        iq,
        ic,
        block_tables,
        block_tables.stride(0),
        seq_id,
        positions,
        cq if cq is not None else iq,
        out,
        t,
        nb,
        triton.cdiv(nb, nsplit),
        index_scale,
        pool_temp,
        INIT=init,
        LOCAL=local,
        CAUSAL=causal,
        TQ=tq,
        BB=bb,
        ROWS=rows,
        POOL=rows == PAGE,
        Q4=q4,
        WL3=_words_layout(bb * rows, wm),
        WM=wm,
        PIPE=pipe,
        SINGLE_SEQ=block_tables.shape[0] == 1,
        NSPLIT=nsplit,
        BIG_IC=ic.numel() > 2**32,
        FIRST_TILE=(
            rows == PAGE
            and block_tables.shape[0] == 1
            and wm == 1
            and (
                (tq == 1 and not pipe and nsplit * bb >= nb)
                or (
                    q4
                    and t == tq
                    and tq in (4, 8)
                    and nb <= (8192 if tq == 4 else 4096)
                    and bb == (8 if tq == 4 else 4)
                )
            )
        ),
        FLAT_MAX=(
            rows == PAGE
            and block_tables.shape[0] == 1
            and t <= 8
            and wm == 1
            and ((tq == 4 and bb == 8) or (tq == 8 and bb == 4 and nb <= 4096))
        ),
        num_warps=NUM_WARPS,
        llvm_fn_attrs="amdgpu-ieee=false",
    )
    return out


def score(
    iq,
    index_cache,
    block_tables,
    seq_id,
    positions,
    nb,
    cq,
    init,
    local,
    pool_temp,
    **kw,
):
    """Score NVFP4 keys after group-16 query quantization, grouping ``tq`` rows."""
    return _score(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local,
                  pool_temp, True, **kw)  # fmt: skip


def score_bf16q(
    iq,
    index_cache,
    block_tables,
    seq_id,
    positions,
    nb,
    cq,
    init,
    local,
    pool_temp,
    **kw,
):
    """Score NVFP4 keys using bf16 queries directly, grouping ``tq`` rows."""
    return _score(iq, index_cache, block_tables, seq_id, positions, nb, cq, init, local,
                  pool_temp, False, **kw)  # fmt: skip
