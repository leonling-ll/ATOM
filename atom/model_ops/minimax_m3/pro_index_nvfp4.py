# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Gluon block scorer for MiniMax-M3.1-Pro's NVFP4 index cache on gfx950.

Pages hold E2M1 codes with one E4M3 scale per 16 elements: per-token pages have 16
rows (1152 bytes), centroid pages four rows (288 bytes). ``score`` quantizes the bf16
queries to the same group-16 NVFP4 values; ``score_bf16q`` uses them directly. ``tq``
query tokens (16 index heads each) share a program; ``tq=0`` means one.

Per-token blocks are softmax-pooled over each group of four keys with ``cq`` and
``pool_temp``; centroid blocks use their four stored rows. A block's score is the
maximum over groups (rows) and heads. The output is contiguous ``[T, nb]`` fp32 with
initial blocks pinned to 1e30, local blocks to 1e29 and future blocks to -inf (left
unwritten in causal mode).

Keys are decoded in registers into the fp16 MFMA B operand (``v_cvt_scalef32_pk_f16_fp4``
then an exact ``v_pk_mul_f16`` by the scale). Query rows are head-major, so a block
maximum takes three cross-lane steps. Programs with several tiles keep two 64-row key
tiles in flight in registers; loads are issued in an explicit order because ``vmcnt``
retires them in order. With several query rows, a tile's MFMAs run at wave priority 0
and its vector work at 1, so the two programs sharing a SIMD alternate between the
matrix and vector pipes. Packed f32 ops are disabled: they cannot co-issue with the other
wave's MFMAs. Buffer operations address up to ``_BUFFER_BYTES``; larger tensors use
64-bit pointers, and single-row tiles read dead cache rows from one aligned query line.
"""

import os
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
_BUFFER_BYTES = 2**31


def page_bytes(rows):
    """Bytes of a packed NVFP4 page, including group-16 E4M3 scales."""
    return rows * (IDX_DIM // 2 + IDX_DIM // 16)


@lru_cache(maxsize=16)
def _device_info(index):
    """CU count and architecture name of device ``index``."""
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
    """Rows per page of ``ic`` (16 per-token, 4 centroid); raises on other layouts."""
    if ic.dtype == torch.uint8 and ic.ndim == 2 and ic.stride(1) == 1:
        for rows in (PAGE, CENTROIDS):
            if ic.shape[1] == page_bytes(rows) and ic.stride(0) == page_bytes(rows):
                return rows
    raise ValueError(
        "index_cache must be contiguous NVFP4 uint8 pages [N,1152] or [N,288]"
    )


@gluon.jit
def _cvt_byte(word, sc2, SEL: gl.constexpr):
    """Byte ``SEL`` of ``word`` (two E2M1 codes) as two fp16 times the scale in ``sc2``."""
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
def _setprio(dummy, P: gl.constexpr):
    """``s_setprio P`` (0 or 1) for the issuing wave; ``dummy`` is any scalar."""
    if P == 0:
        return gl.inline_asm_elementwise("s_setprio 0", "=s,s", [dummy], dtype=gl.int32,
                                         is_pure=False, pack=1)  # fmt: skip
    else:
        return gl.inline_asm_elementwise("s_setprio 1", "=s,s", [dummy], dtype=gl.int32,
                                         is_pure=False, pack=1)  # fmt: skip


@gluon.jit
def _lo16(p):
    """Low fp16 half of ``p``."""
    return p.to(gl.int16).to(gl.float16, bitcast=True)


@gluon.jit
def _hi16(p):
    """High fp16 half of ``p``."""
    return (p >> 16).to(gl.int16).to(gl.float16, bitcast=True)


@gluon.jit
def _nvfp4_scale(amax):
    """E4M3 group scale for a group maximum ``amax``."""
    s = amax / 6.0
    s = gl.where(s < 0.001953125, 0.001953125, s)
    s = gl.where(s > 448.0, 448.0, s)
    return s.to(gl.float8e4nv)


@gluon.jit
def _fake_nvfp4(x, ROWS: gl.constexpr, D: gl.constexpr):
    """Group-16 NVFP4 round trip of fp32 queries, returned as fp16. An eight-ULP bias
    toward zero keeps midpoint ties on the smaller code."""
    g = gl.reshape(x, [ROWS, D // 16, 16])
    a = gl.abs(g)
    s = gl.expand_dims(_nvfp4_scale(gl.max(a, axis=2)).to(gl.float32), 2)
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
    """Permute ``[ROWS, 128]`` into MFMA K order (K ``32s + 8c + e`` holds dim ``32c + 8s + e``)."""
    x = gl.reshape(x, [ROWS, 4, 4, 8])
    x = gl.permute(x, [0, 2, 1, 3])
    return gl.reshape(x, [ROWS, 128])


@gluon.jit
def _phys_rows(bt_row, j0, qbmax, NB, ROWS: gl.constexpr, R: gl.constexpr,
               RL: gl.constexpr):  # fmt: skip
    """Physical page and live mask of each of a tile's ``R`` cache rows."""
    rr = gl.arange(0, R, layout=RL)
    jr = j0 + rr // ROWS
    live = (jr <= qbmax) & (jr < NB)
    phys = gl.amd.cdna4.buffer_load(bt_row, jr, mask=live)
    return phys, live


@gluon.jit
def _phys_any(bt_row, j0, NB, ROWS: gl.constexpr, R: gl.constexpr,
              RL: gl.constexpr):  # fmt: skip
    """Physical pages of a tile's rows bounded by the table width only (no positions)."""
    rr = gl.arange(0, R, layout=RL)
    jr = j0 + rr // ROWS
    return gl.amd.cdna4.buffer_load(bt_row, jr, mask=jr < NB)


@gluon.jit
def _live_rows(j0, qbmax, NB, ROWS: gl.constexpr, R: gl.constexpr, RL: gl.constexpr):
    """Live mask of a tile's ``R`` cache rows."""
    rr = gl.arange(0, R, layout=RL)
    jr = j0 + rr // ROWS
    return (jr <= qbmax) & (jr < NB)


@gluon.jit
def _load_meta(pos_ptr, seq_ptr, q_rows, tval, SINGLE_SEQ: gl.constexpr):
    """Positions and sequence ids of the program's query rows (-1 past ``T``)."""
    pos_t = gl.amd.cdna4.buffer_load(pos_ptr, q_rows, mask=tval)
    if SINGLE_SEQ:
        seq_t = gl.full(q_rows.shape, 0, gl.int32, layout=q_rows.type.layout)
    else:
        seq_t = gl.where(tval, gl.amd.cdna4.buffer_load(seq_ptr, q_rows, mask=tval), -1)
    return pos_t, seq_t


@gluon.jit
def _load_q(iq_ptr, tm, rm, dd, T, TQ: gl.constexpr, BIG_Q: gl.constexpr):
    """The program's head-major query rows ``[M, 128]`` bf16 (zeros past ``T``)."""
    qmask = gl.expand_dims(tm < T, 1) & gl.expand_dims(dd >= 0, 0)
    if BIG_Q:
        qoff = gl.expand_dims(tm.to(gl.int64) * 2048 + (rm // TQ) * 128, 1)
        return gl.load(iq_ptr + qoff + gl.expand_dims(dd, 0), mask=qmask, other=0.0)
    qoff = gl.expand_dims(tm * 2048 + (rm // TQ) * 128, 1) + gl.expand_dims(dd, 0)
    return gl.amd.cdna4.buffer_load(iq_ptr, qoff, mask=qmask)


@gluon.jit
def _load_tile(ic_ptr, icw_ptr, phys, live, ROWS: gl.constexpr, R: gl.constexpr,
               WL3: gl.constexpr, BIG_IC: gl.constexpr, QLINE: gl.constexpr, q_line):  # fmt: skip
    """Load a tile's E2M1 words ``[R, 4, 4]`` and their E4M3 scale bytes. With ``QLINE``, rows
    past the sequence read the 16-byte line ``q_line`` of this launch's queries instead of
    masked branches."""
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
    S_IC: gl.constexpr = ROWS * 72
    if BIG_IC and QLINE:
        p64 = p3.to(gl.int64)
        qw = q_line.to(gl.pointer_type(gl.int32))
        qb = q_line.to(gl.pointer_type(gl.uint8))
        wbase = gl.where(l3, icw_ptr + (p64 * (S_IC // 4) + r3 * 16 + c3 * 4), qw)
        words = gl.load(wbase + s3)
        sbase = gl.where(l3, ic_ptr + (p64 * S_IC + ROWS * 64 + r3 * 8 + c3 * 2), qb)
        scales = gl.load(sbase + s3 // 2)
    elif BIG_IC:
        p64 = p3.to(gl.int64)
        woff = p64 * (S_IC // 4) + r3 * 16 + c3 * 4 + s3
        words = gl.load(icw_ptr + woff, mask=m3, other=0)
        soff = p64 * S_IC + ROWS * 64 + r3 * 8 + c3 * 2 + s3 // 2
        scales = gl.load(ic_ptr + soff, mask=m3, other=0)
    else:
        woff = p3 * (S_IC // 4) + r3 * 16 + c3 * 4 + s3
        words = gl.amd.cdna4.buffer_load(icw_ptr, woff, mask=m3)
        soff = p3 * S_IC + ROWS * 64 + r3 * 8 + c3 * 2 + s3 // 2
        scales = gl.amd.cdna4.buffer_load(ic_ptr, soff, mask=m3)
    return words, scales


@gluon.jit
def _keys_b(words, scales, R: gl.constexpr, BL: gl.constexpr):
    """Decode a tile into the fp16 MFMA B operand ``[128, R]`` without lane exchanges."""
    s16 = scales.to(gl.float8e4nv, bitcast=True).to(gl.float16)
    sb = s16.to(gl.int16, bitcast=True).to(gl.int32) & 0xFFFF
    sc2 = sb | (sb << 16)
    w2 = gl.reshape(gl.permute(words, [0, 2, 1]), [R, 16])
    s2 = gl.reshape(gl.permute(sc2, [0, 2, 1]), [R, 16])
    p0 = _cvt_byte(w2, s2, 0)
    p1 = _cvt_byte(w2, s2, 1)
    p2 = _cvt_byte(w2, s2, 2)
    p3 = _cvt_byte(w2, s2, 3)
    lo = gl.join(gl.join(_lo16(p0), _lo16(p2)), gl.join(_lo16(p1), _lo16(p3)))
    hi = gl.join(gl.join(_hi16(p0), _hi16(p2)), gl.join(_hi16(p1), _hi16(p3)))
    k = gl.reshape(gl.join(lo, hi), [R, 128])
    return gl.convert_layout(gl.permute(k, [1, 0]), BL, assert_trivial=True)


@gluon.jit
def _split_pool4(x):
    """Split ``[M, R]`` into the four keys of each pooling group."""
    even, odd = gl.split(gl.reshape(x, [x.shape[0], x.shape[1] // 4, 2, 2]))
    x0, x2 = gl.split(even)
    x1, x3 = gl.split(odd)
    return x0, x1, x2, x3


@gluon.jit
def _score_tile(qa, ca, kb, ti, jt, qb_t, out_ptr, orow, mine, sc, pool_temp, NB: gl.constexpr,
                INIT: gl.constexpr, LOCAL: gl.constexpr, TQ: gl.constexpr,
                BB: gl.constexpr, ROWS: gl.constexpr, POOL: gl.constexpr,
                MFMA: gl.constexpr, SL: gl.constexpr,
                BIG_OUT: gl.constexpr):  # fmt: skip
    """Score tile ``ti`` from its decoded keys and store its pinned ``[TQ, BB]`` scores."""
    NI: gl.constexpr = 16
    M: gl.constexpr = TQ * NI
    R: gl.constexpr = BB * ROWS
    if TQ > 1:
        _setprio(ti, 0)
        gl.amd.hint.sched_barrier()
    dots = gl.amd.cdna4.mfma(qa, kb, gl.zeros([M, R], gl.float32, layout=MFMA))
    if POOL:
        lg = gl.amd.cdna4.mfma(ca, kb, gl.zeros([16, R], gl.float32, layout=MFMA))
    if TQ > 1:
        gl.amd.hint.sched_barrier()
        _setprio(ti, 1)
    if POOL:
        lg = gl.reshape(lg * (pool_temp * 1.4426950408889634), [16, R // 4, 4])
        e = gl.exp2(lg - gl.expand_dims(gl.max(lg, axis=2), 2))
        den = gl.sum(e, axis=2)
        inv_den = gl.inline_asm_elementwise(
            "v_rcp_f32 $0, $1", "=v,v", [den], dtype=gl.float32, is_pure=True, pack=1
        )
        residual = gl.fma(-den, inv_den, 1.0)
        inv_den = gl.fma(inv_den, residual, inv_den)
        w = gl.reshape(e * gl.expand_dims(inv_den, 2), [16, R])
        if TQ == 1:
            w = gl.convert_layout(w, MFMA)
        else:
            dot_rows = gl.reshape(dots, [TQ, NI, R])
            weights = gl.convert_layout(w, gl.SliceLayout(0, dot_rows.type.layout))
            _, weights = gl.broadcast(dot_rows, gl.expand_dims(weights, 0))
            w = gl.reshape(weights, [M, R])
        dot0, dot1, dot2, dot3 = _split_pool4(dots)
        w0, w1, w2, w3 = _split_pool4(w)
        cent = gl.fma(dot1, w1, dot0 * w0)
        cent = gl.fma(dot2, w2, cent)
        cent = gl.fma(dot3, w3, cent)
        all_cent = gl.permute(gl.reshape(cent, [NI, TQ, BB, 4]), [1, 2, 3, 0])
        block_score = gl.max(gl.reshape(all_cent, [TQ, BB, 4 * NI]), axis=2)
    else:
        all_rows = gl.permute(gl.reshape(dots, [NI, TQ, BB, ROWS]), [1, 2, 3, 0])
        block_score = gl.max(gl.reshape(all_rows, [TQ, BB, ROWS * NI]), axis=2)
    score = gl.convert_layout(block_score, SL) * sc
    jb = ti * BB + jt
    jb2 = gl.expand_dims(jb, 0)
    qb2 = gl.expand_dims(qb_t, 1)
    score = gl.where(jb2 < INIT, 1e30, score)
    score = gl.where(jb2 >= qb2 + 1 - LOCAL, 1e29, score)
    score = gl.where(jb2 <= qb2, score, float("-inf"))
    smask = gl.expand_dims(mine, 1) & (jb2 < NB)
    if BIG_OUT:
        gl.store(out_ptr + orow + jb2, score, mask=smask)
    else:
        gl.amd.cdna4.buffer_store(score, out_ptr, orow + jb2, mask=smask)


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
    FIRST_TILE: gl.constexpr,
    BIG_Q: gl.constexpr,
    BIG_OUT: gl.constexpr,
):
    """Block scores for ``TQ`` query tokens over every ``NSPLIT``-th tile of ``BB`` blocks."""
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
    SL: gl.constexpr = gl.BlockedLayout([1, 1], [64, 1], [1, 4], [0, 1])
    FL: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [1, 4], [1, 0])
    RL: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, WL3))

    t0 = gl.program_id(0) * TQ
    pid_s = gl.program_id(1)
    ns = NSPLIT
    icw_ptr = ic_ptr.to(gl.pointer_type(gl.int32), bitcast=True)
    EARLY: gl.constexpr = PIPE and SINGLE_SEQ
    QLINE: gl.constexpr = TQ == 1
    q_line = (iq_ptr.to(gl.int64) + 15) & -16
    FT: gl.constexpr = FIRST_TILE and not EARLY

    if FT:
        q_rows = t0 + gl.arange(0, TQ, layout=gl.SliceLayout(1, SL))
        tval = q_rows < T
        pos_t, seq_t = _load_meta(pos_ptr, seq_ptr, q_rows, tval, SINGLE_SEQ)
        qb_t = pos_t // 16

    rm = gl.arange(0, M, layout=gl.SliceLayout(1, QL))
    dd = gl.arange(0, DI, layout=gl.SliceLayout(0, QL))
    tm = t0 + rm % TQ
    ka = gl.arange(0, DI, layout=gl.SliceLayout(0, AL))
    dperm = gl.max_contiguous(
        gl.multiple_of(32 * ((ka // 8) % 4) + 8 * (ka // 32) + ka % 8, 8), 8
    )
    if EARLY:
        ph_0 = _phys_any(bt_ptr, pid_s * BB, NB, ROWS, R, RL)
        ph_1 = _phys_any(bt_ptr, (pid_s + ns) * BB, NB, ROWS, R, RL)
        ph_a0 = _phys_any(bt_ptr, (pid_s + 2 * ns) * BB, NB, ROWS, R, RL)
        ph_b0 = _phys_any(bt_ptr, (pid_s + 3 * ns) * BB, NB, ROWS, R, RL)
        q_rows = t0 + gl.arange(0, TQ, layout=gl.SliceLayout(1, SL))
        tval = q_rows < T
        pos_t, seq_t = _load_meta(pos_ptr, seq_ptr, q_rows, tval, True)
        qb_t = pos_t // 16
        q_raw = _load_q(iq_ptr, tm, rm, dd, T, TQ, BIG_Q)
        if POOL:
            r16e = gl.arange(0, 16, layout=gl.SliceLayout(1, AL))
            ca_raw = gl.load(
                cq_ptr + gl.expand_dims(r16e * 0, 1) + gl.expand_dims(dperm, 0)
            )
        qb0 = gl.max(gl.where(tval, qb_t, -1), axis=0)
        lv_0 = _live_rows(pid_s * BB, qb0, NB, ROWS, R, RL)
        lv_1 = _live_rows((pid_s + ns) * BB, qb0, NB, ROWS, R, RL)
        lv_a0 = _live_rows((pid_s + 2 * ns) * BB, qb0, NB, ROWS, R, RL)
        lv_b0 = _live_rows((pid_s + 3 * ns) * BB, qb0, NB, ROWS, R, RL)
        gl.amd.hint.sched_barrier()
        w_a0, s_a0 = _load_tile(
            ic_ptr, icw_ptr, ph_0, lv_0, ROWS, R, WL3, BIG_IC, QLINE, q_line
        )
        gl.amd.hint.sched_barrier()
        w_b0, s_b0 = _load_tile(
            ic_ptr, icw_ptr, ph_1, lv_1, ROWS, R, WL3, BIG_IC, QLINE, q_line
        )
        gl.amd.hint.sched_barrier()
        q = q_raw.to(gl.float32)
    elif FT:
        q_raw = _load_q(iq_ptr, tm, rm, dd, T, TQ, BIG_Q)
        qb_first = gl.max(gl.where(tval, qb_t, -1), axis=0)
        ph_first, lv_first = _phys_rows(bt_ptr, pid_s * BB, qb_first, NB, ROWS, R, RL)
        w_first, s_first = _load_tile(
            ic_ptr, icw_ptr, ph_first, lv_first, ROWS, R, WL3, BIG_IC, QLINE, q_line
        )
        gl.amd.hint.sched_barrier()
        q = q_raw.to(gl.float32)
    else:
        q = _load_q(iq_ptr, tm, rm, dd, T, TQ, BIG_Q).to(gl.float32)
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
    if POOL:
        if EARLY:
            ca = ca_raw.to(gl.float16)
        else:
            r16 = gl.arange(0, 16, layout=gl.SliceLayout(1, AL))
            ca = gl.load(
                cq_ptr + gl.expand_dims(r16 * 0, 1) + gl.expand_dims(dperm, 0)
            ).to(gl.float16)
    else:
        ca = qa

    if not (FT or EARLY):
        q_rows = t0 + gl.arange(0, TQ, layout=gl.SliceLayout(1, SL))
        tval = q_rows < T
        pos_t, seq_t = _load_meta(pos_ptr, seq_ptr, q_rows, tval, SINGLE_SEQ)
        qb_t = pos_t // 16
    jt = gl.arange(0, BB, layout=gl.SliceLayout(0, SL))
    if BIG_OUT:
        orow = gl.expand_dims(q_rows.to(gl.int64) * NB, 1)
    else:
        orow = gl.expand_dims(q_rows * NB, 1)

    big = 2147483647
    s_lo = gl.min(gl.where(tval, seq_t, big), axis=0)
    s_hi = gl.max(gl.where(tval, seq_t, -1), axis=0)
    s_sub = s_lo
    while s_sub <= s_hi:
        mine = tval & (seq_t == s_sub)
        if SINGLE_SEQ:
            bt_row = bt_ptr
        else:
            bt_row = bt_ptr + s_sub.to(gl.int64) * s_bt
        qbmax = gl.max(gl.where(mine, qb_t, -1), axis=0)
        ntile = qbmax // BB + 1
        if EARLY:
            w_a, s_a, w_b, s_b = w_a0, s_a0, w_b0, s_b0
            ph_a, lv_a, ph_b, lv_b = ph_a0, lv_a0, ph_b0, lv_b0
        elif PIPE:
            if not FT:
                ph, lv = _phys_rows(bt_row, pid_s * BB, qbmax, NB, ROWS, R, RL)
            ph_1, lv_1 = _phys_rows(bt_row, (pid_s + ns) * BB, qbmax, NB, ROWS, R, RL)
            ph_a, lv_a = _phys_rows(
                bt_row, (pid_s + 2 * ns) * BB, qbmax, NB, ROWS, R, RL
            )
            ph_b, lv_b = _phys_rows(
                bt_row, (pid_s + 3 * ns) * BB, qbmax, NB, ROWS, R, RL
            )
            gl.amd.hint.sched_barrier()
            if FT:
                w_a, s_a = w_first, s_first
            else:
                w_a, s_a = _load_tile(
                    ic_ptr, icw_ptr, ph, lv, ROWS, R, WL3, BIG_IC, QLINE, q_line
                )
            gl.amd.hint.sched_barrier()
            w_b, s_b = _load_tile(
                ic_ptr, icw_ptr, ph_1, lv_1, ROWS, R, WL3, BIG_IC, QLINE, q_line
            )
            gl.amd.hint.sched_barrier()
        if PIPE:
            for ti in range(pid_s, ntile, 2 * ns):
                kb = _keys_b(w_a, s_a, R, BL)
                gl.amd.hint.sched_barrier()
                w_a, s_a = _load_tile(
                    ic_ptr, icw_ptr, ph_a, lv_a, ROWS, R, WL3, BIG_IC, QLINE, q_line
                )
                ph_a, lv_a = _phys_rows(
                    bt_row, (ti + 4 * ns) * BB, qbmax, NB, ROWS, R, RL
                )
                _score_tile(qa, ca, kb, ti, jt, qb_t, out_ptr, orow, mine, sc, pool_temp, NB,
                            INIT, LOCAL, TQ, BB, ROWS, POOL, MFMA,
                            SL, BIG_OUT)  # fmt: skip
                kb = _keys_b(w_b, s_b, R, BL)
                gl.amd.hint.sched_barrier()
                w_b, s_b = _load_tile(
                    ic_ptr, icw_ptr, ph_b, lv_b, ROWS, R, WL3, BIG_IC, QLINE, q_line
                )
                ph_b, lv_b = _phys_rows(
                    bt_row, (ti + 5 * ns) * BB, qbmax, NB, ROWS, R, RL
                )
                if ti + ns < ntile:
                    _score_tile(qa, ca, kb, ti + ns, jt, qb_t, out_ptr, orow, mine, sc,
                                pool_temp, NB, INIT, LOCAL, TQ, BB, ROWS, POOL, MFMA,
                                SL, BIG_OUT)  # fmt: skip
        else:
            for ti in range(pid_s, ntile, ns):
                if FT:
                    w_c, s_c = w_first, s_first
                else:
                    ph, lv = _phys_rows(bt_row, ti * BB, qbmax, NB, ROWS, R, RL)
                    w_c, s_c = _load_tile(
                        ic_ptr, icw_ptr, ph, lv, ROWS, R, WL3, BIG_IC, QLINE, q_line
                    )
                kb = _keys_b(w_c, s_c, R, BL)
                _score_tile(qa, ca, kb, ti, jt, qb_t, out_ptr, orow, mine, sc, pool_temp, NB,
                            INIT, LOCAL, TQ, BB, ROWS, POOL, MFMA,
                            SL, BIG_OUT)  # fmt: skip
        if not CAUSAL:
            f1 = gl.minimum(NB, (pid_s + 1) * FILL)
            neg = gl.full([TQ, FW], float("-inf"), gl.float32, layout=FL)
            q_fill_rows = t0 + gl.arange(0, TQ, layout=gl.SliceLayout(1, FL))
            minef = gl.convert_layout(mine, gl.SliceLayout(1, FL))
            if BIG_OUT:
                orowf = gl.expand_dims(q_fill_rows.to(gl.int64) * NB, 1)
            else:
                orowf = gl.expand_dims(q_fill_rows * NB, 1)
            for c0 in range(gl.maximum(ntile * BB, pid_s * FILL), f1, FW):
                cols = gl.expand_dims(
                    c0 + gl.arange(0, FW, layout=gl.SliceLayout(0, FL)), 0
                )
                fmask = gl.expand_dims(minef, 1) & (cols < f1)
                if BIG_OUT:
                    gl.store(out_ptr + orowf + cols, neg, mask=fmask)
                else:
                    gl.amd.cdna4.buffer_store(neg, out_ptr, orowf + cols, mask=fmask)
        if SINGLE_SEQ:
            s_sub = big
        else:
            s_sub = gl.min(gl.where(tval & (seq_t > s_sub), seq_t, big), axis=0)


@lru_cache(maxsize=16)
def _words_layout(r, wm=1):
    """Distribution of a tile's words ``[R, 4, 4]`` in the MFMA B operand (cached: building
    the layout costs about 13 us a call)."""
    assert r in (64, 128) and wm in (1, 2, 4), (r, wm)
    wn = 4 // wm
    row_bits = [16 << i for i in range((r // 16).bit_length() - 1)]
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
    """Validate the inputs, pick the tiling and launch ``_score_kernel``."""
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
        bb = 4 if rows == PAGE else 16
    if bb * rows not in (64, 128):
        raise ValueError("bb must form a tile of 64 or 128 cache rows")
    if waves is None:
        waves = 2 if tq > 1 else 4
    nq = triton.cdiv(t, tq)
    if not isinstance(waves, int) or waves <= 0:
        raise ValueError("waves must be a positive integer")
    nsplit = max(1, min(triton.cdiv(nb, bb), triton.cdiv(waves * cus, nq)))
    if pipe is None:
        pipe = nsplit * bb < nb
    os.environ.setdefault("TRITON_ALLOW_FP_FUSION", "1")
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
        BIG_IC=ic.numel() >= _BUFFER_BYTES,
        BIG_Q=iq.numel() * 2 >= _BUFFER_BYTES,
        BIG_OUT=t * nb * 4 >= _BUFFER_BYTES,
        FIRST_TILE=(
            rows == PAGE
            and block_tables.shape[0] == 1
            and wm == 1
            and tq == 1
            and not pipe
            and nsplit * bb >= nb
        ),
        num_warps=NUM_WARPS,
        enable_fp_fusion=True,
        llvm_fn_attrs=(
            ("amdgpu-ieee", "false"),
            ("target-features", "-packed-fp32-ops"),
        ),
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
