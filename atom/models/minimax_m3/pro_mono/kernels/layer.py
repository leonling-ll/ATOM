# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The phase-1 MoE layer kernel of MiniMax-M3.1-Pro: o_proj, the first
all-reduce and the post-attention Gemma RMSNorm with its MXFP8 output, plus the
lookahead norm of layer L + 2 (``upto = "norm"``), then the MoE to this rank's
FFN partial (``upto = "ffn"``: ``kernels/ffn.py``). Stages: ``layout``'s
docstring.

The numerics follow production's recipe at <= 16 rows under the combo config
(Lamport all-reduce, then aiter ``add_rmsnorm_mxfp8(..., gemma=True)``)::

    a    = bf16(fp32 sum of the 8 bf16 partials, rank order 0 .. 7)
    v    = f32(a) + f32(res)                    (unrounded)
    h_mid = bf16(v)
    rstd = rsq(sum_f32(v * v) / 7168 + eps)
    x    = bf16((v * rstd) * (f32(w) + 1))
    x8   = MXFP8 1x32 of x: code = ceil_pow2(max(amax, 1e-10) * f32(1 / 448))

and the lookahead (aiter ``fused_qk_rmsnorm_group_quant``, gemma, on the bf16
residual)::

    rstd2 = rsq(sum_f32(bf16(v)^2) / 7168 + eps2)
    la8   = MXFP8 of bf16((bf16(v) * rstd2) * (f32(w2) + 1))

o_proj's input is quantized as ``quantize_fp8`` does it (the same recipe, on the
bf16 core rows), so the GEMM reads production's operand bytes; its sum order
over K differs from aiter's tuned GEMM, so a partial can differ in its last bf16
bit.

The recipe, not every bit: the sums of squares here are a per-task wave sum
then a fixed butterfly over the 56 tasks (not aiter's order), and ``/ 7168`` is
not shown to be the true divide aiter uses, so x, x8 and la8 may differ from
production by an ulp or one MXFP8 code. h_mid is bit-exact when the partials
are, because the AR sums the ranks' partials in the Lamport all-reduce's order;
x / x8 / la8 are held to a relative error norm that passes every partial one
bf16 ulp off, with their E8M0 scales at most one step from the reference's
(``tests/models/minimax_m3_pro_mono_ref.py``).

The o_proj stage is MiniMax-M3's MXFP8 o stage
(``minimax_m3/mono/kernels/stages/attention.py``) at Pro's K.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int32, Int64, T

from atom.models.minimax_m3.pro_mono.config import COLS, EXPERTS, HIDDEN, TOPK
from atom.models.minimax_m3.pro_mono.kernels import ffn
from atom.models.minimax_m3.pro_mono.layout import (
    ABI_NAMES,
    FFN_REGIONS,
    MIDB_ROW,
    NORM_REGIONS,
    NT,
    O_CPW,
    O_K,
    O_ROW,
    O_ROWS,
    O_SCALES_PW,
    O_TASKS,
    O_TASKS_OWN,
    OSC_ROW,
    OWN,
    REGIONS,
    SM_ROW,
    SMSC_WORDS,
    TL_POINTS,
    TOKEN_TILE,
    TP,
    X4_ROW,
    X8_ROW,
    XH_ROW,
    XSC_WORDS,
    P1Build,
    ag_pair,
    agsq_pair,
    max_u,
    peer_layout,
    plain_layout,
    rs_pair,
    scratch_layout,
    token_tiles,
)
from atom.models.minimax_m3.pro_mono.sources import SOURCES
from atom.mono.device.mx import (
    FP8_MAX,
    clamp_fp8,
    code_ceil,
    mfma_scaled,
    rcp_pow2,
)
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf2_f32,
    bf16_pair,
    bf16_round,
    butterfly,
    fp8_pack4,
    hw_rsq,
    kernel_symbol,
    launcher,
    load_ptr64,
    row_sum,
    rsrc,
    traced,
    wave_sum,
    xshfl,
)
from atom.mono.device.ranks import peer_bases
from atom.mono.device.stamps import stamp, stamp_begin, stamp_flush
from atom.mono.device.sync import Mailbox, preg, publish, sreg
from atom.mono.plan.build_key import key_tuple, symbol_params
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.trace import enter_stage
from atom.mono.runtime.abi import KernelAbi
from atom.mono.runtime.debug import region_ids

ABI = KernelAbi(ABI_NAMES)
_STREAM = fx.Stream(None)
# the core rows' load unit: 8 bf16 (one dwordx4), 4 a 32-block
CORE_ITEMS = O_K // 8


def _f32_of(w):
    return fx.Int32(w).bitcast(fx.Float32)


def _ld_words(addr, word, n):
    """``n`` (1 or 2) plain i32 words of a buffer from word ``word``."""
    if const_expr(n == 1):
        return [fx.Int32(bo.buffer_load(rsrc(addr), word, vec_width=1, dtype=T.i32))]
    v = fx.Vector(bo.buffer_load(rsrc(addr), word, vec_width=n, dtype=T.i32))
    return [fx.Int32(v[e]) for e in range(n)]


def _st_words(words, addr, word, cm=0):
    if const_expr(len(words) == 1):
        bo.buffer_store(fx.Int32(words[0]), rsrc(addr), word, cache_modifier=cm)
    else:
        bo.buffer_store(
            fx.Vector.from_elements(words, fx.Int32),
            rsrc(addr),
            word,
            cache_modifier=cm,
        )


def _unpack(words):
    """bf16 pair words -> f32 values, low half first."""
    out = []
    for w in words:
        out += list(bf2_f32(w))
    return out


def _pack(vals):
    """f32 values -> bf16 pair words (RNE)."""
    return [
        bf16_pair(vals[2 * i], vals[2 * i + 1]).bitcast(fx.Int32)
        for i in range(len(vals) // 2)
    ]


def _absmax(vals):
    am = fx.max(vals[0], -vals[0])
    for v in vals[1:]:
        am = fx.max(am, fx.max(v, -v))
    return am


def _code(am):
    """The E8M0 code of a 32-block of absolute maximum ``am``: ``quantize_fp8``'s
    ceil_pow2(max(amax, 1e-10) * f32(1 / 448))."""
    return code_ceil(fx.max(am, fx.Float32(1e-10)) * fx.Float32(1.0 / FP8_MAX))


def _or(a, b):
    return a | b


# ------------------------------------------------------------------ o_proj


def core_loads(c):
    """The core rows' loads (bf16 [S, 3072], plain: the attention wrote them),
    every round of ``load_core``'s strided loop, issued before anything else
    of the o stage: a round's stores are conditional, so a load in a later
    round went out only after the earlier round's use, which waited (``vmcnt``
    in issue order) for every w_o load issued before it -- 5 serial L2 round
    trips at 8 rows, 11 at 16."""
    s, tid = c["S"], c["tid"]
    n = s * CORE_ITEMS
    r_core = rsrc(c["core"])
    out = []
    for i in range_constexpr(-(-n // THREADS)):
        e = fx.min(tid + THREADS * i, n - 1)
        t = e // CORE_ITEMS
        q = e % CORE_ITEMS
        out.append(
            fx.Vector(
                bo.buffer_load(r_core, (t * O_K + 8 * q) // 2, vec_width=4, dtype=T.i32)
            )
        )
    # issued here, ahead of w_o's: the scheduler would sink them past it
    rocdl.sched_barrier(0)
    return out


@traced
def load_core(c, loads):
    """The core rows (``core_loads``) -> MXFP8 1x32 in LDS: e4m3 words at
    ``xs`` (token t's at t O_ROW), their E8M0 codes four a word at ``osc``. A
    thread takes 8 values, 4 threads a block; S * 384 items is a multiple of
    16, so a 16-lane group is either all live or all past the end (clamped,
    nothing stored)."""
    s, tid = c["S"], c["tid"]
    n = s * CORE_ITEMS
    for i in range_constexpr(len(loads)):
        e = fx.min(tid + THREADS * i, n - 1)
        t = e // CORE_ITEMS
        q = e % CORE_ITEMS
        v = loads[i]
        vals = _unpack([fx.Int32(v[d]) for d in range(4)])
        code = _code(butterfly(_absmax(vals), (1, 2), fx.max))
        inv = rcp_pow2(code)
        qv = [clamp_fp8(x * inv) for x in vals]
        w0 = fp8_pack4(qv[0], qv[1], qv[2], qv[3])
        w1 = fp8_pack4(qv[4], qv[5], qv[6], qv[7])
        # blocks 4 m .. 4 m + 3 (items 16 m ..) in one word, block k in byte k
        codes = butterfly(code << (8 * ((q // 4) % 4)), (4, 8), _or)
        if tid + THREADS * i < n:
            fx.ptr_store(w0, c["xs"] + (t * O_ROW + 2 * q))
            fx.ptr_store(w1, c["xs"] + (t * O_ROW + 2 * q + 1))
            if q % 16 == 0:
                fx.ptr_store(codes, c["osc"] + (t * OSC_ROW + q // 16))


@traced
def stage_o(c):
    """CTA t < O_TASKS: o_proj rows 32 t .. + 32 (two 16-row groups, a group's
    4 waves each a quarter of K) -> bf16 partial rows -> the owner rank's
    ``rs`` (this rank's slot). The weight and its scales are in flight while
    the core rows are quantized, the core rows' loads issued first (their
    quantization then waits for them alone). Past 16 rows a wave runs its
    MFMAs once a token tile (``token_tiles``) on the same weight registers,
    and the K-quarter sum runs a pass a tile."""
    s, tid, lane, wave, bid = c["S"], c["tid"], c["lane"], c["wave"], c["bid"]
    g4 = lane // 16
    l16 = lane % 16
    q4 = wave % 4
    v4f = fx.Vector.make_type(4, fx.Float32)
    if bid < O_TASKS:
        t = bid
        rg = t * 2 + wave // 4
        core = core_loads(c)
        r_wo = rsrc(c["w_o"])
        wts = [
            fx.Vector(
                bo.buffer_load(
                    r_wo,
                    ((rg * (O_K // 32) + (q4 * O_CPW + cc) * 2) * 512 + lane * 16) // 4,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT,
                )
            )
            for cc in range(O_CPW)
        ]
        # lane (row l16, block g4): the E8M0 bytes of this wave's 24 blocks,
        # row-major [7168, 96]; MFMA p takes block g4 of dword p
        sw = (rg * 16 + l16) * (O_K // 32 // 4) + q4 * (O_SCALES_PW // 4)
        r_so = rsrc(c["s_o"])
        s_lo = fx.Vector(bo.buffer_load(r_so, sw, vec_width=4, dtype=T.i32))
        s_hi = fx.Vector(bo.buffer_load(r_so, sw + 4, vec_width=2, dtype=T.i32))
        ws = [s_lo[d] for d in range(4)] + [s_hi[d] for d in range(2)]
        load_core(c, core)
        gpu.barrier()
        c["stamp"](1)
        # B column l16 of tile k is token 16 k + l16 (a token past S: its
        # column is discarded); chunks 2p, 2p + 1 in one 16x16x128 scaled
        # MFMA, its lane holding k 16 g4 .. + 16 of each 64-k chunk
        tiles = token_tiles(s)
        col_tok = [fx.min(TOKEN_TILE * k + l16, s - 1) for k in range(tiles)]
        acc = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(tiles)]
        for p in range_constexpr(O_CPW // 2):
            for k in range_constexpr(tiles):
                xb = [
                    fx.Vector(
                        fx.ptr_load(
                            c["xs"]
                            + (
                                col_tok[k] * O_ROW
                                + (q4 * O_CPW + 2 * p + h) * 16
                                + g4 * 4
                            ),
                            result_type=v4f,
                        )
                    ).bitcast(fx.Int32)
                    for h in range(2)
                ]
                sb = fx.ptr_load(
                    c["osc"] + (col_tok[k] * OSC_ROW + q4 * (O_CPW // 2) + p)
                )
                acc[k] = mfma_scaled(
                    fx.Vector.from_elements(
                        [wts[2 * p + h][e] for h in range(2) for e in range(4)],
                        fx.Int32,
                    ),
                    fx.Vector.from_elements(
                        [xb[h][e] for h in range(2) for e in range(4)], fx.Int32
                    ),
                    acc[k],
                    fx.Int32(ws[p]) >> (g4 * 8),
                    fx.Int32(sb) >> (g4 * 8),
                )
        # tile k's accumulators a tile of ``red`` on
        for k in range_constexpr(tiles):
            fx.ptr_store(acc[k], c["red"] + ((k * WAVES + wave) * 64 + lane) * 4)
        gpu.barrier()
        # the 4 K quarters summed in wave order: thread (token, row of 32), a
        # pass a token tile (THREADS = O_ROWS x TOKEN_TILE cells)
        for k in range_constexpr(tiles):
            if tid < O_ROWS * (min(s, TOKEN_TILE * (k + 1)) - TOKEN_TILE * k):
                tk = tid // O_ROWS
                grp = (tid % O_ROWS) // 16
                tot = row_sum(
                    c["red"] + k * WAVES * 64 * 4,
                    tid % 16,
                    tk,
                    [grp * 4 + w for w in range(4)],
                )
                fx.ptr_store(bf16_round(tot), c["misc"] + (tid + THREADS * k))
        gpu.barrier()
        # a pair of rows a thread -> the owner of these columns, this rank's slot
        if tid < (O_ROWS // 2) * s:
            tk = tid // (O_ROWS // 2)
            rp = tid % (O_ROWS // 2)
            v0 = fx.ptr_load(c["misc"] + (tk * O_ROWS + 2 * rp))
            v1 = fx.ptr_load(c["misc"] + (tk * O_ROWS + 2 * rp + 1))
            col = (t % O_TASKS_OWN) * O_ROWS + 2 * rp
            c["put_bf"](
                preg(c["o_owner"], c["off"]["rs"], "rs"),
                2 * rs_pair(c["rank"], tk, col),
                [v0, v1],
            )


# ------------------------------------------------------- all-reduce + norm


@traced
def ar_reduce(c, ins):
    """Stages reduce / gather, the wave's tokens (one to three passes) at once:
    -> the f32 v = bf16(sum) + res of this lane's two columns, a token each.
    The owner of the columns polls every token's partials of the 8 ranks in
    one batch, sums each token's in rank order and sends its bf16 sum and the
    task's sums of squares to every rank; every other rank polls every
    token's sum in one batch. (A poll a token cost a second round trip at 9
    .. 16 rows, the owner's before its second sum went out.)"""
    lane, la = c["lane"], c["lookahead"]
    n = len(ins)
    # bound before the traced ifs: names their branches bind are not seen
    # past them otherwise (a second pass's, unused at <= 8 rows; a third's,
    # unused at <= 16)
    a0, a1 = fx.Float32(0.0), fx.Float32(0.0)
    b0, b1 = fx.Float32(0.0), fx.Float32(0.0)
    d0, d1 = fx.Float32(0.0), fx.Float32(0.0)
    enter_stage("reduce")
    if c["is_owner"]:
        own_rs = preg(c["own"], c["off"]["rs"], "rs")
        ws = c["poll"](
            [
                (own_rs, rs_pair(src, ins[k][0], ins[k][1] - c["obeg"]), 1)
                for k in range(n)
                for src in range(TP)
            ],
            batch=n * TP,
        )
        held_o = []
        for k in range_constexpr(n):
            t, col, rr = ins[k][0], ins[k][1], ins[k][2]
            # rank order 0 .. TP - 1 (``ranks.sum_partials``'): every rank agrees
            lo, hi = bf2_f32(ws[k * TP][0])
            for src in range_constexpr(1, TP):
                plo, phi = bf2_f32(ws[k * TP + src][0])
                lo = lo + plo
                hi = hi + phi
            lo, hi = bf16_round(lo), bf16_round(hi)
            for p in range_constexpr(TP):
                c["put_bf"](
                    preg(c["bases"][p], c["off"]["ag"], "ag"),
                    2 * ag_pair(t, col),
                    [lo, hi],
                )
            v0 = lo + rr[0]
            v1 = hi + rr[1]
            words = [wave_sum(v0 * v0 + v1 * v1).bitcast(fx.Int32)]
            if const_expr(la):
                q0, q1 = bf16_round(v0), bf16_round(v1)
                words.append(wave_sum(q0 * q0 + q1 * q1).bitcast(fx.Int32))
            if lane == 0:
                for p in range_constexpr(TP):
                    c["put_words"](
                        preg(c["bases"][p], c["off"]["agsq"], "agsq"),
                        agsq_pair(c["j"], t),
                        words,
                    )
            held_o.append((v0, v1))
        a0, a1 = held_o[0]
        if const_expr(n >= 2):
            b0, b1 = held_o[1]
        if const_expr(n == 3):
            d0, d1 = held_o[2]
    enter_stage("gather")
    if c["not_owner"]:
        w = c["poll"](
            [
                (preg(c["own"], c["off"]["ag"], "ag"), ag_pair(ins[k][0], ins[k][1]), 1)
                for k in range(n)
            ]
        )
        g0, g1 = bf2_f32(w[0][0])
        a0 = g0 + ins[0][2][0]
        a1 = g1 + ins[0][2][1]
        if const_expr(n >= 2):
            h0, h1 = bf2_f32(w[1][0])
            b0 = h0 + ins[1][2][0]
            b1 = h1 + ins[1][2][1]
        if const_expr(n == 3):
            e0, e1 = bf2_f32(w[2][0])
            d0 = e0 + ins[2][2][0]
            d1 = e1 + ins[2][2][1]
    return [[a0, a1], [b0, b1], [d0, d1]][:n]


@traced
def mx_store(c, q8, q8s, t, col, o, cm=0):
    """MXFP8 1x32 of this lane's two bf16 outputs ``o`` (16 lanes a block):
    e4m3 words (an even lane stores its and its neighbour's) at ``q8`` [S,
    7168] and the task's 4 codes in one dword of ``q8s`` [S, 224]; ``cm``:
    device scope where this launch's FFN reads them."""
    lane = c["lane"]
    code = _code(butterfly(_absmax(o), (1, 2, 4, 8), fx.max))
    inv = rcp_pow2(code)
    qv = [clamp_fp8(oe * inv) for oe in o]
    nb = [xshfl(qe, 1) for qe in qv]
    word = fp8_pack4(qv[0], qv[1], nb[0], nb[1])
    if lane % 2 == 0:
        bo.buffer_store(word, rsrc(q8), (t * HIDDEN + col) // 4, cache_modifier=cm)
    # blocks 0 .. 3 of the task on lanes 0 / 16 / 32 / 48: byte k of one dword
    codes = butterfly(code << (8 * (lane // 16)), (16, 32), _or)
    if lane == 0:
        bo.buffer_store(
            codes, rsrc(q8s), t * (HIDDEN // COLS) + c["j"], cache_modifier=cm
        )


@traced
def ar_norm(c, ins, held, w_post, w_ahead):
    """Stage norm: the 56 tasks' sums of squares (lane q polls task q's; one
    wave sums them in a fixed butterfly, so every CTA and rank gets the same
    bits), then the stores."""
    lane, la = c["lane"], c["lookahead"]
    enter_stage("norm")
    sq_at = preg(c["own"], c["off"]["agsq"], "agsq")
    q = fx.min(lane, NT - 1)
    got = c["poll"](
        [(sq_at, agsq_pair(q, ins[k][0]), 2 if la else 1) for k in range(len(ins))]
    )
    c["stamp"](4)
    wf = [w + fx.Float32(1.0) for w in _unpack(w_post)]
    if const_expr(la):
        wf2 = [w + fx.Float32(1.0) for w in _unpack(w_ahead)]
    for k in range_constexpr(len(ins)):
        t, col = ins[k][0], ins[k][1]
        word = (t * HIDDEN + col) // 2
        part = (lane < NT).select(_f32_of(got[k][0]), fx.Float32(0.0))
        r = hw_rsq(wave_sum(part) / float(HIDDEN) + fx.Float32(c["eps"]))
        v = held[k]
        y = [(v[e] * r) * wf[e] for e in range(2)]
        ffn_on = c["ffn_on"]
        cm = CM_DEV if const_expr(ffn_on) else 0
        _st_words(_pack(v), c["h_mid"], word)
        _st_words(_pack(y), c["x"], word, cm)
        yb = [bf16_round(ye) for ye in y]
        mx_store(c, c["x8"], c["x8s"], t, col, yb, cm)
        if const_expr(ffn_on and c["routed_act"] == "fp4"):
            ffn.mx4_store(c, t, col, yb)
        if const_expr(la):
            part2 = (lane < NT).select(_f32_of(got[k][1]), fx.Float32(0.0))
            r2 = hw_rsq(wave_sum(part2) / float(HIDDEN) + fx.Float32(c["eps_ahead"]))
            vb = [bf16_round(ve) for ve in v]
            y2 = [bf16_round((vb[e] * r2) * wf2[e]) for e in range(2)]
            mx_store(c, c["la8"], c["la8s"], t, col, y2)


@traced
def ar_tokens(c, ps):
    """A wave's tokens (pass indices ``ps``: token wave + 8 p) of this CTA's
    128 columns, two a lane: reduce, gather, norm."""
    lane, wave, j = c["lane"], c["wave"], c["j"]
    col = j * COLS + 2 * lane
    ins = []
    for k in range_constexpr(len(ps)):
        t = wave + 8 * ps[k]
        ins.append((t, col, _unpack(_ld_words(c["res"], (t * HIDDEN + col) // 2, 1))))
    # the weights before any poll
    w_post = _ld_words(c["g_post"], col // 2, 1)
    w_ahead = None
    if const_expr(c["lookahead"]):
        w_ahead = _ld_words(c["g_ahead"], col // 2, 1)
    held = ar_reduce(c, ins)
    c["stamp"](3)
    ar_norm(c, ins, held, w_post, w_ahead)


# ------------------------------------------------------------------ build


@functools.cache
def build(key: P1Build):
    """The launcher of ``key`` (cached per process); its kernel takes
    ``ABI_NAMES``."""
    s = key.tokens
    ffn_on = key.upto == "ffn"
    passes = -(-s // WAVES)
    play = peer_layout()
    slay = scratch_layout(key)
    play_plain = plain_layout(key)
    tl_on = key.timeline
    keyed = key_tuple(key, SOURCES)
    name = kernel_symbol("minimax_m3_pro_mono_layer", **symbol_params(key))

    fp8 = ffn_on and key.routed_act == "fp8"
    bf16 = ffn_on and key.routed_act == "bf16"
    # ``xs``: the o stage's MXFP8 core rows, then (FFN) the router's bf16 x
    # slice, a W4A8 build's MXFP8 x rows or an A16W4 build's bf16 x K halves
    # (the routed B operands: union to ug) and the shared mid's MXFP8 rows
    # (down; A16W4 every pick's bf16 mid row past them, ``ffn.load_mids``) --
    # stages apart, past barriers
    xs_row = O_ROW
    if fp8:
        xs_row = max(O_ROW, X8_ROW)
    if bf16:
        xs_row = max(O_ROW, XH_ROW)
    assert xs_row >= (ffn.R_KLEN + 8) // 2 and xs_row >= SM_ROW
    # A16W4's down: the staged mids past the shared mid's rows. A build that
    # shrinks ``xs`` a token (a wider one, to fit the CU's LDS) has to stage
    # them in passes instead
    assert not bf16 or xs_row >= SM_ROW + TOPK * MIDB_ROW, (
        f"A16W4 staged mids need {SM_ROW + TOPK * MIDB_ROW} xs words a token, "
        f"the build has {xs_row}"
    )

    @fx.struct
    class Smem:
        xs: fx.Array[fx.Int32, s * xs_row, 16]
        osc: fx.Array[fx.Int32, s * OSC_ROW, 16]
        # a wave's 16x16 accumulators, a token tile's after another
        red: fx.Array[fx.Float32, token_tiles(s) * WAVES * 64 * 4, 16]
        misc: fx.Array[fx.Float32, O_ROWS * s, 16]
        tls: fx.Array[fx.Int64, TL_POINTS, 16]

    mw = ffn.mask_words(s)
    nu_max = max_u(s)
    slots = ffn.ROUTE_SLOTS

    @fx.struct
    class FfnLds:
        # the W4A4 build's MXFP4 x rows (a W4A8 build's x8 rows and an A16W4
        # build's x halves are in ``xs``; 16 B unused there)
        x4l: fx.Array[fx.Int32, 4 if fp8 or bf16 else s * X4_ROW, 16]
        # the routed B operands' E8M0 codes, four a word (x4's or x8's; A16W4
        # has none: 16 B unused)
        xsl: fx.Array[fx.Int32, 4 if bf16 else s * XSC_WORDS, 16]
        smsc: fx.Array[fx.Int32, s * SMSC_WORDS, 16]
        contrib: fx.Array[fx.Float32, s * TOPK * ffn.DOWN_ROWS, 16]
        route: fx.Array[fx.Int32, s * 2 * TOPK, 16]
        mask: fx.Array[fx.Int32, EXPERTS * mw, 16]
        slot: fx.Array[fx.Int32, EXPERTS, 16]
        expert: fx.Array[fx.Int32, nu_max, 16]
        first: fx.Array[fx.Int32, nu_max + 1, 16]
        order: fx.Array[fx.Int32, s * TOPK, 16]
        cand: fx.Array[fx.Int32, 2 * slots, 16]
        csig: fx.Array[fx.Float32, slots, 16]
        scan: fx.Array[fx.Int32, WAVES, 16]
        claim: fx.Array[fx.Int32, 4, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def p1_layer(
        core: Int64, w_o: Int64, s_o: Int64, res: Int64, g_post: Int64,
        g_ahead: Int64, moe_w: Int64, h_mid: Int64, x: Int64, x8: Int64,
        x8s: Int64, la8: Int64, la8s: Int64, ffn_out: Int64, scratch: Int64,
        plain: Int64, nvalid: Int64,
        peers: Int64, rank: Int32, layer: Int32, tl: Int64,
    ):  # fmt: skip
        _ = keyed
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        tls = lds.tls.ptr
        stamp_begin(tl_on, tls, tid, TL_POINTS)
        if const_expr(key.debug):
            mb = Mailbox(
                layer, scratch + fx.Int64(slay["diag"][0]), region_ids(REGIONS)
            )
        else:
            mb = Mailbox(layer)
        lane = tid % 64
        wave = tid // 64
        j = bid
        owner = j // (OWN // COLS)
        c = {
            "S": s, "lookahead": key.lookahead, "eps": key.eps,
            "eps_ahead": key.eps_ahead, "tid": tid, "bid": bid, "lane": lane,
            "wave": wave, "j": j, "rank": rank,
            "bases": peer_bases(peers, TP), "own": load_ptr64(peers, rank),
            # the rank owning this CTA's o task's columns
            "o_owner": load_ptr64(
                peers, fx.min(bid, O_TASKS - 1) // O_TASKS_OWN
            ),
            "obeg": owner * OWN, "is_owner": owner == rank,
            "not_owner": owner != rank,
            "off": {r: play[r][0] for r in NORM_REGIONS},
            "core": core, "w_o": w_o, "s_o": s_o, "res": res,
            "g_post": g_post, "g_ahead": g_ahead, "h_mid": h_mid, "x": x,
            "x8": x8, "x8s": x8s, "la8": la8, "la8s": la8s,
            "xs": lds.xs.ptr, "osc": lds.osc.ptr, "red": lds.red.ptr,
            "misc": lds.misc.ptr,
            "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words,
            "poll": mb.poll, "peek": mb.peek, "stamp": lambda k: stamp(tl_on, tls, tid, k),
            "ffn_on": ffn_on, "routed_act": key.routed_act,
            "planes": key.planes if ffn_on else None,
        }  # fmt: skip
        if const_expr(ffn_on):
            # the norm writes MXFP4 x to ``plain`` (one buffer resource over
            # every plain region, a region a word offset); the rest of the
            # FFN's context is made past the norm
            c |= {
                "plain": plain,
                "pw": {r: off // 4 for r, (off, _) in play_plain.items()},
            }
        enter_stage("o")
        stage_o(c)
        stamp(tl_on, tls, tid, 2)
        if const_expr(passes == 1):
            if (bid < NT) & (wave < s):
                ar_tokens(c, [0])
        else:
            # waves holding a token in every pass; the rest skip the last
            every = s - WAVES * (passes - 1)
            if (bid < NT) & (wave < every):
                ar_tokens(c, list(range(passes)))
            if const_expr(every < WAVES):  # noqa: SIM102
                if (bid < NT) & (wave >= every):
                    ar_tokens(c, list(range(passes - 1)))
        stamp(tl_on, tls, tid, 5)
        if const_expr(ffn_on):
            fl = alloc.allocate(FfnLds).peek()
            c |= {
                "moe_w": moe_w, "ffn": ffn_out,
                "rlog": [
                    preg(b, play["rlog"][0], "rlog") for b in c["bases"]
                ],
                "rlog_own": preg(c["own"], play["rlog"][0], "rlog"),
                "xl": lds.xs.ptr, "sm8": lds.xs.ptr,
                "midl": lds.xs.ptr + s * SM_ROW,
                "xbl": lds.xs.ptr if fp8 or bf16 else fl.x4l.ptr,
                "xsl": fl.xsl.ptr,
                "smsc": fl.smsc.ptr,
                "contrib": fl.contrib.ptr, "cand": fl.cand.ptr,
                "csig": fl.csig.ptr, "scan": fl.scan.ptr,
                "claim": fl.claim.ptr, "nvalid": nvalid,
                "rt": {
                    "route": fl.route.ptr, "mask": fl.mask.ptr,
                    "slot": fl.slot.ptr, "expert": fl.expert.ptr,
                    "first": fl.first.ptr, "order": fl.order.ptr,
                },
            }  # fmt: skip
            c |= {
                r: sreg(scratch, slay[r][0], r)
                for r in FFN_REGIONS
                if r != "rlog" and r in slay
            }
            # this CTA's x / x8 / x4 columns are out
            if bid < NT:
                publish(mb.put, c["xrdy"], bid, fx.Int32(1), tid == 0)
            ffn.run_ffn(c, bid)
        stamp(tl_on, tls, tid, TL_POINTS - 1)
        stamp_flush(tl_on, tls, tl, tid, bid, TL_POINTS)

    @launcher(name)
    def launch(
        core: Int64, w_o: Int64, s_o: Int64, res: Int64, g_post: Int64,
        g_ahead: Int64, moe_w: Int64, h_mid: Int64, x: Int64, x8: Int64,
        x8s: Int64, la8: Int64, la8s: Int64, ffn_out: Int64, scratch: Int64,
        plain: Int64, nvalid: Int64,
        peers: Int64, rank: Int32, layer: Int32, tl: Int64,
        stream: fx.Stream = _STREAM,
    ):  # fmt: skip
        _ = keyed
        p1_layer(
            core, w_o, s_o, res, g_post, g_ahead, moe_w, h_mid, x, x8, x8s,
            la8, la8s, ffn_out, scratch, plain, nvalid, peers, rank, layer, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(p1_layer, launch)
    return launch
