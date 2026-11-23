# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The FFN stages of the phase-1 layer kernel (``upto = "ffn"``): router band,
route, expert union, routed experts (A16W4, W4A8 or W4A4,
``P1Build.routed_act``), MXFP8 shared expert and their combine into this
rank's FFN partial (stages and placement: ``layout``).

Ported from Kimi-K3's K2b MoE stage (``kimi_k3/mono/kernels/stages/moe.py``:
the router band with its parts, ``stage_route``, ``load_route``'s union, the
ug / down tasks over the union) with MiniMax-M3's math
(``minimax_m3/mono/kernels/stages``: SwiGLU-OAI, the a4w4 activation quant,
the MXFP8 shared expert's operands) at Pro's TP8 widths.

Numerics: the routed experts in the A16W4 recipe (``routed_act = "bf16"``,
the default: production's a16w4 numerics at these widths, aiter's
``gemm1_a16w4_port`` / ``gemm2_a16w4_port``: the bf16 x and bf16 mids, the
MXFP4 weight dequantized in the kernel to bf16 -- ``v_cvt_scalef32_pk_bf16_fp4``
of each e2m1 code times its E8M0 scale, exact -- and the bf16 MFMA 16x16x32
with fp32 accumulation; gfx950 has no MFMA of a bf16 and an fp4 operand),
the W4A4 one (``"fp4"``: aiter a4w4 numerics, MXFP4 activations, as
M3.1-Flash mono's routed stage) or the W4A8 one (``"fp8"``: the norm's MXFP8
x8, which the shared expert also reads, and MXFP8 mids, ``quantize_fp8``'s
rule; the gfx950 f8f6f4 MFMA with the MXFP4 weight as A, cbsz 4, and the
MXFP8 activation as B, blgp 0), and production's MXFP8 shared expert::

    logit  = bf16(x @ w_gate^T)                       (fp32 accumulate)
    s      = sigmoid(logit); picks = top-4 of s + bias (lower expert on a tie)
    w_k    = s_k / (s_0 + s_1 + s_2 + s_3) * 2.0      (rank order sum)
    xa     = x               (A16W4: the norm's bf16 x)
           | MXFP4(x)        (W4A4: E8M0 RoundUp of amax * f32(1/6), RNE e2m1)
           | x8              (W4A8: the norm's MXFP8 x)
    mid_k  = MXA(bf16(SwiGLU-OAI(xa @ w13_e^T)))      (fp32 gate / up; MXA:
                                                       none | MXFP4 | MXFP8)
    routed = bf16(sum_k bf16(w_k (mid_k @ w2_e^T)))   (fp32 sum, top-k order)
    gu     = bf16(x8 @ w_sgu^T); smid = MXFP8(bf16(SwiGLU-OAI(gu)))
    shared = bf16(smid @ w_sd^T)
    ffn    = bf16(routed + shared)

with SwiGLU-OAI(g, u) = min(g, 7) sigmoid(1.702 min(g, 7)) (clamp(u, -7, 7) +
1). The GEMVs sum K in another order than aiter's kernels, so a value can
differ in its last bit and an MXFP4 / MXFP8 code flip with it.

A16W4 against aiter's a16w4 kernels: the bf16 MFMA operands are theirs (lane
group g of a 128-k step takes k 32 g + 8 j .. + 8 in MFMA j, both operands;
``mx.fp4_bf16_tile``), and a gate / up value is four K quarters of 1792 (14
steps each, in order) summed (q0 + q1) + (q2 + q3) where gemm1 at k_wave 4
sums ((q0 + q1) + q2) + q3. The routed combine is this kernel's (fp32 sum of
the bf16 picks, top-k order, then bf16) where gemm2 adds the bf16 picks with
bf16 atomics in arrival order: more exact, not the same bits.

The routed experts' zero pad (intermediate 640 .. 767, ``layout.INTER_V``) is
skipped: its mids are exactly zero, whole MXFP4 blocks of their own.
"""

import math

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from aiter.ops.flydsl.kernels.act import swiglu_mul_batch
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from atom.models.minimax_m3.pro_mono.config import (
    EXPERTS,
    HIDDEN,
    ROUTE_SCALE,
    SWIGLU_LIMIT,
    TOPK,
)
from atom.models.minimax_m3.pro_mono.layout import (
    DIMS,
    DOWN_BATCH,
    DOWN_ROWS,
    DOWN_TASKS,
    E_RANK,
    INTER_R,
    INTER_S,
    MOE_WEIGHTS,
    NT,
    R0,
    R_KLEN,
    R_PARTS,
    R_ROWS,
    RT0,
    SD_STEPS,
    SH0,
    SH_CPW,
    SH_ROWS,
    SH_TASKS,
    SM_ROW,
    SMSC_WORDS,
    TP,
    UG_COLS,
    UG_GROUPS,
    UG_HALVES,
    W2_STEPS,
    W13_HALF,
    W13_Q,
    X4_ROW,
    X4_WORDS,
    X8_ROW,
    X8_WORDS,
    XH_ROW,
    XH_WORDS,
    XSC_WORDS,
    first_ug,
    inter_words,
    interb_word,
    mxfp8_tile_dword,
    rlog_pair,
    route_pair,
    rpart_pair,
    smid_pair,
    ug_split,
    ugp_pair,
    w2_scale_word,
    w13_scale_word,
    xh_word,
)
from atom.mono.device import gemv, route
from atom.mono.device.mx import (
    FP4,
    FP8,
    FP8_MAX,
    UNIT_SCALE,
    clamp_fp8,
    code_ceil,
    fp4_bf16_tile,
    fp4_pack2,
    fp4_scale,
    fp4_tile_load,
    mfma_scaled,
    rcp_pow2,
)
from atom.mono.device.ops import (
    CM_DEV,
    CM_NT,
    bf2_f32,
    bf16_pair,
    bf16_round,
    block_excl_scan,
    butterfly,
    fp8_pack4,
    hw_exp2,
    hw_rcp,
    ld_dev,
    ld_f32,
    load_ptr64,
    mfma_bf16,
    popcount,
    row_sum,
    rsrc,
    st_dev,
    traced,
    uniform,
    xshfl,
)
from atom.mono.device.select import (
    BATCH,
    count_beaten,
    entry_key,
    f32_key,
    wave_compact,
    wave_floor,
    wave_kth,
)
from atom.mono.device.sync import publish
from atom.mono.plan.execution import BLOCKS, THREADS, WAVES
from atom.mono.plan.trace import enter_stage

LOG2E = 1.0 / math.log(2.0)
W13_BYTES = DIMS.w13_bytes
W2_BYTES = DIMS.w2_bytes


def weight(c, name):
    """Operand ``name``'s address from the layer's ``moe_w`` table
    (``layout.MOE_WEIGHTS``), loaded where it is used: ten fewer kernel
    arguments, and no address held from the kernel's start."""
    return load_ptr64(c["moe_w"], MOE_WEIGHTS.index(name))


def _v4i():
    return fx.Vector.make_type(4, fx.Int32)


def _or(a, b):
    return a | b


def _f32_of(w):
    return fx.Int32(w).bitcast(fx.Float32)


def absmax(vals):
    am = fx.max(vals[0], -vals[0])
    for v in vals[1:]:
        am = fx.max(am, fx.max(v, -v))
    return am


def mx8_code(am):
    """``quantize_fp8``'s E8M0 code of a 32-block of absolute maximum ``am``:
    ceil_pow2(max(amax, 1e-10) * f32(1 / 448))."""
    return code_ceil(fx.max(am, fx.Float32(1e-10)) * fx.Float32(1.0 / FP8_MAX))


def swiglu(g, u):
    """SwiGLU-OAI of fp32 gate / up (aiter's: gate upper-clamped, up two-sided,
    alpha 1.702, up + 1)."""
    return swiglu_mul_batch([g], [u], fx.Float32(-SWIGLU_LIMIT))[0]


# ------------------------------------------------------------ x hand-off


@traced
def mx4_store(c, t, col, o):
    """The norm's MXFP4 1x32 of this lane's two bf16 outputs ``o`` (16 lanes a
    block; aiter's a4w4 quant: E8M0 RoundUp of amax * f32(1/6), RNE e2m1) ->
    ``x4`` (lane 4 m stores lanes 4 m .. 4 m + 3's bytes) and the task's 4
    codes in one ``x4s`` word, plain at device scope (``xrdy`` announces
    them)."""
    lane = c["lane"]
    code, scale = fp4_scale(butterfly(absmax(o), (1, 2, 4, 8), fx.max))
    byte = fp4_pack2(fx.Int32(0), o[0], o[1], scale, 0) & 0xFF
    word = butterfly(byte << (8 * (lane % 4)), (1, 2), _or)
    if lane % 4 == 0:
        st_dev(word, c["plain"], c["pw"]["x4"] + (t * HIDDEN + col) // 8)
    codes = butterfly(code << (8 * (lane // 16)), (16, 32), _or)
    if lane == 0:
        st_dev(codes, c["plain"], c["pw"]["x4s"] + t * XSC_WORDS + c["j"])


@traced
def wait_x(c):
    """Every norm CTA's flag: x, x8 and x4 of all its columns are stored."""
    if c["tid"] < NT:
        c["poll"]([(c["xrdy"], c["tid"], 1)])
    gpu.barrier()


# ------------------------------------------------------------ router


def router_loads(c, task):
    """Router task ``task``'s weights for this wave: rows 16 i .. of the
    rank's band, K slice q (``R_KLEN``), one 128-column chunk a wave."""
    i, q = task // R_PARTS, task % R_PARTS
    return gemv.loads(
        c["lane"],
        c["wave"],
        weight(c, "w_gate"),
        i * R_ROWS,
        R_KLEN,
        ldk=HIDDEN,
        k0=q * R_KLEN,
        paired=True,
    )


@traced
def stage_router(c, task, ops):
    """Router task (i, q) = (task / R_PARTS, task % R_PARTS): its rows against
    x's K slice q (bf16 MFMA, fp32). Parts q > 0 publish their fp32 partials;
    part 0 adds them in part order, rounds to bf16 (the gate's output dtype)
    and sends the logits to every rank's ``rlog``."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    i, q = task // R_PARTS, task % R_PARTS
    xrow = R_KLEN + gemv.LDS_PAD
    wait_x(c)
    gemv.rows_to_lds(
        tid, c["x"], R_KLEN, s, c["xl"], xrow, CM_DEV, ldk=HIDDEN, k0=q * R_KLEN
    )
    acc = gemv.mfmas(lane, wave, c["xl"], xrow, R_KLEN, s, ops)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    if tid < R_ROWS * s:
        r = tid % R_ROWS
        t = tid // R_ROWS
        v = row_sum(red, r, t)
        if q > 0:
            c["put"](c["rpart"], rpart_pair(task, t, r), v)
        if q == 0:
            got = c["poll"](
                [(c["rpart"], rpart_pair(task + p, t, r), 1) for p in range(1, R_PARTS)]
            )
            for p in range_constexpr(R_PARTS - 1):
                v = v + _f32_of(got[p][0])
            v = bf16_round(v)
            for p in range_constexpr(TP):
                c["put"](c["rlog"][p], rlog_pair(c["rank"], t, i * R_ROWS + r), v)
    gpu.barrier()


# ------------------------------------------------------------ route

# The route's words of ``red`` (one wave's): the picks' sigmoids and experts
# by rank. Its candidates: ``cand`` / ``csig`` in LDS.
R_SIG = 64
R_EXP = R_SIG + TOPK
# candidates, two a lane: past this many the route takes the radix path
ROUTE_CAP = 128
ROUTE_SLOTS = ROUTE_CAP + BATCH  # ``wave_compact``'s dead tail included


@traced
def _route_rank(cand, csig, n, lane, red):
    """Each of the n (<= ``ROUTE_CAP``) candidates' rank: how many beat it
    (``count_beaten``). The sort keys are distinct, so the ranks are; ranks
    0 .. 3 are the top 4, stored by rank: the sigmoid at red[R_SIG + r], the
    expert at red[R_EXP + r]."""
    qs = [lane, lane + 64]
    ranks = count_beaten(cand, [entry_key(cand, q) for q in qs], n)
    for i in range_constexpr(2):
        q = qs[i]
        if (q < n) & (ranks[i] < TOPK):
            expert = fx.Int32(EXPERTS - 1) - fx.ptr_load(cand + 2 * q)
            fx.ptr_store(fx.ptr_load(csig + q), red + (R_SIG + ranks[i]))
            fx.ptr_store(expert.bitcast(fx.Float32), red + (R_EXP + ranks[i]))


@traced
def stage_route(c, t):
    """Token t (wave 0): the 384 logits of every rank's ``rlog``, 6 a lane
    (expert lane + 64 i) -> sigmoid -> + bias -> the top 4 ranked by score,
    descending, equal scores the lower expert first -> the unbiased sigmoids
    renormalized in that order, x ``ROUTE_SCALE`` -> ``route``.

    The top 4 by counting (``device.select``'s one-wave form); past
    ``ROUTE_CAP`` candidates (many equal keys) a radix search, same order."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    if wave == 0:
        per = EXPERTS // 64
        specs = []
        for i in range_constexpr(per):
            e = lane + 64 * i
            specs.append((c["rlog_own"], rlog_pair(e // E_RANK, t, e % E_RANK), 1))
        # the bias does not depend on the logits: in flight under the poll
        bias = [ld_f32(weight(c, "bias"), lane + 64 * i) for i in range(per)]
        got = c["poll"](specs)
        keys, sigs = [], []
        for i in range_constexpr(per):
            lg = _f32_of(got[i][0])
            sg = hw_rcp(fx.Float32(1.0) + hw_exp2(lg * fx.Float32(-LOG2E)))
            sc = sg + bias[i]
            # a NaN logit (a pad row's garbage) ranks last
            live = sc == sc  # noqa: PLR0124
            keys.append(f32_key(live.select(sc, fx.Float32(-3.0e38))))
            sigs.append((sg == sg).select(sg, fx.Float32(0.0)))  # noqa: PLR0124
        # an entry's id: the lower expert the larger, so it wins a tie
        ids = [fx.Int32(EXPERTS - 1) - (lane + 64 * i) for i in range(per)]
        cand, csig = c["cand"], c["csig"]
        floor = wave_floor(keys, TOPK)
        n = wave_compact(
            cand,
            ROUTE_CAP,
            [(keys[i] >= floor, keys[i], ids[i]) for i in range(per)],
            lane,
            [(csig, sigs)],
        )
        if n <= ROUTE_CAP:
            _route_rank(cand, csig, n, lane, red)
        else:
            sel = wave_kth(keys, TOPK, lane)
            wave_compact(
                cand,
                ROUTE_CAP,
                [(sel[i], keys[i], ids[i]) for i in range(per)],
                lane,
                [(csig, sigs)],
            )
            _route_rank(cand, csig, fx.Int32(TOPK), lane, red)
        # the sigmoids in rank order, summed serially
        total = fx.Float32(0.0)
        for j in range_constexpr(TOPK):
            total = total + fx.ptr_load(red + (R_SIG + j))
        scale = fx.Float32(ROUTE_SCALE) / total
        if lane < TOPK:
            me = fx.ptr_load(red + (R_EXP + lane)).bitcast(fx.Int32)
            ms = fx.ptr_load(red + (R_SIG + lane))
            c["put"](
                c["route"],
                route_pair(t, lane),
                fx.min(fx.max(me, 0), fx.Int32(EXPERTS - 1)),
            )
            c["put"](c["route"], route_pair(t, TOPK + lane), ms * scale)
    gpu.barrier()


# ------------------------------------------------------------ the union


def passes(n):
    """A CTA's passes over n work items, a thread an item a pass."""
    return (n + THREADS - 1) // THREADS


def mask_words(s: int) -> int:
    """Words of an expert's token bitmap (``load_route``)."""
    return -(-s // 32)


def pick_expert(rt, p):
    """Pick p's (token p / 4, rank p % 4) expert from the route table."""
    t, k = p // TOPK, p % TOPK
    return fx.max(fx.min(fx.ptr_load(rt["route"] + route_pair(t, k)), EXPERTS - 1), 0)


@traced
def load_route(c):
    """``route`` -> LDS (ids, weights); the union U (ascending): expert[slot],
    slot[e], first[slot] (``first[|U|]`` = every pick) and the picks in slot
    order, a slot's tokens ascending (``order``: pick ids 4 t + k), the row
    order of ``inter``. Returns |U|.

    Every CTA derives the same order (ug writes a pick's mid row, down reads
    it): each pick ORs its token's bit into its expert's bitmap, so an
    expert's pick count is a popcount and a pick's rank in its slot the
    bitmap's tokens below its own, whatever order the ORs land in."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rt = c["rt"]
    mw = mask_words(s)
    n = s * 2 * TOPK
    for i in range_constexpr(passes(n)):
        j = tid + THREADS * i
        if j < n:
            v = c["poll"]([(c["route"], j, 1)])[0][0]
            fx.ptr_store(v, rt["route"] + j)
    for i in range_constexpr(passes(EXPERTS * mw)):
        j = tid + THREADS * i
        if j < EXPERTS * mw:
            fx.ptr_store(fx.Int32(0), rt["mask"] + j)
    gpu.barrier()
    for i in range_constexpr(passes(s * TOPK)):
        p = tid + THREADS * i
        if p < s * TOPK:
            route.mark_token(rt["mask"], mw, pick_expert(rt, p), p // TOPK)
    gpu.barrier()
    # two experts a thread: their pick counts; slots and first picks by scans
    ea, eb = fx.min(2 * tid, EXPERTS - 1), fx.min(2 * tid + 1, EXPERTS - 1)
    na, nb = fx.Int32(0), fx.Int32(0)
    for w in range_constexpr(mw):
        na = na + popcount(fx.ptr_load(rt["mask"] + (ea * mw + w)))
        nb = nb + popcount(fx.ptr_load(rt["mask"] + (eb * mw + w)))
    na = (2 * tid < EXPERTS).select(na, fx.Int32(0))
    nb = (2 * tid + 1 < EXPERTS).select(nb, fx.Int32(0))
    ua = (na > 0).select(fx.Int32(1), fx.Int32(0))
    ub = (nb > 0).select(fx.Int32(1), fx.Int32(0))
    off, tot = block_excl_scan(ua + ub, lane, wave, c["scan"])
    poff, ptot = block_excl_scan(na + nb, lane, wave, c["scan"])
    if ua == 1:
        fx.ptr_store(off, rt["slot"] + ea)
        fx.ptr_store(ea, rt["expert"] + off)
        fx.ptr_store(poff, rt["first"] + off)
    if ub == 1:
        fx.ptr_store(off + ua, rt["slot"] + eb)
        fx.ptr_store(eb, rt["expert"] + off + ua)
        fx.ptr_store(poff + na, rt["first"] + off + ua)
    if tid == THREADS - 1:
        fx.ptr_store(ptot, rt["first"] + tot)
    gpu.barrier()
    for i in range_constexpr(passes(s * TOPK)):
        p = tid + THREADS * i
        if p < s * TOPK:
            t = p // TOPK
            e = pick_expert(rt, p)
            # the tokens below t that picked e: the picks before p in its slot
            rank = route.tokens_below(rt["mask"], mw, e, t)
            slot = fx.ptr_load(rt["slot"] + e)
            fx.ptr_store(p, rt["order"] + (fx.ptr_load(rt["first"] + slot) + rank))
    gpu.barrier()
    return uniform(tot)


def fp8_routed(c) -> bool:
    """Whether the build's routed experts are W4A8 (a build-time constant)."""
    return c["routed_act"] == "fp8"


def bf16_routed(c) -> bool:
    """Whether the build's routed experts are A16W4 (a build-time constant)."""
    return c["routed_act"] == "bf16"


@traced
def load_xb(c):
    """The routed experts' B operands, every token's row -> LDS (``xbl`` /
    ``xsl``): the norm CTAs' MXFP4 x (W4A4: ``x4`` / ``x4s``, plain), MXFP8
    x8 (W4A8: ``x8`` / ``x8s``, the outputs) or the CTA's K half of the bf16
    x (A16W4: ``x``), at device scope behind ``xrdy``."""
    if const_expr(bf16_routed(c)):
        load_xh(c)
    elif const_expr(fp8_routed(c)):
        load_x8(c)
    else:
        load_x4(c)


@traced
def load_xh(c):
    """A16W4: K half ``ugh`` (the half of every ug task of this CTA) of each
    token's bf16 x (the norm's ``x``, stored at device scope) -> LDS rows of
    ``XH_ROW`` words. A whole x row (14 KB) would not fit 16 of them."""
    s, tid = c["S"], c["tid"]
    wait_x(c)
    k0 = c["ugh"] * XH_WORDS
    n = s * XH_WORDS // 4
    for i in range_constexpr(passes(n)):
        e = fx.min(tid + THREADS * i, n - 1)
        t = e // (XH_WORDS // 4)
        w = e % (XH_WORDS // 4) * 4
        v = fx.Vector(ld_dev(c["x"], t * (UG_HALVES * XH_WORDS) + k0 + w, 4))
        if tid + THREADS * i < n:
            fx.ptr_store(v, c["xbl"] + (t * XH_ROW + w))
    gpu.barrier()


@traced
def load_x8(c):
    """The norm's MXFP8 x (``x8`` / ``x8s``: stored at device scope, as the
    shared expert reads them) -> LDS rows of ``X8_ROW`` words, the codes
    ``XSC_WORDS`` a token."""
    s, tid = c["S"], c["tid"]
    wait_x(c)
    n = s * X8_WORDS // 4
    for i in range_constexpr(passes(n)):
        e = fx.min(tid + THREADS * i, n - 1)
        t = e // (X8_WORDS // 4)
        w = e % (X8_WORDS // 4) * 4
        v = fx.Vector(ld_dev(c["x8"], t * X8_WORDS + w, 4))
        if tid + THREADS * i < n:
            fx.ptr_store(v, c["xbl"] + (t * X8_ROW + w))
    m = s * XSC_WORDS
    for i in range_constexpr(passes(m)):
        e = fx.min(tid + THREADS * i, m - 1)
        v = fx.Int32(ld_dev(c["x8s"], e))
        if tid + THREADS * i < m:
            fx.ptr_store(v, c["xsl"] + e)
    gpu.barrier()


@traced
def load_x4(c):
    """The norm CTAs' MXFP4 x (``x4`` / ``x4s``, plain at device scope behind
    ``xrdy``) -> LDS, every token's row: the routed experts' B operands."""
    s, tid = c["S"], c["tid"]
    wait_x(c)
    n = s * X4_WORDS // 4
    for i in range_constexpr(passes(n)):
        e = fx.min(tid + THREADS * i, n - 1)
        t = e // (X4_WORDS // 4)
        w = e % (X4_WORDS // 4) * 4
        v = fx.Vector(ld_dev(c["plain"], c["pw"]["x4"] + t * X4_WORDS + w, 4))
        if tid + THREADS * i < n:
            fx.ptr_store(v, c["xbl"] + (t * X4_ROW + w))
    m = s * XSC_WORDS
    for i in range_constexpr(passes(m)):
        e = fx.min(tid + THREADS * i, m - 1)
        v = fx.Int32(ld_dev(c["plain"], c["pw"]["x4s"] + e))
        if tid + THREADS * i < m:
            fx.ptr_store(v, c["xsl"] + e)
    gpu.barrier()


# ------------------------------------------------------------ routed ug


def ug_row_group(pair, wave):
    """Wave ``wave``'s 16-row group of w13 in ug pair ``pair`` (slot pair /
    UG_GROUPS, 32 mid columns 32 g ..): waves w % 4 = 0, 1 the gate's two
    (rows 32 g .., + 16), 2, 3 the up's (rows INTER_R + 32 g ..); w / 4 picks
    the K half (A16W4: the K quarter of the task's half, ``ug_steps``)."""
    g = pair % UG_GROUPS
    # bit ops, not % 4: the o stage's wave % 4, merged with this one, was
    # held from there to here and spilled an SGPR
    rgl = wave & 3
    return (rgl >> 1) * (INTER_R // 16) + 2 * g + (rgl & 1)


def ug_steps(c):
    """This wave's 128-k steps of an ug task: W4A4 / W4A8 the K half wave / 4
    (``W13_HALF`` steps); A16W4 the quarter wave / 4 of the CTA's K half
    ``ugh`` (``W13_Q`` steps; wave w / 4 + 2 ugh sums K 1792 (w / 4 + 2 ugh)
    .. + 1792, gemm1_a16w4's k_wave 4 slice)."""
    q = c["wave"] >> 2
    if const_expr(bf16_routed(c)):
        base = c["ugh"] * W13_HALF + q * W13_Q
        return [base + i for i in range(W13_Q)]
    return [q * W13_HALF + i for i in range(W13_HALF)]


def ug_loads(c, pair):
    """Ug pair ``pair``'s w13 operands for this wave (its row group, its
    ``ug_steps``) and their scale words."""
    lane, wave, rt = c["lane"], c["wave"], c["rt"]
    slot = pair // UG_GROUPS
    e = uniform(fx.ptr_load(rt["expert"] + slot))
    rg = ug_row_group(pair, wave)
    r_w = rsrc(weight(c, "w13") + fx.Int64(e) * W13_BYTES)
    steps = ug_steps(c)
    av = [fp4_tile_load(r_w, 0, rg, HIDDEN, st, lane * 4) for st in steps]
    r_s = rsrc(weight(c, "s13"))
    sw = [
        fx.Int32(
            bo.buffer_load(
                r_s, w13_scale_word(e, rg, st, lane), vec_width=1, dtype=T.i32
            )
        )
        for st in steps[::2]
    ]
    return av + sw


def x8_lds_operand(c, t, st, kb):
    """Token t's MXFP8 x from LDS as the B operand of 128-k step ``st``: its
    k 16 kb .. + 16 and 64 + 16 kb .. + 16 (the f8f6f4 MFMA's fp8 layout)."""
    base = c["xbl"] + (t * X8_ROW + st * 32 + kb * 4)
    hv = [fx.Vector(fx.ptr_load(base + 16 * h, result_type=_v4i())) for h in range(2)]
    return fx.Vector.from_elements(
        [hv[h][e] for h in range(2) for e in range(4)], fx.Int32
    )


def ug_mx_step(c, av, sw, acc, i, st, rg, t, kb):
    """W4A4 / W4A8: step i (128-k step ``st``) of ``ug_tile``: the MXFP4 tile
    against token t's MXFP4 / MXFP8 x row in LDS, one scaled MFMA."""
    fp8 = fp8_routed(c)
    if const_expr(fp8):
        bw = x8_lds_operand(c, t, st, kb)
    else:
        bw = fx.Vector(
            fx.ptr_load(c["xbl"] + (t * X4_ROW + st * 16 + kb * 4), result_type=_v4i())
        )
    sb = (fx.ptr_load(c["xsl"] + (t * XSC_WORDS + st)) >> (kb * 8)) & 0xFF
    # byte (st % 2) 2 + rg % 2 of the word; st % 2 is i % 2 (half W13_HALF
    # is even)
    sa = (sw[i // 2] >> ((i % 2) * 16 + (rg % 2) * 8)) & 0xFF
    return mfma_scaled(av[i], bw, acc, sa, sb, FP4, FP8 if fp8 else FP4)


@traced
def ug_tile(c, pair, ops, first, count):
    """Ug pair ``pair`` (``ops``: ``ug_loads``) against its slot's picks at
    first .. first + count (S <= 16: one tile), a pick an MFMA column, its
    token's x row (MXFP4, MXFP8, or A16W4 the CTA's K half of the bf16 x) from
    LDS; then each pick's 32 mids -> bf16 -> MXFP4 or MXFP8 (``inter`` /
    ``isc``) or bf16 (``inter``). An A16W4 task holds one K half: the half-1
    task sends its gate / up partials (``ugp``), the half-0 task adds them
    to its own and stores the mids."""
    tid, lane, wave, red, rt = c["tid"], c["lane"], c["wave"], c["red"], c["rt"]
    bf16 = bf16_routed(c)
    g = pair % UG_GROUPS
    rg = ug_row_group(pair, wave)
    steps = ug_steps(c)
    av = ops[: len(steps)]
    sw = ops[len(steps) :]
    t = fx.ptr_load(rt["order"] + first + fx.min(lane % 16, count - 1)) // TOPK
    kb = lane // 16
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(len(steps)):
        st = steps[i]
        if const_expr(bf16):
            # byte (st % 2) 2 + rg % 2 of the word; st % 2 is i % 2 (a
            # quarter's first step is even)
            sa = (sw[i // 2] >> ((i % 2) * 16 + (rg % 2) * 8)) & 0xFF
            wa = fp4_bf16_tile(av[i], sa)
            # k 32 kb + 8 j .. + 8 of the step, in the K half's row
            local = (wave >> 2) * W13_Q + i
            for j in range_constexpr(4):
                xb = fx.Vector(
                    fx.ptr_load(c["xbl"] + xh_word(t, local, kb, j), result_type=_v4i())
                )
                acc = mfma_bf16(wa[j], xb.bitcast(fx.BFloat16), acc)
        else:
            acc = ug_mx_step(c, av, sw, acc, i, st, rg, t, kb)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    # thread (pick column tt, mid column col of 32): g / u summed over the K
    # halves (A16W4: quarters) in wave order
    tt = tid // UG_COLS
    col = tid % UG_COLS
    live = tt < count
    tc = fx.min(tt, count - 1)
    r = col % 16
    lo = col < 16
    gv = lo.select(row_sum(red, r, tc, waves=(0, 4)), row_sum(red, r, tc, waves=(1, 5)))
    uv = lo.select(row_sum(red, r, tc, waves=(2, 6)), row_sum(red, r, tc, waves=(3, 7)))
    row = first + tt
    if const_expr(bf16):
        ug_bf16_mids(c, g, row, col, live, gv, uv)
    else:
        ug_mx_mids(c, g, row, col, live, gv, uv)


@traced
def ug_mx_mids(c, g, row, col, live, gv, uv):
    """W4A4 / W4A8: the gate / up of (pick row ``row``, mid column 32 g +
    ``col``) -> SwiGLU-OAI -> bf16 -> the 32 mids' MXFP4 or MXFP8 ->
    ``inter`` / ``isc``."""
    fp8 = fp8_routed(c)
    y = bf16_round(swiglu(gv, uv))
    am = butterfly(fx.max(y, -y), (16, 8, 4, 2, 1), fx.max)
    iw = inter_words(c["routed_act"])
    if const_expr(fp8):
        # MXFP8 of the 32 mids (``quantize_fp8``'s rule, as the shared mid):
        # column 4 m packs columns 4 m .. 4 m + 3, a word
        code = mx8_code(am)
        q = clamp_fp8(y * rcp_pow2(code))
        qn = xshfl(q, 1)
        word = fp8_pack4(q, qn, xshfl(q, 2), xshfl(qn, 2))
        if live & (col % 4 == 0):
            st_dev(word, c["plain"], c["pw"]["inter"] + row * iw + g * 8 + col // 4)
    else:
        # MXFP4 of the 32 mids (32 lanes): an even column packs its pair's
        # byte, 4 bytes (8 columns) a word
        code, scale = fp4_scale(am)
        byte = fp4_pack2(fx.Int32(0), y, xshfl(y, 1), scale, 0) & 0xFF
        mine = (col % 2 == 0).select(byte << (4 * (col % 8)), fx.Int32(0))
        word = butterfly(mine, (1, 2, 4), _or)
        if live & (col % 8 == 0):
            st_dev(word, c["plain"], c["pw"]["inter"] + row * iw + g * 4 + col // 8)
    if live & (col == 0):
        st_dev(code, c["plain"], c["pw"]["isc"] + row * UG_GROUPS + g)


@traced
def ug_bf16_mids(c, g, row, col, live, gv, uv):
    """A16W4: this K half's gate / up of (pick row ``row``, mid column 32 g +
    ``col``). Half 1 sends them (``ugp``, one 16 B put a cell); half 0 adds
    them to its own -- (q0 + q1) + (q2 + q3) -- -> SwiGLU-OAI -> the bf16 mid
    -> ``inter`` (an even column stores its pair's word)."""
    h = c["ugh"]
    at = ugp_pair(row, g, col)
    # half 1 never waits, so a half-0 task's wait always ends (its pair's
    # half-1 task is another CTA's, which no ug task of its own holds up)
    if live & (h == 1):
        c["put_words"](c["ugp"], at, [gv.bitcast(fx.Int32), uv.bitcast(fx.Int32)])
    if live & (h == 0):
        got = c["poll"]([(c["ugp"], at, 2)])[0]
        y = bf16_round(swiglu(gv + _f32_of(got[0]), uv + _f32_of(got[1])))
        # both lanes of a pair are live (one pick row)
        word = bf16_pair(y, xshfl(y, 1)).bitcast(fx.Int32)
        if col % 2 == 0:
            st_dev(word, c["plain"], c["pw"]["inter"] + interb_word(row, g, col))


@traced
def run_ug(c, bid, nu):
    """Every ug task of the step's union, a CTA every BLOCKS-th from
    ``first_ug(bid)``; a pair's ``ugf`` flag past its mid stores. A16W4: task
    2 p + h is pair p's K half h (every task of a CTA one half, ``ugh``), the
    half-0 task publishes the flag."""
    rt = c["rt"]
    split = ug_split(c["routed_act"])
    total = nu * (UG_GROUPS * split)
    task = first_ug(bid)
    while task < total:
        pair = task >> 1 if split == 2 else task
        ops = ug_loads(c, pair)
        slot = pair // UG_GROUPS
        first = fx.ptr_load(rt["first"] + slot)
        count = uniform(fx.ptr_load(rt["first"] + slot + 1) - first)
        ug_tile(c, pair, ops, first, count)
        who = c["tid"] == 0
        if const_expr(split == 2):
            who = who & (c["ugh"] == 0)
        publish(c["put"], c["ugf"], pair, fx.Int32(1), who)
        task = task + BLOCKS


# ------------------------------------------------------------ shared


def fp8_tile_load(r_w, row, cols, kc, g4):
    """A lane's 32 B of an MXFP8 weight in aiter's (16, 16) preshuffle
    ([N / 16][K / 32][2][16][16] bytes), row ``row`` at 128-k chunk ``kc``: k
    16 g .. + 16 and 64 + 16 g .. + 16 (g = lane / 16), the f8f6f4 MFMA's
    operand layout (MiniMax-M3's ``fp8_tile_load``)."""
    dw = mxfp8_tile_dword(row, cols, kc, g4)
    halves = [
        fx.Vector(
            bo.buffer_load(
                r_w, dw + 256 * h, vec_width=4, dtype=T.i32, cache_modifier=CM_NT
            )
        )
        for h in range(2)
    ]
    return fx.Vector.from_elements(
        [halves[h][e] for h in range(2) for e in range(4)], fx.Int32
    )


def fp8_scale_load(r_s, row, cols, kc, g4):
    """The row-major E8M0 dword of row ``row``'s 128-k chunk ``kc``, shifted to
    this lane's block (its low byte, as the MFMA takes it)."""
    w = fx.Int32(
        bo.buffer_load(r_s, row * (cols // 128) + kc, vec_width=1, dtype=T.i32)
    )
    return w >> (g4 * 8)


def sh_loads(c, j):
    """The shared gate_up's half task j: gate rows 8 j .. + 8 on lanes l16 < 8,
    the up rows alike on the rest -- one MFMA tile; the waves split K."""
    lane, wave = c["lane"], c["wave"]
    l16, g4 = lane % 16, lane // 16
    row = (l16 < 8).select(SH_ROWS * j + l16, INTER_S + SH_ROWS * j + l16 - 8)
    r_w, r_s = rsrc(weight(c, "w_sgu")), rsrc(weight(c, "s_sgu"))
    kcs = [wave * SH_CPW + i for i in range(SH_CPW)]
    return (
        [fp8_tile_load(r_w, row, HIDDEN, kc, g4) for kc in kcs],
        [fp8_scale_load(r_s, row, HIDDEN, kc, g4) for kc in kcs],
    )


def x8_operand(c, t, kc, g4):
    """Token t's MXFP8 x (the norm's ``x8`` / ``x8s``, device scope) as the B
    operand of 128-k chunk kc, and its scale byte."""
    w0 = (t * HIDDEN + kc * 128 + 16 * g4) // 4
    halves = [fx.Vector(ld_dev(c["x8"], w0 + 16 * h, 4)) for h in range(2)]
    b = fx.Vector.from_elements(
        [halves[h][e] for h in range(2) for e in range(4)], fx.Int32
    )
    sb = fx.Int32(ld_dev(c["x8s"], t * XSC_WORDS + kc)) >> (g4 * 8)
    return b, sb


@traced
def stage_shared(c, j, ops):
    """Half task j: the GEMV against every token's x8 (B column l % 16 =
    token) -> bf16 gate / up (production's gate_up output) -> SwiGLU-OAI ->
    bf16 -> ``smid`` columns 8 j .. + 8 of every token."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    wts, scs = ops
    g4 = lane // 16
    col = fx.min(lane % 16, s - 1)
    wait_x(c)
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i in range_constexpr(SH_CPW):
        kc = wave * SH_CPW + i
        b, sb = x8_operand(c, col, kc, g4)
        acc = mfma_scaled(wts[i], b, acc, scs[i], sb, FP8, FP8)
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    gpu.barrier()
    # mid columns 2 rp, 2 rp + 1 of token tk: gate in tile rows 2 rp .., up in
    # 8 + 2 rp ..
    if tid < (SH_ROWS // 2) * s:
        rp = tid % (SH_ROWS // 2)
        tk = tid // (SH_ROWS // 2)
        y = []
        for h in range_constexpr(2):
            gs = bf16_round(row_sum(red, 2 * rp + h, tk))
            us = bf16_round(row_sum(red, SH_ROWS + 2 * rp + h, tk))
            y.append(bf16_round(swiglu(gs, us)))
        c["put_bf"](c["smid"], 2 * smid_pair(tk, SH_ROWS * j + 2 * rp), y)
    gpu.barrier()


@traced
def load_smid(c):
    """Every token's shared mid (``smid``) -> MXFP8 1x32 (``quantize_fp8``'s
    recipe, as ``swiglu_oai_split_mxfp8`` writes it) in LDS: the shared
    down's B operands. A thread a (token, 32-block)."""
    s, tid = c["S"], c["tid"]
    nblk = INTER_S // 32
    n = s * nblk
    b = fx.min(tid, n - 1)
    t = b // nblk
    k = b % nblk
    got = c["poll"]([(c["smid"], smid_pair(t, 32 * k + 4 * h), 2) for h in range(8)])
    vals = []
    for h in range_constexpr(8):
        for w in got[h]:
            vals += list(bf2_f32(w))
    code = mx8_code(absmax(vals))
    inv = rcp_pow2(code)
    q = [clamp_fp8(v * inv) for v in vals]
    words = [
        fp8_pack4(q[4 * i], q[4 * i + 1], q[4 * i + 2], q[4 * i + 3]) for i in range(8)
    ]
    # a 128-column chunk's 4 blocks are 4 adjacent lanes (20 blocks a token)
    codes = butterfly(code << (8 * (k % 4)), (1, 2), _or)
    if tid < n:
        for i in range_constexpr(8):
            fx.ptr_store(words[i], c["sm8"] + (t * SM_ROW + 8 * k + i))
        if k % 4 == 0:
            fx.ptr_store(codes, c["smsc"] + (t * SMSC_WORDS + k // 4))
    gpu.barrier()


# ------------------------------------------------------------ down


def w2_scale_words(c, e, rgd):
    """The dwords of w2's E8M0 codes (row e 7168 + 16 rgd + lane % 16, K
    block 4 st + lane / 16) for the K steps: steps 2 m and 2 m + 1 share one
    (``scale_index``'s d bit). Raw: ``w2_code`` picks the byte at its use."""
    lane = c["lane"]
    # rgd >= 0: a shift, not the floor division's sign fix-up
    return [
        fx.Int32(
            bo.buffer_load(
                rsrc(weight(c, "s2")),
                w2_scale_word(e, rgd >> 1, m, lane),
                vec_width=1,
                dtype=T.i32,
            )
        )
        for m in range(-(-W2_STEPS // 2))
    ]


def w2_code(words, rgd, st):
    """Step st's code of ``w2_scale_words``: byte 2 (st % 2) + rgd % 2."""
    return (words[st // 2] >> ((st % 2) * 16 + (rgd & 1) * 8)) & 0xFF


N_W2_WORDS = -(-W2_STEPS // 2)


def down_batch(c) -> int:
    """A wave's slots whose w2 is in flight together: ``DOWN_BATCH``; A16W4
    one (a slot's bf16 B operands are 80 VGPRs, four times W4A4's)."""
    return 1 if bf16_routed(c) else DOWN_BATCH


def down_loads(c, rgd, nu, b0):
    """The w2 tiles and scale words of a wave's batch from slot b0 (slots b0,
    b0 + 8, ..; past |U| the last slot's): flat, ``stage_down`` unpacks."""
    lane, rt = c["lane"], c["rt"]
    nbat = down_batch(c)
    slots = [fx.min(b0 + WAVES * j, nu - 1) for j in range(nbat)]
    es = [uniform(fx.ptr_load(rt["expert"] + sl)) for sl in slots]
    av = [
        fp4_tile_load(
            rsrc(weight(c, "w2") + fx.Int64(es[j]) * W2_BYTES),
            0,
            rgd,
            INTER_R,
            st,
            lane * 4,
        )
        for j in range(nbat)
        for st in range(W2_STEPS)
    ]
    sw = [w for j in range(nbat) for w in w2_scale_words(c, es[j], rgd)]
    return av + sw


def inter_operands(c, row, live):
    """A lane's B operands of mid row ``row`` (its K steps' words at lane group
    lane / 16 -- MXFP4: k 32 kb .. + 32; MXFP8: k 16 kb .. + 16 and 64 + 16 kb
    .. + 16 -- and their codes; bf16: k 32 kb + 8 j .. + 8, the 4 operands of
    bf16 MFMAs j = 0 .. 3, no code), device scope (the slot's ug flags polled
    first); a dead column multiplies zero."""
    kb = c["lane"] // 16
    iw = inter_words(c["routed_act"])
    base = c["pw"]["inter"] + row * iw
    out = []
    if const_expr(bf16_routed(c)):
        for st in range_constexpr(W2_STEPS):
            ops = []
            for j in range_constexpr(4):
                at = c["pw"]["inter"] + interb_word(row, st * 4 + kb, 8 * j)
                v = fx.Vector(ld_dev(c["plain"], at, 4))
                ops.append(
                    fx.Vector.from_elements(
                        [live.select(v[d], fx.Int32(0)) for d in range(4)], fx.Int32
                    )
                )
            out.append((ops, None))
        return out
    for st in range_constexpr(W2_STEPS):
        if const_expr(fp8_routed(c)):
            at = base + st * 32 + kb * 4
            hv = [fx.Vector(ld_dev(c["plain"], at + 16 * h, 4)) for h in range(2)]
            words = [hv[h][e] for h in range(2) for e in range(4)]
        else:
            bw = fx.Vector(ld_dev(c["plain"], base + st * 16 + kb * 4, 4))
            words = [bw[d] for d in range(4)]
        code = fx.Int32(
            ld_dev(c["plain"], c["pw"]["isc"] + row * UG_GROUPS + st * 4 + kb)
        )
        bw = fx.Vector.from_elements(
            [live.select(w, fx.Int32(0)) for w in words], fx.Int32
        )
        out.append((bw, live.select(code, fx.Int32(UNIT_SCALE))))
    return out


def tile_row(c, first, count):
    """A lane's pick row of the tile at first (``count`` picks) and whether its
    column is live; a dead column (past the picks, or a slot past U: count 0)
    reads row first."""
    col = c["lane"] % 16
    return first + fx.max(fx.min(col, count - 1), 0), col < count


@traced
def down_tile(c, rgd, ops, j, first, count, bs):
    """Batch slot j's w2 rows 16 rgd .. (``ops``) against its picks (their B
    operands ``bs``), a pick a column -> bf16(w acc) of each pick into LDS
    ``contrib`` (lane: rows 4 (lane / 16) + i of pick column lane % 16)."""
    lane, rt = c["lane"], c["rt"]
    nbs = down_batch(c) * W2_STEPS
    row, live = tile_row(c, first, count)
    words = ops[nbs + N_W2_WORDS * j : nbs + N_W2_WORDS * (j + 1)]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    b_fmt = FP8 if fp8_routed(c) else FP4
    for st in range_constexpr(W2_STEPS):
        if const_expr(bf16_routed(c)):
            # A16W4: the w2 tile dequantized to bf16, 4 bf16 MFMAs a step
            wa = fp4_bf16_tile(ops[j * W2_STEPS + st], w2_code(words, rgd, st))
            for q in range_constexpr(4):
                acc = mfma_bf16(wa[q], bs[st][0][q].bitcast(fx.BFloat16), acc)
        else:
            acc = mfma_scaled(
                ops[j * W2_STEPS + st], bs[st][0], acc, w2_code(words, rgd, st),
                bs[st][1], FP4, b_fmt,
            )  # fmt: skip
    if live:
        pid = fx.ptr_load(rt["order"] + row)
        t, k = pid // TOPK, pid % TOPK
        wt = fx.ptr_load(rt["route"] + route_pair(t, TOPK + k)).bitcast(fx.Float32)
        for i in range_constexpr(4):
            fx.ptr_store(
                bf16_round(acc[i] * wt),
                c["contrib"] + (pid * DOWN_ROWS + 4 * (lane // 16) + i),
            )


def sdown_loads(c, rgd):
    """The shared down's rows 16 rgd .., K chunk min(wave, SD_STEPS - 1)."""
    lane, wave = c["lane"], c["wave"]
    kc = fx.min(wave, SD_STEPS - 1)
    row = rgd * DOWN_ROWS + lane % 16
    return (
        fp8_tile_load(rsrc(weight(c, "w_sd")), row, INTER_S, kc, lane // 16),
        fp8_scale_load(rsrc(weight(c, "s_sd")), row, INTER_S, kc, lane // 16),
    )


@traced
def stage_down(c, rgd, nu, ops0, sops):
    """Hidden rows 16 rgd ..: the shared down (waves < SD_STEPS a K chunk each,
    MXFP8 of ``smid`` from LDS), then every slot's w2 against its picks (wave
    w the slots w, w + 8, ..; a batch of ``down_batch`` in flight, the next issued
    under this one's MFMAs; ``ops0`` the first, issued before any ug flag was
    waited for) -> bf16(w acc) a pick in LDS -> the routed sum in top-k order
    (fp32, then bf16) + bf16 shared -> bf16 ``ffn``."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt = c["rt"]
    nbat = down_batch(c)
    g4 = lane // 16
    # every wave (a branch on the wave left its mask live from the kernel's
    # start); waves past SD_STEPS repeat the last chunk, left out of the sum
    kc = fx.min(wave, SD_STEPS - 1)
    tk = fx.min(lane % 16, s - 1)
    base = c["sm8"] + (tk * SM_ROW + kc * 32 + 4 * g4)
    hv = [fx.Vector(fx.ptr_load(base + 16 * j, result_type=_v4i())) for j in range(2)]
    b = fx.Vector.from_elements(
        [hv[j][e] for j in range(2) for e in range(4)], fx.Int32
    )
    sb = fx.ptr_load(c["smsc"] + (tk * SMSC_WORDS + kc)) >> (g4 * 8)
    acc = mfma_scaled(
        sops[0], b, fx.Vector.filled(4, 0.0, fx.Float32), sops[1], sb, FP8, FP8
    )
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
    b0 = wave
    ops = ops0
    while b0 < nu:
        nb = b0 + WAVES * nbat
        slots = [fx.min(b0 + WAVES * j, nu - 1) for j in range(nbat)]
        # the batch's ug flags, a lane each (nbat x UG_GROUPS of them)
        nf = nbat * UG_GROUPS
        f = fx.min(lane, nf - 1)
        mine = slots[0]
        for j in range_constexpr(1, nbat):
            mine = (f // UG_GROUPS == j).select(slots[j], mine)
        c["poll"]([(c["ugf"], mine * UG_GROUPS + f % UG_GROUPS, 1)])
        firsts, counts, bss = [], [], []
        for j in range_constexpr(nbat):
            first = fx.ptr_load(rt["first"] + slots[j])
            count = fx.ptr_load(rt["first"] + slots[j] + 1) - first
            firsts.append(first)
            # a slot past U repeats the last: its picks are written once
            counts.append(((b0 + WAVES * j) < nu).select(count, fx.Int32(0)))
        # every slot's B operands, then the next batch's weights: a B load
        # issued after them waited for them (vmcnt counts in order)
        for j in range_constexpr(nbat):
            row, live = tile_row(c, firsts[j], counts[j])
            bss.append(inter_operands(c, row, live))
        ops_n = down_loads(c, rgd, nu, fx.min(nb, nu - 1))
        for j in range_constexpr(nbat):
            down_tile(c, rgd, ops, j, firsts[j], counts[j], bss[j])
        b0 = nb
        ops = ops_n
    gpu.barrier()
    if tid < (DOWN_ROWS // 2) * s:
        rp = tid % (DOWN_ROWS // 2)
        tt = tid // (DOWN_ROWS // 2)
        out = []
        for h in range_constexpr(2):
            r = 2 * rp + h
            y = fx.ptr_load(c["contrib"] + (tt * TOPK * DOWN_ROWS + r))
            for k in range_constexpr(1, TOPK):
                y = y + fx.ptr_load(c["contrib"] + ((tt * TOPK + k) * DOWN_ROWS + r))
            sh = bf16_round(row_sum(red, r, tt, waves=range(SD_STEPS)))
            out.append(bf16_round(y) + sh)
        bo.buffer_store(
            bf16_pair(out[0], out[1]).bitcast(fx.Int32),
            rsrc(c["ffn"]),
            (tt * HIDDEN + rgd * DOWN_ROWS + 2 * rp) // 2,
        )
    gpu.barrier()


# ------------------------------------------------------------ the FFN


@traced
def run_ffn(c, bid):
    """Every FFN stage of this CTA, in program order (``layout``'s
    placement): router / shared / route tasks, the union, ug, down."""
    s = c["S"]
    enter_stage("router")
    rt_task = bid - R0
    if bid >= R0:
        stage_router(c, rt_task, router_loads(c, rt_task))
    enter_stage("shared")
    sj = bid - SH0
    if (bid >= SH0) & (sj < SH_TASKS):
        stage_shared(c, sj, sh_loads(c, sj))
    enter_stage("route")
    rt_tok = bid - RT0
    if (bid >= RT0) & (rt_tok < s):
        stage_route(c, rt_tok)
    c["stamp"](6)
    enter_stage("union")
    if const_expr(bf16_routed(c)):
        # A16W4: the K half of every ug task of this CTA (tasks BLOCKS apart)
        c["ugh"] = first_ug(bid) & 1
    nu = load_route(c)
    load_xb(c)
    c["stamp"](7)
    enter_stage("ug")
    run_ug(c, bid, nu)
    c["stamp"](8)
    enter_stage("down")
    load_smid(c)
    for i in range_constexpr(-(-DOWN_TASKS // BLOCKS)):
        rgd = bid + BLOCKS * i
        if rgd < DOWN_TASKS:
            sops = sdown_loads(c, rgd)
            ops0 = down_loads(c, rgd, nu, fx.min(c["wave"], nu - 1))
            stage_down(c, rgd, nu, ops0, sops)
    c["stamp"](9)
