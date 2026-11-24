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
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
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
    MOE_TABLE,
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
    inter_words,
    interb_word,
    list_ctas,
    list_down,
    max_u,
    mxfp8_tile_dword,
    queued,
    rlog_pair,
    route_pair,
    rpart_pair,
    smid_pair,
    task_list,
    ugp_pair,
    w2_scale_word,
    w13_scale_word,
    xh_word,
)
from atom.mono.device import gemv, queue, route
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
    (``layout.MOE_TABLE``: the weights, then the queue word), loaded where it
    is used: eleven fewer kernel arguments, and no address held from the
    kernel's start."""
    return load_ptr64(c["moe_w"], MOE_TABLE.index(name))


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
    """``route`` -> LDS (ids, weights), the real rows' (``real_rows``: a
    graph's padding rows enlarge no union); the union U (ascending): expert[slot],
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
    nv = c["nv"]
    # the real rows' picks only: a padding row's never come (``stage_route``)
    n = s * 2 * TOPK
    for i in range_constexpr(passes(n)):
        j = tid + THREADS * i
        if j < nv * (2 * TOPK):
            v = c["poll"]([(c["route"], j, 1)])[0][0]
            fx.ptr_store(v, rt["route"] + j)
    for i in range_constexpr(passes(EXPERTS * mw)):
        j = tid + THREADS * i
        if j < EXPERTS * mw:
            fx.ptr_store(fx.Int32(0), rt["mask"] + j)
    gpu.barrier()
    for i in range_constexpr(passes(s * TOPK)):
        p = tid + THREADS * i
        if p < nv * TOPK:
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
        if p < nv * TOPK:
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


def ug_row_group(task, wave):
    """Wave ``wave``'s 16-row group of w13 in ug task ``task`` (an ug pair:
    slot task / UG_GROUPS, 32 mid columns 32 g ..): waves w % 4 = 0, 1 the
    gate's two (rows 32 g .., + 16), 2, 3 the up's (rows INTER_R + 32 g ..);
    w / 4 picks the K slice (``ug_kslice``)."""
    g = task % UG_GROUPS
    # bit ops, not % 4: the o stage's wave % 4, merged with this one, was
    # held from there to here and spilled an SGPR
    rgl = wave & 3
    return (rgl >> 1) * (INTER_R // 16) + 2 * g + (rgl & 1)


def ug_steps(c) -> int:
    """A wave's 128-k steps of an ug task: W4A4 / W4A8 a K half
    (``W13_HALF``), A16W4 a quarter (``W13_Q``: the task is one K half)."""
    return W13_Q if bf16_routed(c) else W13_HALF


def ug_kslice(c):
    """This wave's first 128-k step of an ug task: W4A4 / W4A8 the K half
    wave / 4; A16W4 the quarter wave / 4 of the CTA's K half ``ugh`` (wave w
    sums K 1792 (w / 4 + 2 ugh) .. + 1792, gemm1_a16w4's k_wave 4 slice).
    Its steps: ``ug_steps``."""
    q = uniform(c["wave"] >> 2)
    if const_expr(bf16_routed(c)):
        return c["ugh"] * W13_HALF + q * W13_Q
    return q * W13_HALF


# a task's w13 operands come in parts of its wave's K slice: the next task's
# part p is issued once this task's part p is used, so the next task's
# weights stream in behind this one's. A part's x operands (LDS) are read
# ahead of its MFMAs, all of them where the scheduler can: W4A8's 8 dwords a
# step held 14 steps' at once (112 VGPRs, the build spilled), so its first
# part, read under every weight of the task, is shorter; A16W4 reads 16
# dwords a step (4 bf16 operands) and dequantizes each tile to 16 more, so
# its 14 steps come in three. Every part even (a scale word holds two steps'
# codes).
UG_PARTS = {"fp4": (14, 14), "fp8": (8, 10, 10), "bf16": (4, 4, 6)}
S13_BYTES = EXPERTS * DIMS.w13_scale_bytes
assert all(not any(n % 2 for n in p) for p in UG_PARTS.values())
assert sum(UG_PARTS["bf16"]) == W13_Q
assert sum(UG_PARTS["fp4"]) == sum(UG_PARTS["fp8"]) == W13_HALF


def ug_parts(c):
    """The build's parts (``UG_PARTS``): (first step in the K slice, steps)."""
    sizes = UG_PARTS[c["routed_act"]]
    return [(sum(sizes[:p]), n) for p, n in enumerate(sizes)]


def ug_loads(c, task, part, live):
    """Part ``part`` (``ug_parts``) of task ``task``'s w13 operands for this
    wave (its row group, its K slice) and their scale words. ``live`` false:
    issued against an empty buffer range (zeros, no memory traffic), so a
    loop's last prefetch costs nothing."""
    lane, wave, rt = c["lane"], c["wave"], c["rt"]
    i0, n = ug_parts(c)[part]
    slot = task // UG_GROUPS
    e = uniform(fx.ptr_load(rt["expert"] + slot))
    rg = uniform(ug_row_group(task, wave))
    s0 = ug_kslice(c) + i0
    # the row group's first step of the part in the descriptor's base, a
    # step's 1 KB in soffset: one address VGPR for every tile (a VGPR offset
    # a step was hoisted out of the loop, 28 of them)
    r_w = rsrc(
        c["w13"] + fx.Int64(e) * W13_BYTES + fx.Int64(rg * (2 * HIDDEN) + s0 * 256) * 4,
        live.select(fx.Int32(W13_BYTES), fx.Int32(0)),
    )
    # the scale words first: a step's MFMA then waits for its own tile, not
    # for the part's last one (``vmcnt`` counts in issue order)
    r_s = rsrc(c["s13"], live.select(fx.Int32(S13_BYTES), fx.Int32(0)))
    sw = [
        fx.Int32(
            bo.buffer_load(
                r_s, w13_scale_word(e, rg, s0 + i, lane), vec_width=1, dtype=T.i32
            )
        )
        for i in range(0, n, 2)
    ]
    av = [
        fx.Vector(
            bo.buffer_load(
                r_w, lane * 4, vec_width=4, dtype=T.i32, cache_modifier=CM_NT,
                soffset_bytes=i * 1024,
            )
        )
        for i in range(n)
    ]  # fmt: skip
    # issued here, in this order: the scheduler would sink a part's scale
    # loads past the next part's tiles, and the first MFMA would wait for all
    rocdl.sched_barrier(0)
    return av + sw


def x8_lds_operand(c, t, st, kb):
    """Token t's MXFP8 x from LDS as the B operand of 128-k step ``st``: its
    k 16 kb .. + 16 and 64 + 16 kb .. + 16 (the f8f6f4 MFMA's fp8 layout)."""
    base = c["xbl"] + (t * X8_ROW + st * 32 + kb * 4)
    hv = [fx.Vector(fx.ptr_load(base + 16 * h, result_type=_v4i())) for h in range(2)]
    return fx.Vector.from_elements(
        [hv[h][e] for h in range(2) for e in range(4)], fx.Int32
    )


@traced
def ug_mfmas(c, task, ops, part, t, acc):
    """Task ``task``'s MFMAs of part ``part`` (``ops``: ``ug_loads``) against
    its picks' x rows (MXFP4, MXFP8, or A16W4 the CTA's K half of the bf16 x)
    from LDS (``t``: this lane's pick column's token): K steps in order, the
    sum order of one pass over the K slice."""
    lane, wave = c["lane"], c["wave"]
    fp8 = fp8_routed(c)
    i0, n = ug_parts(c)[part]
    rg = ug_row_group(task, wave)
    q = wave >> 2
    av = ops[:n]
    sw = ops[n:]
    kb = lane // 16
    for i in range_constexpr(n):
        # byte (st % 2) 2 + rg % 2 of the word; st % 2 is i % 2 (a part
        # starts at an even step)
        sa = (sw[i // 2] >> ((i % 2) * 16 + (rg % 2) * 8)) & 0xFF
        if const_expr(bf16_routed(c)):
            # the tile dequantized to 4 bf16 operands; x's k 32 kb + 8 j ..
            # + 8 of the step, in the K half's row
            wa = fp4_bf16_tile(av[i], sa)
            local = q * W13_Q + i0 + i
            for j in range_constexpr(4):
                xb = fx.Vector(
                    fx.ptr_load(c["xbl"] + xh_word(t, local, kb, j), result_type=_v4i())
                )
                acc = mfma_bf16(wa[j], xb.bitcast(fx.BFloat16), acc)
        else:
            st = q * W13_HALF + i0 + i
            if const_expr(fp8):
                bw = x8_lds_operand(c, t, st, kb)
            else:
                bw = fx.Vector(
                    fx.ptr_load(
                        c["xbl"] + (t * X4_ROW + st * 16 + kb * 4), result_type=_v4i()
                    )
                )
            sb = (fx.ptr_load(c["xsl"] + (t * XSC_WORDS + st)) >> (kb * 8)) & 0xFF
            acc = mfma_scaled(av[i], bw, acc, sa, sb, FP4, FP8 if fp8 else FP4)
    return acc


@traced
def ug_epilogue(c, task, first, count):
    """Task ``task``'s gate / up sums (``red``, past a barrier) of its picks at
    first .. first + count -> its mids (``ug_mx_mids``; A16W4 the K-half
    hand-off, ``ug_bf16_mids``)."""
    tid, red = c["tid"], c["red"]
    g = task % UG_GROUPS
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
    if const_expr(bf16_routed(c)):
        ug_bf16_mids(c, g, row, col, live, gv, uv)
    else:
        ug_mx_mids(c, g, row, col, live, gv, uv)


@traced
def ug_mx_mids(c, g, row, col, live, gv, uv):
    """W4A4 / W4A8: the gate / up of (pick row ``row``, mid column 32 g +
    ``col``) -> SwiGLU-OAI -> bf16 -> the 32 mids' MXFP4 or MXFP8 ->
    ``inter`` / ``isc`` (plain stores, announced by ``ugf`` once drained)."""
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
    -> ``inter`` (an even column stores its pair's word; plain, announced by
    ``ugf`` once drained)."""
    h = c["ugh"]
    at = ugp_pair(row, g, col)
    # half 1 never waits, so a half-0 task's wait always ends: its pair's
    # half-1 task is in the other task list, whose CTAs wait for nothing in
    # ug (``layout.task_lists``)
    if live & (h == 1):
        c["put_words"](c["ugp"], at, [gv.bitcast(fx.Int32), uv.bitcast(fx.Int32)])
    if live & (h == 0):
        got = c["poll"]([(c["ugp"], at, 2)])[0]
        y = bf16_round(swiglu(gv + _f32_of(got[0]), uv + _f32_of(got[1])))
        # both lanes of a pair are live (one pick row)
        word = bf16_pair(y, xshfl(y, 1)).bitcast(fx.Int32)
        if col % 2 == 0:
            st_dev(word, c["plain"], c["pw"]["inter"] + interb_word(row, g, col))


def ug_pick_token(c, first, count):
    """This lane's pick column's token (a column past the slot's picks repeats
    the last; its sums are never stored)."""
    rt = c["rt"]
    return fx.ptr_load(rt["order"] + first + fx.min(c["lane"] % 16, count - 1)) // TOPK


@traced
def take_ahead(c):
    """Thread 0 takes the next count of this CTA's task list's queue word
    (``qword``); its value is stored to LDS ``claim`` later (``put_claim``),
    so the atomic's return waits behind no load issued after it."""
    got = fx.Int32(0)
    if c["tid"] == 0:
        got = fx.Int32(queue.take_unmerged(queue.word(c["qword"])))
    return got


@traced
def put_claim(c, got):
    """``take_ahead``'s count -> LDS ``claim`` (thread 0); read past a
    barrier (``read_claim``)."""
    if c["tid"] == 0:
        fx.ptr_store(got, c["claim"])


def read_claim(c):
    """The list index of the claim in LDS: the list's first round is placed
    (``list_ctas`` entries), the queue counts the entries past it."""
    return list_ctas(c["routed_act"]) + uniform(fx.ptr_load(c["claim"]))


def flag_due(c, prev):
    """Whether this CTA owes ug task ``prev``'s ``ugf`` flag: a task ran
    (``prev`` >= 0) and, A16W4, it is the pair's half-0 task (the half-1 task
    hands its partials over in ``ugp``, the half-0 task finishes the pair)."""
    due = prev >= 0
    if const_expr(bf16_routed(c)):
        due = due & (c["ugh"] == 0)
    return due


@traced
def run_ug(c, bid, nu):
    """This CTA's ug tasks: its task list (``layout.task_lists``: W4A4 / W4A8
    one, A16W4 one a K half, this CTA's ``ugh``) is the union's ug tasks (a
    pair each), then its down tasks (``run_down``); its first round is placed
    (``c["k0"]``), the rest taken from the list's queue word (a CTA its
    earlier work or slower loads delayed takes fewer); a task's result is the
    same whichever CTA runs it. Returns the CTA's first list entry past the
    ug ones and the one claimed after it.

    The weights stream: the next task's part p is issued once this task's
    part p is used (``UG_PARTS``), so a CTA's loads never drain between
    tasks; the claim of the task after next is taken a whole task ahead. A
    task's ``ugf`` flag (A16W4: the half-0 task's, the pair's) is published in
    the next task, once its mid stores have drained behind the next task's
    loads (a flag at once would wait for those: ``vmcnt`` counts in issue
    order), the CTA's last task's at the end.

    A build that is not ``queued`` (1 row) takes no claim: its next task is
    ``list_ctas`` on (the list round-robin, the first round as placed), the
    down tasks ``run_down``'s placed ones."""
    tid, lane, wave, rt, red = c["tid"], c["lane"], c["wave"], c["rt"], c["red"]
    q = queued(c["S"])
    per = list_ctas(c["routed_act"])
    total = nu * UG_GROUPS
    last = total - 1
    task = c["k0"]
    # the claim before the loads: its return waits behind none of them
    got = take_ahead(c) if q else fx.Int32(0)
    t0 = fx.min(task, last)
    live0 = task < total
    parts = range(len(ug_parts(c)))
    ops = [ug_loads(c, t0, p, live0) for p in parts]
    if const_expr(q):
        put_claim(c, got)
        gpu.barrier()
    nxt = read_claim(c) if q else task + per
    if const_expr(q):
        # every thread has read it before the loop's first claim lands there
        gpu.barrier()
    prev = fx.Int32(-1)
    while task < total:
        got = take_ahead(c) if q else fx.Int32(0)
        slot = task // UG_GROUPS
        first = fx.ptr_load(rt["first"] + slot)
        count = uniform(fx.ptr_load(rt["first"] + slot + 1) - first)
        t = ug_pick_token(c, first, count)
        nt = fx.min(nxt, last)
        nlive = nxt < total
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        ops_n = []
        for p in range_constexpr(len(parts)):
            acc = ug_mfmas(c, task, ops[p], p, t, acc)
            if const_expr(p + 1 < len(parts)):
                rocdl.sched_barrier(0)
            ops_n.append(ug_loads(c, nt, p, nlive))
        fx.ptr_store(acc, red + (wave * 64 + lane) * 4)
        # every vector memory op older than the next task's loads is done:
        # the previous task's mid stores and the claim (at least the next
        # task's ``ug_steps`` tile loads were issued after them)
        rocdl.s_waitcnt(vmcnt=ug_steps(c))
        if const_expr(q):
            put_claim(c, got)
        gpu.barrier()
        if flag_due(c, prev) & (tid == 0):
            c["put"](c["ugf"], prev, fx.Int32(1))
        nxt_n = read_claim(c) if q else nxt + per
        ug_epilogue(c, task, first, count)
        # ``red`` and ``claim`` read by every thread before they are rewritten
        gpu.barrier()
        prev = task
        task = nxt
        nxt = nxt_n
        ops = ops_n
    if flag_due(c, prev):
        publish(c["put"], c["ugf"], prev, fx.Int32(1), tid == 0)
    return task, nxt


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


def w2_scale_words(c, e, rgd, nbytes):
    """The dwords of w2's E8M0 codes (row e 7168 + 16 rgd + lane % 16, K
    block 4 st + lane / 16) for the K steps: steps 2 m and 2 m + 1 share one
    (``scale_index``'s d bit). Raw: ``w2_code`` picks the byte at its use.
    ``nbytes``: the codes' buffer range (0: zeros, no memory traffic)."""
    lane = c["lane"]
    # rgd >= 0: a shift, not the floor division's sign fix-up
    return [
        fx.Int32(
            bo.buffer_load(
                rsrc(c["s2"], nbytes),
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


W2_ALL_BYTES = EXPERTS * W2_BYTES
S2_BYTES = EXPERTS * DIMS.w2_scale_bytes
W_SD_BYTES = HIDDEN * INTER_S
S_SD_BYTES = HIDDEN * INTER_S // 32


def down_batch(c) -> int:
    """A wave's slots whose w2 is in flight together: ``DOWN_BATCH``; A16W4
    one (a slot's bf16 B operands are 80 VGPRs, four times W4A4's)."""
    return 1 if bf16_routed(c) else DOWN_BATCH


def down_loads(c, rgd, nu, b0, j, live):
    """Batch slot j's w2 tiles and scale words (slot b0 + 8 j of a wave's
    batch from b0), its scale words first (a step's MFMA then waits for its
    own tile); past |U|, or when ``live`` is false, against an empty buffer
    range (zeros, no memory traffic)."""
    lane, rt = c["lane"], c["rt"]
    # wave-uniform (a buffer range in VGPRs is a waterfall loop)
    b0 = uniform(b0)
    slot = fx.min(b0 + WAVES * j, nu - 1)
    on = live & (b0 + WAVES * j < nu)
    e = uniform(fx.ptr_load(rt["expert"] + slot))
    sw = w2_scale_words(c, e, rgd, on.select(fx.Int32(S2_BYTES), fx.Int32(0)))
    r_w = rsrc(
        c["w2"] + fx.Int64(e) * W2_BYTES, on.select(fx.Int32(W2_BYTES), fx.Int32(0))
    )
    av = [fp4_tile_load(r_w, 0, rgd, INTER_R, st, lane * 4) for st in range(W2_STEPS)]
    return av + sw


def batch_loads(c, rgd, nu, b0, live):
    """``down_loads`` of every slot of a wave's batch from b0."""
    return [down_loads(c, rgd, nu, b0, j, live) for j in range(down_batch(c))]


def inter_operands(c, row):
    """A lane's B operands of mid row ``row`` (its K steps' words at lane group
    lane / 16 -- MXFP4: k 32 kb .. + 32; MXFP8: k 16 kb .. + 16 and 64 + 16 kb
    .. + 16 -- and their codes; bf16: k 32 kb + 8 j .. + 8, the 4 operands of
    bf16 MFMAs j = 0 .. 3, no code), device scope (every ug flag polled
    first: ``wait_ug``), as loaded (``live_operand`` at their use)."""
    kb = c["lane"] // 16
    iw = inter_words(c["routed_act"])
    base = c["pw"]["inter"] + row * iw
    out = []
    for st in range_constexpr(W2_STEPS):
        if const_expr(bf16_routed(c)):
            ops = []
            for j in range_constexpr(4):
                at = c["pw"]["inter"] + interb_word(row, st * 4 + kb, 8 * j)
                ops.append(fx.Vector(ld_dev(c["plain"], at, 4)))
            out.append((ops, None))
        else:
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
            bw = fx.Vector.from_elements(words, fx.Int32)
            out.append((bw, code))
    return out


def _live_words(v, live):
    """Vector ``v``'s words, zero where the column is dead."""
    return fx.Vector.from_elements(
        [live.select(v[d], fx.Int32(0)) for d in range(len(v))], fx.Int32
    )


def live_operand(b, live):
    """``inter_operands``' step: a dead column multiplies zero. Done where the
    MFMA takes it, so the wait for the load is there too."""
    bw, code = b
    if const_expr(code is None):
        # bf16: 4 operands, no code
        return [_live_words(v, live) for v in bw], None
    return _live_words(bw, live), live.select(code, fx.Int32(UNIT_SCALE))


def tile_row(c, first, count):
    """A lane's pick row of the tile at first (``count`` picks) and whether its
    column is live; a dead column (past the picks, or a slot past U: count 0)
    reads row first."""
    col = c["lane"] % 16
    return first + fx.max(fx.min(col, count - 1), 0), col < count


@traced
def down_tile(c, rgd, ops, first, count, bs):
    """A batch slot's w2 rows 16 rgd .. (``ops``: ``down_loads``) against its
    picks (their B operands ``bs``), a pick a column -> bf16(w acc) of each
    pick into LDS ``contrib`` (lane: rows 4 (lane / 16) + i of pick column
    lane % 16)."""
    lane, rt = c["lane"], c["rt"]
    row, live = tile_row(c, first, count)
    words = ops[W2_STEPS:]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    b_fmt = FP8 if fp8_routed(c) else FP4
    for st in range_constexpr(W2_STEPS):
        bw, sb = live_operand(bs[st], live)
        if const_expr(bf16_routed(c)):
            # A16W4: the w2 tile dequantized to bf16, 4 bf16 MFMAs a step
            wa = fp4_bf16_tile(ops[st], w2_code(words, rgd, st))
            for q in range_constexpr(4):
                acc = mfma_bf16(wa[q], bw[q].bitcast(fx.BFloat16), acc)
        else:
            acc = mfma_scaled(ops[st], bw, acc, w2_code(words, rgd, st), sb, FP4, b_fmt)
    if live:
        pid = fx.ptr_load(rt["order"] + row)
        t, k = pid // TOPK, pid % TOPK
        wt = fx.ptr_load(rt["route"] + route_pair(t, TOPK + k)).bitcast(fx.Float32)
        for i in range_constexpr(4):
            fx.ptr_store(
                bf16_round(acc[i] * wt),
                c["contrib"] + (pid * DOWN_ROWS + 4 * (lane // 16) + i),
            )


def sdown_loads(c, rgd, live):
    """The shared down's rows 16 rgd .., K chunk wave (waves past SD_STEPS,
    or every wave when ``live`` is false: zeros, no memory traffic; their sums
    are left out)."""
    lane, wave = c["lane"], c["wave"]
    kc = fx.min(wave, SD_STEPS - 1)
    # wave-uniform (a buffer range in VGPRs is a waterfall loop)
    on = live & (uniform(wave) < SD_STEPS)
    row = rgd * DOWN_ROWS + lane % 16
    return (
        fp8_tile_load(
            rsrc(c["w_sd"], on.select(fx.Int32(W_SD_BYTES), fx.Int32(0))),
            row,
            INTER_S,
            kc,
            lane // 16,
        ),
        fp8_scale_load(
            rsrc(c["s_sd"], on.select(fx.Int32(S_SD_BYTES), fx.Int32(0))),
            row,
            INTER_S,
            kc,
            lane // 16,
        ),
    )


@traced
def wait_ug(c, nu):
    """Every ug task's flag (the step's mids all stored; A16W4 a pair's): a
    CTA's down tasks then read ``inter`` without a poll between their batches
    (a poll's load waits, in issue order, for the next batch's weights in
    flight)."""
    tid = c["tid"]
    total = nu * UG_GROUPS
    n = max_u(c["S"]) * UG_GROUPS
    c["poll"](
        [(c["ugf"], fx.min(tid + THREADS * i, total - 1), 1) for i in range(passes(n))]
    )
    gpu.barrier()


@traced
def stage_down(c, rgd, nu, ops0, sops, nrgd):
    """Hidden rows 16 rgd ..: the shared down (waves < SD_STEPS a K chunk each,
    MXFP8 of ``smid`` from LDS), then every slot's w2 against its picks (wave
    w the slots w, w + 8, ..; a batch of ``down_batch`` in flight, the next
    issued under this one's MFMAs, past the last batch the next task's first:
    ``ops0`` / ``sops`` this task's first and its shared down's, issued a task
    ahead) -> bf16(w acc) a pick in LDS -> the routed sum in top-k order
    (fp32, then bf16) + bf16 shared -> bf16 ``ffn``. Every ug flag is in
    (``wait_ug``). Returns the next task ``nrgd``'s first batch and shared
    down operands (zeros past DOWN_TASKS)."""
    s, tid, lane, wave, red = c["S"], c["tid"], c["lane"], c["wave"], c["red"]
    rt = c["rt"]
    nbat = down_batch(c)
    g4 = lane // 16
    # every wave (a branch on the wave left its mask live from the kernel's
    # start); waves past SD_STEPS multiply zeros, left out of the sum
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
    nlive = nrgd < DOWN_TASKS
    nr = fx.min(nrgd, DOWN_TASKS - 1)
    # the next task's shared down first: its MFMA then waits for no batch
    # issued after it (``vmcnt`` counts in issue order)
    sops_n = sdown_loads(c, nr, nlive)
    b0 = wave
    ops = ops0
    # at least one pass, a wave past U's included (its picks none, its next
    # batch empty): every path issues the same loads, so the next task's
    # shared MFMA waits for none of them
    go = fx.Int32(1)
    while go != 0:
        nb = b0 + WAVES * nbat
        more = nb < nu
        slots = [fx.min(b0 + WAVES * j, nu - 1) for j in range(nbat)]
        firsts, counts, bss = [], [], []
        for j in range_constexpr(nbat):
            first = fx.ptr_load(rt["first"] + slots[j])
            count = fx.ptr_load(rt["first"] + slots[j] + 1) - first
            firsts.append(first)
            # a slot past U repeats the last: its picks are written once
            counts.append(((b0 + WAVES * j) < nu).select(count, fx.Int32(0)))
        # every slot's B operands first; then a slot's MFMAs, and the next
        # batch's weights for that slot (this task's, or past its last batch
        # the next task's first) into the registers they freed: in flight
        # under the next slot's MFMAs and B wait, and carried round the loop
        # without a copy (a copy would wait for them)
        for j in range_constexpr(nbat):
            row, _ = tile_row(c, firsts[j], counts[j])
            bss.append(inter_operands(c, row))
        nrg = more.select(rgd, nr)
        nb0 = more.select(nb, wave)
        nlv = more | nlive
        ops_n = []
        for j in range_constexpr(nbat):
            down_tile(c, rgd, ops[j], firsts[j], counts[j], bss[j])
            rocdl.sched_barrier(0)
            ops_n.append(down_loads(c, nrg, nu, nb0, j, nlv))
            rocdl.sched_barrier(0)
        b0 = nb
        ops = ops_n
        go = (b0 < nu).select(fx.Int32(1), fx.Int32(0))
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
            # a padding row has no picks (its contribs were never written)
            y = (tt < c["nv"]).select(y, fx.Float32(0.0))
            sh = bf16_round(row_sum(red, r, tt, waves=range(SD_STEPS)))
            out.append(bf16_round(y) + sh)
        bo.buffer_store(
            bf16_pair(out[0], out[1]).bitcast(fx.Int32),
            rsrc(c["ffn"]),
            (tt * HIDDEN + rgd * DOWN_ROWS + 2 * rp) // 2,
        )
    gpu.barrier()
    return ops, sops_n


@traced
def run_down(c, nu, task, nxt):
    """This CTA's down tasks: its task list's entries past the union's ug
    tasks (``run_ug`` hands over where the list stands; A16W4's two lists
    take every other down task, ``layout.list_down``), the claim a task
    ahead, each task's first batch and shared down issued under the one
    before. A build that is not ``queued``: tasks bid, bid + BLOCKS, ..
    (placed)."""
    q = queued(c["S"])
    act = c["routed_act"]
    h = c["ugh"] if bf16_routed(c) else 0
    total_ug = nu * UG_GROUPS
    d = list_down(task, total_ug, h, act) if q else fx.Int32(c["bid"])
    dn = list_down(nxt, total_ug, h, act) if q else d + BLOCKS
    live = d < DOWN_TASKS
    dd = fx.min(d, DOWN_TASKS - 1)
    # this CTA's first down weights in flight under the smid and ug waits
    sops = sdown_loads(c, dd, live)
    ops = batch_loads(c, dd, nu, c["wave"], live)
    load_smid(c)
    if live:
        wait_ug(c, nu)
    while d < DOWN_TASKS:
        got = take_ahead(c) if q else fx.Int32(0)
        ops, sops = stage_down(c, d, nu, ops, sops, dn)
        if const_expr(q):
            put_claim(c, got)
            gpu.barrier()
        d = dn
        dn = list_down(read_claim(c), total_ug, h, act) if q else dn + BLOCKS


# ------------------------------------------------------------ the FFN


def real_rows(c):
    """The pass's real rows (``nvalid``'s count, 1 .. S), or S when
    ``nvalid`` is 0: an empty buffer range reads 0 without a fault."""
    s, p = c["S"], c["nvalid"]
    has = p != fx.Int64(0)
    got = fx.Int32(
        bo.buffer_load(
            rsrc(p, has.select(fx.Int32(4), fx.Int32(0))), 0, vec_width=1, dtype=T.i32
        )
    )
    return uniform(has.select(fx.max(fx.min(got, s), 1), fx.Int32(s)))


@traced
def run_ffn(c, bid):
    """Every FFN stage of this CTA, in program order (``layout``'s
    placement): router / shared / route tasks, the union, ug, down."""
    s = c["S"]
    c["nv"] = real_rows(c)
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
    # a padding row is routed nowhere (production's ``num_valid_rows``)
    if (bid >= RT0) & (rt_tok < s) & (rt_tok < c["nv"]):
        stage_route(c, rt_tok)
    c["stamp"](6)
    enter_stage("union")
    # this CTA's task list (A16W4: its K half ``ugh``, the x half it stages)
    # and its placed entry there
    lst, c["k0"] = task_list(bid, c["routed_act"])
    if const_expr(bf16_routed(c)):
        c["ugh"] = lst
    nu = load_route(c)
    load_xb(c)
    c["stamp"](7)
    enter_stage("ug")
    # the routed and shared planes' bases, loaded once: a load in a task
    # loop would wait (``vmcnt`` in issue order) for every weight load in
    # flight
    c |= {n: weight(c, n) for n in ("w13", "s13", "w2", "s2", "w_sd", "s_sd", "queue")}
    # the list's queue word: the launch's words are one a list, in list order
    c["qword"] = c["queue"] + fx.Int64(lst) * 4 if bf16_routed(c) else c["queue"]
    task, nxt = run_ug(c, bid, nu)
    c["stamp"](8)
    enter_stage("down")
    run_down(c, nu, task, nxt)
    c["stamp"](9)
