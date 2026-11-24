# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The phase-1 layer kernel's build key, argument list, mailbox regions and
hand-off contract: plain Python, importable without FlyDSL or aiter.

Stages, in program order (one launch a MoE layer and rank, ``BLOCKS`` CTAs of
``THREADS``)::

    o       CTA t < O_TASKS: the core rows (bf16 [S, 3072], written by the
            attention) -> MXFP8 1x32 in LDS (``quantize_fp8``'s bytes), then
            o_proj rows 32 t .. + 32 by scaled MFMA -> bf16 partial rows pushed
            to the rank owning those columns (``rs``: a reduce-scatter)
    reduce  CTA j < NT (128 columns): the owner sums the 8 ranks' partials in
            rank order (fp32 -> bf16), gathers the sum to every rank (``ag``)
            and the task's sums of squares (``agsq``: over f32 v and, for the
            lookahead, over bf16(v))
    gather  every other rank's CTA j reads the sum from ``ag``
    norm    the 56 tasks' sums of squares -> rstd; residual_mid = bf16(v),
            x = bf16(Gemma(v)), x8 = MXFP8(x); the lookahead's
            MXFP8(bf16(Gemma_{L+2}(bf16 v))) for layer L + 2's indexer

``upto = "norm"`` ends there and writes x and x8 to HBM, where the
production MoE reads them. ``upto = "ffn"`` goes on to the rank's FFN partial
(the norm CTAs flag their columns, ``xrdy``; a W4A4 build's also write x as
MXFP4 for the routed experts, a W4A8 build's routed experts read x8, an A16W4
build's the bf16 x)::

    router  R_TASKS CTAs from R0: this rank's 48 router rows (3 row groups x
            R_PARTS K slices) against bf16 x; part 0 adds the parts in order,
            bf16 -> every rank's ``rlog`` (an all-gather of the 384 logits)
    shared  SH_TASKS CTAs from SH0: 8 gate + 8 up rows of the MXFP8 shared
            gate_up against x8 -> bf16 g, u -> SwiGLU-OAI -> bf16 -> ``smid``
    route   S CTAs from RT0, token t (a real row: t < ``nvalid``'s count):
            sigmoid, top-4 of sigmoid + bias, the 4 unbiased sigmoids
            renormalized x 2.0 -> ``route``
    union   every CTA: ``route`` -> the union U of the real rows' experts and
            their picks (LDS), the routed B operand (x4, x8, or A16W4 the
            CTA's K half of the bf16 x) -> LDS
    ug      |U| x UG_GROUPS tasks, the first round placed and the rest from
            the launch's work queue (``MOE_TABLE``), each CTA's weights
            streaming from one task into the next: 32 gate + 32 up rows of an
            expert (MXFP4) against its picks -> SwiGLU-OAI -> bf16 -> MXFP4
            (W4A4), MXFP8 (W4A8) or bf16 (A16W4) mid rows (plain ``inter`` /
            ``isc``, a task's ``ugf`` flag). A16W4 splits each task over the
            K halves (``ug_split``), a task list and a queue word a half, a
            CTA's tasks all of its half (``task_lists``): the half-1 task
            sends its fp32 gate / up partials (``ugp``), the half-0 task adds
            them to its own and finishes the task
    down    DOWN_TASKS (16 hidden rows), the task list's entries past the ug
            tasks (the same queue: a CTA that drew fewer or faster ug tasks
            takes more down tasks, so the 448 fill no rounds of their own;
            A16W4 deals them to its two lists alternately; a 1-row build
            places ug and down tasks round-robin, ``queued``), each CTA's
            next task's first w2 batch in flight under the one before; every
            ug flag polled once: w2 of every expert of U against its
            picks' mids -> bf16(w acc) a pick, summed in top-k order -> bf16;
            the shared down (MXFP8 of ``smid``) -> bf16; bf16(routed + shared)
            -> ``ffn_out`` [S, 7168], this rank's FFN partial (a padding
            row's: its shared expert's alone)

Both builds' outputs are the production pair's residual and the AR #1 norm's
(``ffn_out``: the MoE's TP-partial output), so the next layer's AR #2 is unchanged.

Every peer region is sized for ``MAX_ROWS_LAYOUT`` rows whatever the build's
token count, so a wider build changes no offset.
"""

import functools
from dataclasses import dataclass, field

from atom.models.minimax_m3.pro_mono.config import (
    COLS,
    EPS,
    EXPERTS,
    HIDDEN,
    LAYERS,
    MAX_ROWS,
    MAX_ROWS_LAYOUT,
    MX_GROUP,
    ROUTED_ACTS,
    TOPK,
    ProDims,
    RoutedPlanes,
)
from atom.mono.plan.check import RegionDecl
from atom.mono.plan.execution import BLOCKS, THREADS
from atom.mono.plan.layout import byte_layout, pair_layout
from atom.mono.plan.trace import Space

DIMS = ProDims(8)
TP = DIMS.tp
O_K = DIMS.o_k  # 3072
OWN = DIMS.own_cols  # 896
NT = DIMS.nt  # 56 AR + norm tasks of COLS
O_ROWS = 32  # o_proj rows a task: two 16-row groups, 4 waves each over K
O_TASKS = HIDDEN // O_ROWS  # 224
O_TASKS_OWN = OWN // O_ROWS  # 28 a column owner
# a wave's K quarter: 64-k chunks (the (16, 16) shuffle's two 32-k tiles) and
# the E8M0 bytes of its 32-blocks
O_CPW = O_K // 64 // 4  # 12
O_SCALES_PW = O_K // MX_GROUP // 4  # 24
# a token's MXFP8 core row in LDS (words, padded off the bank stride) and its
# E8M0 codes (words, 4 a word)
O_ROW = O_K // 4 + 4
OSC_ROW = O_K // MX_GROUP // 4
UPTO = ("norm", "ffn")
# timeline points: 0 start, 1 core quantized, 2 o pushed, 3 reduced /
# gathered, 4 sums of squares in, 5 stored; FFN: 6 router / shared / route
# done, 7 union and x4 in, 8 ug done, 9 down done; 10 end
TL_POINTS = 11

assert O_TASKS <= 256 and OWN % O_ROWS == 0 and OWN % COLS == 0

# ---------------------------------------------------------------- the FFN
E_RANK = DIMS.e_rank  # 48 router rows a rank
R_ROWS = 16  # router rows a task (one MFMA tile)
R_GROUPS = E_RANK // R_ROWS  # 3
R_PARTS = 7  # K slices: 1024 columns, one 128-column chunk a wave
R_KLEN = HIDDEN // R_PARTS
R_TASKS = R_GROUPS * R_PARTS  # 21
# a routed expert's valid intermediate on this rank. The planes are laid out
# at the loader's padded width (``P1Build.inter_r``: 640 at Pro's pad_align
# 128, so no pad; 768 at the default 256), whose pad gate / up rows and w2
# columns are zero, so its mids are exactly zero, whole MXFP4 blocks of their
# own: the kernel computes the valid columns only and the sums are
# production's.
INTER_V = DIMS.inter_valid  # 640
UG_COLS = 32  # mid columns an ug task: one MXFP4 block of gate and of up
UG_GROUPS = INTER_V // UG_COLS  # 20
W13_STEPS = HIDDEN // 128  # 56 K steps of the f8f6f4 MFMA
W13_HALF = W13_STEPS // 2  # a wave's (4 row groups x 2 K halves = 8 waves)
W2_STEPS = INTER_V // 128  # 5
INTER_WORDS = INTER_V // 8  # a pick's MXFP4 mid row
DOWN_ROWS = 16
DOWN_TASKS = HIDDEN // DOWN_ROWS  # 448
DOWN_BATCH = 2  # a wave's experts whose w2 is in flight together
# the shared expert (MXFP8, unpadded 640 a rank): a task 8 gate + 8 up rows
# (one MFMA tile), the waves splitting K
INTER_S = DIMS.inter_s  # 640
SH_ROWS = 8
SH_TASKS = INTER_S // SH_ROWS  # 80
SH_CPW = W13_STEPS // 8  # 7 K chunks a wave
SD_STEPS = INTER_S // 128  # 5
# x as MXFP4 in LDS: a token's row of e2m1 words (padded off the bank stride)
# and its E8M0 codes four a word (a 128-column chunk's)
X4_WORDS = HIDDEN // 8
X4_ROW = X4_WORDS + 4
XSC_WORDS = HIDDEN // 128
# W4A8: x as MXFP8 in LDS (the norm's x8 rows, padded alike; the codes as
# x4's), and a pick's MXFP8 mid row in ``inter``
X8_WORDS = HIDDEN // 4
X8_ROW = X8_WORDS + 4
INTER8_WORDS = INTER_V // 4
# A16W4: a token's K half of the bf16 x in LDS (its words, padded alike: the
# bytes of an x8 row), and a pick's bf16 mid row in ``inter``. A task's K half
# is 2 quarters of W13_Q steps, a wave each (and its row group)
UG_HALVES = 2
XH_WORDS = HIDDEN // UG_HALVES // 2
XH_ROW = XH_WORDS + 4
INTERB_WORDS = INTER_V // 2
W13_Q = W13_HALF // 2  # 14
# the routed experts' activation format (``config.ROUTED_ACTS``): "fp4" W4A4
# (aiter a4w4 numerics), "fp8" W4A8 (MXFP8 x and mids, the f8f6f4 MFMA with an
# fp4 A and an fp8 B), "bf16" A16W4 (production's a16w4 numerics: bf16 x and
# mids, the MXFP4 weight dequantized to bf16 in the kernel, the bf16 MFMA)
ROUTED_ACT_DEFAULT = "bf16"


def inter_words(act: str) -> int:
    """Words of a pick's mid row in ``inter`` (MXFP4, MXFP8 or bf16)."""
    return {"fp4": INTER_WORDS, "fp8": INTER8_WORDS, "bf16": INTERB_WORDS}[act]


def ug_split(act: str) -> int:
    """The K slices of an ug task, each its own task on its own CTA: 2 for
    A16W4 (its bf16 x does not fit LDS whole at 16 rows, a K half does), 1
    else."""
    return UG_HALVES if act == "bf16" else 1


def xh_word(t, local, kb, j):
    """A16W4: the LDS word (from ``xs``) of token t's K-half row holding the
    B operand of bf16 MFMA j of the half's 128-k step ``local`` on lane group
    ``kb``: k 128 local + 32 kb + 8 j .. + 8 of the half (``fp4_bf16_tile``'s
    operand j of the w13 tile, k 32 kb + 8 j .. + 8 of its step)."""
    return t * XH_ROW + local * 64 + kb * 16 + 4 * j


def interb_word(row, blk, col):
    """A16W4: the word of ``inter`` holding the bf16 mid (column 32 ``blk`` +
    ``col``, an even ``col`` and the next) of pick row ``row``. An ug task
    stores block g, column col; the down's B operand j of step st, lane group
    kb, is block 4 st + kb, column 8 j (its 4 words)."""
    return row * INTERB_WORDS + blk * (UG_COLS // 2) + col // 2


# the shared mid as MXFP8 in LDS (the down's B operand): a token's row of e4m3
# words (padded) and its codes four a word
SM_WORDS = INTER_S // 4
SM_ROW = SM_WORDS + 4
SMSC_WORDS = INTER_S // 128
# placement: the router on the last CTAs (no o task: their weights are in
# flight from the start), the route before them, the shared gate_up past the
# norm CTAs
R0 = BLOCKS - R_TASKS  # 235
RT0 = R0 - MAX_ROWS  # 219
SH0 = NT  # 56
# the first ug round from the CTAs that have no stage between o and ug (past
# the shared gate_up's, before the route's): a narrow step's tasks start there
UG_FIRST_CTA = SH0 + SH_TASKS


def first_ug(bid):
    """CTA ``bid``'s placed ug task, an index of the step's first round (one a
    CTA, index 0 on ``UG_FIRST_CTA``; ints or traced values). A16W4: index
    2 p + h is ug pair p's K half h."""
    return (bid + (BLOCKS - UG_FIRST_CTA)) & (BLOCKS - 1)


# A step's routed tasks, per CTA (``ffn.run_ug`` / ``ffn.run_down``): a task
# list, its entries k the union's ug pairs (k < |U| x UG_GROUPS) then down
# tasks. W4A4 / W4A8 have one list; A16W4 one a K half (``ug_split``): list h
# is half h of every pair, and its CTAs stage only that half of x. A list's
# first round (``list_ctas`` entries) is placed, a CTA each (``task_list``);
# a queued build (``queued``) takes the later entries from the list's queue
# word, a narrow one takes every ``list_ctas``-th.
#
# Why A16W4's half-0 tasks, which wait for their pair's half-1 partials, never
# deadlock: a half-1 task waits for nothing (x is staged before any ug task),
# and the half-1 tasks are list 1's, whose CTAs run no half-0 task; so every
# half-1 task a CTA claims (or is placed) ends, however many tasks every CTA
# holds claimed ahead, and the half-0 waits end with them. A down task waits
# for every pair's flag: the half-0 tasks, which end. One list in claim order
# (a pair's half 1 before its half 0) would also be deadlock-free, but a CTA
# could then draw either half and would have to stage x's other half (as many
# bytes as the task's weights) in between.


def list_ctas(act: str) -> int:
    """The CTAs of a task list: BLOCKS / ``ug_split``."""
    return BLOCKS // ug_split(act)


def task_list(bid, act: str):
    """CTA ``bid``'s (task list, placed entry): A16W4 (``first_ug`` % 2, its
    K half; ``first_ug`` / 2, its pair), else (0, ``first_ug``); ints or
    traced values."""
    t = first_ug(bid)
    if ug_split(act) == 2:
        return t & 1, t >> 1
    return 0, t


def list_down(k, ug_tasks, lst, act: str):
    """The down task of list ``lst``'s entry k past its ``ug_tasks`` ug ones:
    A16W4's two lists take every other down task (list h: h, h + 2, ..)."""
    return (k - ug_tasks) * ug_split(act) + lst


assert UG_GROUPS * UG_COLS == INTER_V and INTER_V % 128 == 0
# A16W4: a CTA's ug tasks are all of one K half (its task list's) and a
# quarter's steps start on an even step (two steps a w13 scale word)
assert BLOCKS % 2 == 0 and W13_Q % 2 == 0 and 2 * W13_Q == W13_HALF
assert R_KLEN == 1024 and SH0 + SH_TASKS <= RT0
assert UG_COLS * MAX_ROWS <= THREADS and SH_ROWS * MAX_ROWS <= THREADS


# ---- operand addressing (ints or traced values)


def fp4_tile_dword(rg, k, st, lane):
    """``mx.fp4_tile_load``'s dword in one expert's MXFP4 plane (aiter's (16,
    16) shuffle, K = ``k``): lane ``lane``'s 16 B of row group ``rg`` at 128-k
    step ``st`` -- row 16 rg + lane % 16, k 128 st + 32 (lane / 16) .. + 32."""
    return rg * (2 * k) + st * 256 + lane * 4


def w13_scale_word(planes: RoutedPlanes, e, rg, st, lane):
    """The dword of w13's E8M0 codes (aiter ``shuffle_scale`` over [E
    ``w13_rows``, 224], ``mx.scale_index``) holding row e ``w13_rows`` + 16 rg
    + lane % 16, K block 4 st + lane / 16; the code is its byte (st % 2) 2 +
    rg % 2."""
    r32 = e * (planes.w13_rows // 32) + rg // 2
    cols = planes.w13_scale_cols
    return ((r32 * (cols // 8) + st // 2) * 4 + lane // 16) * 16 + lane % 16


def w2_scale_word(planes: RoutedPlanes, e, rgd_half, m, lane):
    """The dword of w2's E8M0 codes (``shuffle_scale`` over [E 7168,
    ``w2_scale_cols``]: 640's 20 groups padded to 24) holding row e 7168 + 16
    rgd + lane % 16 (``rgd_half`` = rgd / 2), K blocks 4 st + lane / 16 of
    steps st = 2 m, 2 m + 1: byte (st % 2) 2 + rgd % 2."""
    r32 = e * (HIDDEN // 32) + rgd_half
    cols = planes.w2_scale_cols
    return ((r32 * (cols // 8) + m) * 4 + lane // 16) * 16 + lane % 16


def w13_up_group(planes: RoutedPlanes) -> int:
    """The first 16-row group of w13's up block (row ``inter_r``)."""
    return planes.inter_r // 16


def mxfp8_tile_dword(row, cols, kc, g):
    """Dword offset, in an MXFP8 weight of ``cols`` columns in aiter's (16, 16)
    preshuffle (``[N / 16][K / 32][2][16][16]`` bytes), of the first 16 B piece
    of lane group ``g``'s operand of row ``row`` at 128-k chunk ``kc`` (k 16 g
    .. + 16; the second piece, k 64 + 16 g .. + 16, is 256 dwords on).
    MiniMax-M3's ``mxfp8_tile_dword``."""
    return (
        (row // 16) * (cols // 32 * 128)
        + kc * 512
        + (g // 2) * 128
        + (g % 2) * 64
        + (row % 16) * 4
    )


def max_u(s: int) -> int:
    """The bound of the expert union U of an ``s``-row step (``load_route``)."""
    return min(s * TOPK, EXPERTS)


ABI_NAMES = (
    "core", "w_o", "s_o", "res", "g_post", "g_ahead", "moe_w",
    "h_mid", "x", "x8", "x8s", "la8", "la8s", "ffn_out",
    "scratch", "plain", "nvalid", "peers", "rank", "layer", "tl",
)  # fmt: skip
# ``nvalid``: an i32 on the device holding the pass's real row count (the
# forward context's ``scheduled_tokens_gpu``; rows at or past it are a graph's
# padding, routed to no expert), or 0: every row is real.
# ``moe_w``: a device table of the FFN operands' addresses (i64, this order;
# 0 for a norm build): this rank's band of the bf16 router (48 rows), the fp32
# correction bias, the routed MXFP4 planes and the shared expert's MXFP8 ones
# (``weights.ProMoeLayer``'s fields)
MOE_WEIGHTS = (
    "w_gate", "bias", "w13", "s13", "w2", "s2", "w_sgu", "s_sgu", "w_sd", "s_sd",
)  # fmt: skip
# ``moe_w``'s entries: the weights, then the layer's work-queue word
# (``queue_word_offset`` into the runner's ``scratch``, so the step's zeroing
# clears it). In the table, not a kernel argument: an argument's SGPRs are
# held from the kernel's start, and the 2-pass all-reduce builds had none to
# spare.
MOE_TABLE = MOE_WEIGHTS + ("queue",)


@dataclass(frozen=True)
class P1Build:
    """Every parameter of a phase-1 layer build (``atom.mono.plan.build_key``)."""

    tokens: int = field(metadata={"sym": "s"})
    upto: str = field(default="norm", metadata={"sym": "u"})
    # the lookahead norm of layer L + 2 (False on layers 66 and 67)
    lookahead: bool = field(default=True, metadata={"sym": "la"})
    eps: float = EPS
    eps_ahead: float = EPS
    debug: bool = field(default=False, metadata={"sym": "g"})
    timeline: bool = field(default=False, metadata={"sym": "t"})
    # the routed experts' activations (``ROUTED_ACTS``; an FFN build's only)
    routed_act: str = field(default=ROUTED_ACT_DEFAULT, metadata={"sym": "ra"})
    # the routed planes' loaded intermediate (``RoutedPlanes``, read off the
    # bound weights): an FFN build's, 0 for a norm build
    inter_r: int = field(default=0, metadata={"sym": "ir"})

    def __post_init__(self):
        if not 1 <= self.tokens <= MAX_ROWS:
            raise ValueError(f"tokens {self.tokens} outside 1 .. {MAX_ROWS}")
        if self.upto not in UPTO:
            raise ValueError(f"upto {self.upto!r} not in {UPTO}")
        if self.routed_act not in ROUTED_ACTS:
            raise ValueError(f"routed_act {self.routed_act!r} not in {ROUTED_ACTS}")
        if self.upto == "ffn":
            RoutedPlanes(self.inter_r)
        elif self.inter_r:
            raise ValueError(f"inter_r {self.inter_r} on a norm build")

    @property
    def planes(self) -> RoutedPlanes:
        """An FFN build's routed planes."""
        return RoutedPlanes(self.inter_r)


# ------------------------------------------------------------- peer regions
# rs   [src rank][token][owned column] bf16 pairs: the partials of the columns
#      this rank owns
# ag   [token][column] bf16 pairs: the owners' sums
# agsq [task][token] two words a cell (sum of squares over f32 v, over bf16 v)
# rlog [src rank][token][its 48 experts] the router logits (f32 words holding
#      bf16 values)


def rs_pair(src, t, col_in_own):
    """``rs``'s pair index of rank ``src``'s partial, token ``t``, owned
    column ``col_in_own`` (even)."""
    return ((src * MAX_ROWS_LAYOUT + t) * OWN + col_in_own) // 2


def ag_pair(t, col):
    return (t * HIDDEN + col) // 2


def agsq_pair(task, t):
    """``agsq``'s first pair of (task, token): two pairs a cell."""
    return (task * MAX_ROWS_LAYOUT + t) * 2


def rlog_pair(src, t, e_in_rank):
    """``rlog``'s pair of rank ``src``'s logit of its expert ``e_in_rank``,
    token ``t``."""
    return (src * MAX_ROWS_LAYOUT + t) * E_RANK + e_in_rank


def peer_items():
    L = MAX_ROWS_LAYOUT
    return (
        ("rs", TP * L * OWN // 2),
        ("ag", L * HIDDEN // 2),
        ("agsq", NT * L * 2),
        ("rlog", TP * L * E_RANK),
    )


def peer_layout() -> dict:
    """name -> (byte offset, bytes), each region 16 B aligned."""
    return pair_layout(peer_items(), align=16)


def peer_bytes() -> int:
    """The PeerBuffer's data size, a multiple of 16 (``step_begin``)."""
    end = max(o + n for o, n in peer_layout().values())
    return -(-end // 16) * 16


# ---------------------------------------------------------- scratch regions
# Tagged pairs at device scope (one GPU), a FFN build's only:
# xrdy  [AR task] a norm CTA's x / x8 / x4 columns are stored
# rpart [router task][token][row] a router K slice's f32 partial (parts > 0)
# route [token][8] the picks' experts, then their weights (f32), top-k order
# ugf   [ug task] a task's mid rows are stored (A16W4: a K-half pair's)
# smid  [token][shared mid column] bf16 pairs
# ugp   (A16W4) [pick row][ug group][mid column] the half-1 task's f32 gate
#       and up partials, two pairs a cell


def rpart_pair(task, t, r):
    return (task * MAX_ROWS_LAYOUT + t) * R_ROWS + r


def route_pair(t, k):
    """``route``'s pair of token t's k-th pick (k < TOPK: its expert; TOPK +
    k: its weight)."""
    return t * 2 * TOPK + k


def smid_pair(t, col):
    return (t * INTER_S + col) // 2


def ugp_pair(row, g, col):
    """``ugp``'s first pair of pick row ``row``'s mid column ``col`` of ug
    group ``g``: the gate partial, then the up's."""
    return ((row * UG_GROUPS + g) * UG_COLS + col) * 2


def scratch_layout(key: P1Build) -> dict:
    """Device-scope scratch (tagged pairs only, ``byte_layout``'s rule): a
    debug build's wait records, a FFN build's hand-offs. name -> (byte offset,
    bytes)."""
    from atom.mono.runtime.debug import DIAG_BYTES

    L = MAX_ROWS_LAYOUT
    bf16 = key.routed_act == "bf16"
    lay = {"diag": (0, DIAG_BYTES)} if key.debug else {}
    if key.upto == "ffn":
        start = max((o + n for o, n in lay.values()), default=0)
        items = (
            ("xrdy", NT),
            ("rpart", R_TASKS * L * R_ROWS),
            ("route", L * 2 * TOPK),
            ("ugf", max_u(L) * UG_GROUPS),
            ("smid", L * INTER_S // 2),
            # last: the other regions' offsets are every format's
            *((("ugp", L * TOPK * UG_GROUPS * UG_COLS * 2),) if bf16 else ()),
        )
        lay |= pair_layout(items, align=16, start=start)
        # the work queues: an i32 count a task list and launch tag, plain words
        # in a region no pair shares (the step's zeroing clears them with the
        # pairs)
        end = max(o + n for o, n in lay.values())
        words = QUEUE_TAGS * ug_split(key.routed_act)
        lay |= byte_layout((("queue", words * 4),), start=end)
    return lay


# the routed stage's work queues (``ffn.run_ug`` then ``ffn.run_down``): an
# i32 a task list (A16W4: two, list order) and launch tag (``Mailbox``: layer
# + 1), counting the list's ug / down tasks taken past its first round
QUEUE_TAGS = LAYERS + 1
# a narrower build places its routed tasks instead (``queued``)
QUEUE_MIN_ROWS = 2


def queued(tokens: int) -> bool:
    """Whether a ``tokens``-row FFN build takes its routed tasks past the
    first round from the launch's queue, or places them as before the queue:
    list entry k on the CTA ``task_list`` puts it at, k + ``list_ctas`` on
    the same one, down task d on CTA d mod BLOCKS (no atomic). At one row the
    union (|U| <= 4: 80 ug pairs, a list's) fits the first round, nothing is
    left to balance, and the queue's claims, an agent-scope atomic every CTA
    on one word, cost the layer 2.5 us (v1tuned_1123T1524, W4A4 kernel slope
    32.94 -> 35.42 us, the ug stage's median CTA 0.1 -> 3.9 us)."""
    return tokens >= QUEUE_MIN_ROWS


@functools.cache
def queue_word_offset(key: P1Build, layer: int) -> int:
    """The byte offset in ``scratch`` of launch ``layer``'s first queue word
    (its task lists' words follow it, ``ug_split`` of them): computed on the
    host and handed to the kernel in its ``moe_w`` table (``MOE_TABLE``;
    ``atom.mono.device.queue``: never ``tag * 4`` inside a kernel; the kernel
    adds only its list's 4 h)."""
    if not 0 <= layer < LAYERS:
        raise ValueError(f"layer {layer} outside 0 .. {LAYERS - 1}")
    words = ug_split(key.routed_act)
    return scratch_layout(key)["queue"][0] + (layer + 1) * words * 4


def scratch_bytes(key: P1Build) -> int:
    lay = scratch_layout(key)
    end = max((o + n for o, n in lay.values()), default=0)
    return max(16, -(-end // 16) * 16)


def plain_layout(key: P1Build) -> dict:
    """Plain words at device scope, a FFN build's (``plain``; flagged by the
    tagged pairs that announce them): a W4A4 build's x as MXFP4 (``x4``: a
    token's e2m1 words, ``x4s``: its codes four a word, a norm CTA's chunk;
    a W4A8 build reads the norm's x8 instead, an A16W4 build the bf16 x) and
    the routed mids (``inter``: a pick's MXFP4, MXFP8 or bf16 row in the
    union's pick order; ``isc``: an MX row's codes, a word each). name ->
    (byte offset, bytes)."""
    if key.upto != "ffn":
        return {}
    L = MAX_ROWS_LAYOUT
    x4 = (
        (("x4", L * X4_WORDS * 4), ("x4s", L * XSC_WORDS * 4))
        if key.routed_act == "fp4"
        else ()
    )
    return byte_layout(
        (
            *x4,
            ("inter", L * TOPK * inter_words(key.routed_act) * 4),
            *((("isc", L * TOPK * UG_GROUPS * 4),) if key.routed_act != "bf16" else ()),
        )
    )


def plain_bytes(key: P1Build) -> int:
    lay = plain_layout(key)
    end = max((o + n for o, n in lay.values()), default=0)
    return max(16, -(-end // 16) * 16)


NORM_REGIONS = ("rs", "ag", "agsq")
FFN_REGIONS = ("xrdy", "rpart", "rlog", "route", "ugf", "smid", "ugp")
# every region a build may touch: a debug record names its region by its
# index here (``region_ids``)
REGIONS = NORM_REGIONS + FFN_REGIONS


def decls(key: P1Build) -> list[RegionDecl]:
    """The hand-off contract: each region's space and the stage that puts it."""
    out = [
        RegionDecl("rs", Space.PEER, "o"),
        RegionDecl("ag", Space.PEER, "reduce"),
        RegionDecl("agsq", Space.PEER, "reduce"),
    ]
    if key.upto == "ffn":
        out += [
            RegionDecl("xrdy", Space.SCRATCH, "norm"),
            # part 0 of a router task polls its other parts
            RegionDecl("rpart", Space.SCRATCH, "router", exchange=True),
            RegionDecl("rlog", Space.PEER, "router"),
            RegionDecl("route", Space.SCRATCH, "route"),
            RegionDecl("ugf", Space.SCRATCH, "ug"),
            RegionDecl("smid", Space.SCRATCH, "shared"),
        ]
    if key.upto == "ffn" and key.routed_act == "bf16":
        # a K-half pair's half-0 task polls its half-1 task's partials
        out.append(RegionDecl("ugp", Space.SCRATCH, "ug", exchange=True))
    return out


def o_owner(task: int) -> int:
    """The rank owning o task ``task``'s rows (columns of the hidden)."""
    return task // O_TASKS_OWN


def ar_owner(task: int) -> int:
    """The rank owning AR + norm task ``task``'s columns."""
    return task // DIMS.own_tasks
