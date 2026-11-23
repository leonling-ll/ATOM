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
MXFP4 for the routed experts, a W4A8 build's routed experts read x8)::

    router  R_TASKS CTAs from R0: this rank's 48 router rows (3 row groups x
            R_PARTS K slices) against bf16 x; part 0 adds the parts in order,
            bf16 -> every rank's ``rlog`` (an all-gather of the 384 logits)
    shared  SH_TASKS CTAs from SH0: 8 gate + 8 up rows of the MXFP8 shared
            gate_up against x8 -> bf16 g, u -> SwiGLU-OAI -> bf16 -> ``smid``
    route   S CTAs from RT0, token t: sigmoid, top-4 of sigmoid + bias, the
            4 unbiased sigmoids renormalized x 2.0 -> ``route``
    union   every CTA: ``route`` -> the union U of the step's experts and
            their picks (LDS), the routed B operand (x4, or x8) -> LDS
    ug      |U| x UG_GROUPS tasks round-robin: 32 gate + 32 up rows of an
            expert (MXFP4) against its picks -> SwiGLU-OAI -> bf16 -> MXFP4
            (W4A4) or MXFP8 (W4A8) mid rows (plain ``inter`` / ``isc``, a
            task's ``ugf`` flag)
    down    DOWN_TASKS (16 hidden rows): w2 of every expert of U against its
            picks' mids -> bf16(w acc) a pick, summed in top-k order -> bf16;
            the shared down (MXFP8 of ``smid``) -> bf16; bf16(routed + shared)
            -> ``ffn_out`` [S, 7168], this rank's FFN partial

Both builds' outputs are the production pair's residual and the AR #1 norm's
(``ffn_out``: the MoE's TP-partial output), so the next layer's AR #2 is unchanged.

Every peer region is sized for ``MAX_ROWS_LAYOUT`` rows whatever the build's
token count, so a wider build changes no offset.
"""

from dataclasses import dataclass, field

from atom.models.minimax_m3.pro_mono.config import (
    COLS,
    EPS,
    EXPERTS,
    HIDDEN,
    MAX_ROWS,
    MAX_ROWS_LAYOUT,
    MX_GROUP,
    ROUTED_ACTS,
    TOPK,
    ProDims,
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
# a routed expert's intermediate on this rank: the valid 640 and the 768 the
# planes are laid out (and zero padded) to. The pad's gate / up rows and w2
# columns are zero, so its mid is exactly zero, a whole MXFP4 block of its own
# (640 = 20 x 32): the kernel skips it and the sums are production's.
INTER_V = DIMS.inter_valid  # 640
INTER_R = DIMS.inter_r  # 768
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
# the routed experts' activation format (``config.ROUTED_ACTS``): "fp4" W4A4
# (aiter a4w4 numerics), "fp8" W4A8 (MXFP8 x and mids, the f8f6f4 MFMA with an
# fp4 A and an fp8 B)
ROUTED_ACT_DEFAULT = "fp8"


def inter_words(act: str) -> int:
    """Words of a pick's mid row in ``inter`` (MXFP4 or MXFP8)."""
    return INTER_WORDS if act == "fp4" else INTER8_WORDS


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

assert UG_GROUPS * UG_COLS == INTER_V and INTER_V % 128 == 0
assert R_KLEN == 1024 and SH0 + SH_TASKS <= RT0 and DOWN_TASKS <= 2 * BLOCKS
assert UG_COLS * MAX_ROWS <= THREADS and SH_ROWS * MAX_ROWS <= THREADS


# ---- operand addressing (ints or traced values)


def fp4_tile_dword(rg, k, st, lane):
    """``mx.fp4_tile_load``'s dword in one expert's MXFP4 plane (aiter's (16,
    16) shuffle, K = ``k``): lane ``lane``'s 16 B of row group ``rg`` at 128-k
    step ``st`` -- row 16 rg + lane % 16, k 128 st + 32 (lane / 16) .. + 32."""
    return rg * (2 * k) + st * 256 + lane * 4


def w13_scale_word(e, rg, st, lane):
    """The dword of w13's E8M0 codes (aiter ``shuffle_scale`` over [E 1536,
    224], ``mx.scale_index``) holding row e 1536 + 16 rg + lane % 16, K block
    4 st + lane / 16; the code is its byte (st % 2) 2 + rg % 2."""
    r32 = e * (DIMS.w13_rows // 32) + rg // 2
    cols = DIMS.w13_scale_cols
    return ((r32 * (cols // 8) + st // 2) * 4 + lane // 16) * 16 + lane % 16


def w2_scale_word(e, rgd_half, m, lane):
    """The dword of w2's E8M0 codes (``shuffle_scale`` over [E 7168, 24])
    holding row e 7168 + 16 rgd + lane % 16 (``rgd_half`` = rgd / 2), K
    blocks 4 st + lane / 16 of steps st = 2 m, 2 m + 1: byte (st % 2) 2 +
    rgd % 2."""
    r32 = e * (HIDDEN // 32) + rgd_half
    cols = DIMS.w2_scale_cols
    return ((r32 * (cols // 8) + m) * 4 + lane // 16) * 16 + lane % 16


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
    "scratch", "plain", "peers", "rank", "layer", "tl",
)  # fmt: skip
# ``moe_w``: a device table of the FFN operands' addresses (i64, this order;
# 0 for a norm build): this rank's band of the bf16 router (48 rows), the fp32
# correction bias, the routed MXFP4 planes and the shared expert's MXFP8 ones
# (``weights.ProMoeLayer``'s fields)
MOE_WEIGHTS = (
    "w_gate", "bias", "w13", "s13", "w2", "s2", "w_sgu", "s_sgu", "w_sd", "s_sd",
)  # fmt: skip


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

    def __post_init__(self):
        if not 1 <= self.tokens <= MAX_ROWS:
            raise ValueError(f"tokens {self.tokens} outside 1 .. {MAX_ROWS}")
        if self.upto not in UPTO:
            raise ValueError(f"upto {self.upto!r} not in {UPTO}")
        if self.routed_act not in ROUTED_ACTS:
            raise ValueError(f"routed_act {self.routed_act!r} not in {ROUTED_ACTS}")


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
# ugf   [ug task] a task's mid rows are stored
# smid  [token][shared mid column] bf16 pairs


def rpart_pair(task, t, r):
    return (task * MAX_ROWS_LAYOUT + t) * R_ROWS + r


def route_pair(t, k):
    """``route``'s pair of token t's k-th pick (k < TOPK: its expert; TOPK +
    k: its weight)."""
    return t * 2 * TOPK + k


def smid_pair(t, col):
    return (t * INTER_S + col) // 2


def scratch_layout(key: P1Build) -> dict:
    """Device-scope scratch (tagged pairs only, ``byte_layout``'s rule): a
    debug build's wait records, a FFN build's hand-offs. name -> (byte offset,
    bytes)."""
    from atom.mono.runtime.debug import DIAG_BYTES

    L = MAX_ROWS_LAYOUT
    lay = {"diag": (0, DIAG_BYTES)} if key.debug else {}
    if key.upto == "ffn":
        start = max((o + n for o, n in lay.values()), default=0)
        items = (
            ("xrdy", NT),
            ("rpart", R_TASKS * L * R_ROWS),
            ("route", L * 2 * TOPK),
            ("ugf", max_u(L) * UG_GROUPS),
            ("smid", L * INTER_S // 2),
        )
        lay |= pair_layout(items, align=16, start=start)
    return lay


def scratch_bytes(key: P1Build) -> int:
    lay = scratch_layout(key)
    end = max((o + n for o, n in lay.values()), default=0)
    return max(16, -(-end // 16) * 16)


def plain_layout(key: P1Build) -> dict:
    """Plain words at device scope, a FFN build's (``plain``; flagged by the
    tagged pairs that announce them): a W4A4 build's x as MXFP4 (``x4``: a
    token's e2m1 words, ``x4s``: its codes four a word, a norm CTA's chunk;
    a W4A8 build reads the norm's x8 instead) and the routed mids (``inter``:
    a pick's MXFP4 or MXFP8 row in the union's pick order, ``isc``: its codes,
    a word each). name -> (byte offset, bytes)."""
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
            ("isc", L * TOPK * UG_GROUPS * 4),
        )
    )


def plain_bytes(key: P1Build) -> int:
    lay = plain_layout(key)
    end = max((o + n for o, n in lay.values()), default=0)
    return max(16, -(-end // 16) * 16)


NORM_REGIONS = ("rs", "ag", "agsq")
FFN_REGIONS = ("xrdy", "rpart", "rlog", "route", "ugf", "smid")
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
    return out


def o_owner(task: int) -> int:
    """The rank owning o task ``task``'s rows (columns of the hidden)."""
    return task // O_TASKS_OWN


def ar_owner(task: int) -> int:
    """The rank owning AR + norm task ``task``'s columns."""
    return task // DIMS.own_tasks
