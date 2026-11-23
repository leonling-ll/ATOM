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

``upto = "norm"`` (the only build so far) ends there and writes x and x8 to
HBM, where the production MoE reads them; the router and expert stages follow
in later builds (design T7 .. T10). Its outputs are the production pair's
residual and the AR #1 norm's, so the next layer's AR #2 is unchanged.

Every peer region is sized for ``MAX_ROWS_LAYOUT`` rows whatever the build's
token count, so a wider build changes no offset.
"""

from dataclasses import dataclass, field

from atom.models.minimax_m3.pro_mono.config import (
    COLS,
    EPS,
    HIDDEN,
    MAX_ROWS,
    MAX_ROWS_LAYOUT,
    MX_GROUP,
    ProDims,
)
from atom.mono.plan.check import RegionDecl
from atom.mono.plan.layout import pair_layout
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
UPTO = ("norm",)
# timeline points: 0 start, 1 core quantized, 2 o pushed, 3 reduced /
# gathered, 4 sums of squares in, 5 stored, 6 end
TL_POINTS = 7

assert O_TASKS <= 256 and OWN % O_ROWS == 0 and OWN % COLS == 0

ABI_NAMES = (
    "core", "w_o", "s_o", "res", "g_post", "g_ahead",
    "h_mid", "x", "x8", "x8s", "la8", "la8s",
    "scratch", "peers", "rank", "layer", "tl",
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

    def __post_init__(self):
        if not 1 <= self.tokens <= MAX_ROWS:
            raise ValueError(f"tokens {self.tokens} outside 1 .. {MAX_ROWS}")
        if self.upto not in UPTO:
            raise ValueError(f"upto {self.upto!r} not in {UPTO}")


# ------------------------------------------------------------- peer regions
# rs   [src rank][token][owned column] bf16 pairs: the partials of the columns
#      this rank owns
# ag   [token][column] bf16 pairs: the owners' sums
# agsq [task][token] two words a cell (sum of squares over f32 v, over bf16 v)


def rs_pair(src, t, col_in_own):
    """``rs``'s pair index of rank ``src``'s partial, token ``t``, owned
    column ``col_in_own`` (even)."""
    return ((src * MAX_ROWS_LAYOUT + t) * OWN + col_in_own) // 2


def ag_pair(t, col):
    return (t * HIDDEN + col) // 2


def agsq_pair(task, t):
    """``agsq``'s first pair of (task, token): two pairs a cell."""
    return (task * MAX_ROWS_LAYOUT + t) * 2


def peer_items():
    L = MAX_ROWS_LAYOUT
    return (
        ("rs", TP * L * OWN // 2),
        ("ag", L * HIDDEN // 2),
        ("agsq", NT * L * 2),
    )


def peer_layout() -> dict:
    """name -> (byte offset, bytes), each region 16 B aligned."""
    return pair_layout(peer_items(), align=16)


def peer_bytes() -> int:
    """The PeerBuffer's data size, a multiple of 16 (``step_begin``)."""
    end = max(o + n for o, n in peer_layout().values())
    return -(-end // 16) * 16


def scratch_layout(key: P1Build) -> dict:
    """Device-scope scratch: only a debug build's wait records so far."""
    from atom.mono.runtime.debug import DIAG_BYTES

    return {"diag": (0, DIAG_BYTES)} if key.debug else {}


def scratch_bytes(key: P1Build) -> int:
    lay = scratch_layout(key)
    end = max((o + n for o, n in lay.values()), default=0)
    return max(16, -(-end // 16) * 16)


REGIONS = ("rs", "ag", "agsq")


def decls(key: P1Build) -> list[RegionDecl]:
    """The hand-off contract: each region's space and the stage that puts it."""
    del key  # every build so far has the same regions
    return [
        RegionDecl("rs", Space.PEER, "o"),
        RegionDecl("ag", Space.PEER, "reduce"),
        RegionDecl("agsq", Space.PEER, "reduce"),
    ]


def o_owner(task: int) -> int:
    """The rank owning o task ``task``'s rows (columns of the hidden)."""
    return task // O_TASKS_OWN


def ar_owner(task: int) -> int:
    """The rank owning AR + norm task ``task``'s columns."""
    return task // DIMS.own_tasks
