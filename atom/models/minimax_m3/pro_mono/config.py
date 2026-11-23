# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The deployments MiniMax-M3.1-Pro mono decode (phase 1) serves, and why any
other is refused.

Phase 1 replaces one segment of each MoE layer (4 .. 67) in decode and DSpark
verify forwards of at most ``MAX_ROWS`` rows: o_proj, the first all-reduce and
the post-attention Gemma RMSNorm (+ MXFP8, + the lookahead's norm of layer
L + 2), the router, the routed and shared experts and their combine. Attention,
the indexer, the dense layers, AR #2 and prefill stay on the production path.

Every width a kernel loops over, places tasks by or lays a region out with is a
build-time constant of ``ProDims(8)``: the kernels place o_proj, the column
owners of the all-reduce and the router's expert band by the TP8 split, so any
other TP size is refused by name (``ShardError``) before anything loads.

The phase is off unless ``ProVariants.mono_ffn_rows`` is set. Its value is
latched once, at the first MoE layer of the first forward (``route``), and
agreed over the TP group (``latch_rows``): the per-forward re-read of
``ATOM_M3PRO_VARIANTS`` must never flip it, because a forward whose ranks
disagree on mono waits in-kernel for ever.
"""

from dataclasses import dataclass

from atom.config import CompilationLevel
from atom.mono.plan.shard import Shard, ShardError, padded
from atom.mono.runtime.deployment import (
    common_refusals,
    first_refusal,
    resident_grid_refusals,
)
from atom.utils import envs

# ---------------------------------------------------------------- model widths

HIDDEN = 7168
Q_HEADS = 128
HEAD_DIM = 256
# V is K with the 64 rope dims dropped: o_proj's input is a head's 192 dims
V_HEAD_DIM = 192
EXPERTS = 384
TOPK = 4
# the routed experts' and the shared expert's intermediate width (one shared)
INTER_FULL = 5120
SHARED_EXPERTS = 1
# FusedMoE's MXFP4 weight prep pads a rank's routed intermediate to this
# (``moe.py`` ``pad_align``), and ``intermediate_pad = 0`` makes production
# compute the zero pad too, so a kernel over the padded width is exact
ROUTED_PAD_ALIGN = 256
EPS = 1e-6
ROUTE_SCALE = 2.0
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
# ``sparse_indexer_lookahead_{num_layers, start_layer}``: layer L's norm also
# writes the indexer input of layer L + AHEAD
LOOKAHEAD_AHEAD = 2
LOOKAHEAD_START = 2
LAYERS = 68
FIRST_MOE_LAYER = 4
# the Pro attention's page (``sparse_block_size``)
PAGE = 16

# an E8M0 scale covers a 32-block along K (MXFP8 and MXFP4 alike)
MX_GROUP = 32
# the AR + norm task: 128 columns, two a lane
COLS = 128

# Rows a forward serves at most: the widths P1 builds and gates on. A 16-row
# MFMA B operand holds them in one tile; 32 needs token groups (design T16).
MAX_ROWS = 16
# Rows the kernel ABI, the peer regions and the scratch are sized for from day
# one, so a later 32-row build changes no layout (``begin_step`` zeroes the
# whole buffers either way).
MAX_ROWS_LAYOUT = 32

# the TP sizes the kernels are built for
SUPPORTED_TP = (8,)


@dataclass(frozen=True)
class ProDims:
    """A TP rank's share of Pro's widths. Only TP8 exists: the o_proj
    placement, the all-reduce's column owners (``own_cols`` = 7 tasks of
    ``COLS``) and the router band (``e_rank`` experts) are baked for it."""

    tp: int

    def __post_init__(self):
        shard = Shard(self.tp)
        shard.split(Q_HEADS, "query heads")
        shard.split(INTER_FULL, "expert intermediate")
        shard.split(EXPERTS, "router experts")
        own = shard.split(HIDDEN, "hidden columns")
        if own % COLS:
            raise ShardError(
                f"a rank's {own} columns at TP {self.tp} are not whole "
                f"{COLS}-column tasks"
            )
        if self.tp not in SUPPORTED_TP:
            raise ShardError(
                f"TP {self.tp}: the Pro mono kernels are built for TP "
                f"{SUPPORTED_TP} only"
            )

    @property
    def heads(self) -> int:
        """The rank's query heads."""
        return Q_HEADS // self.tp

    @property
    def o_k(self) -> int:
        """o_proj's K on this rank: its heads' V dims."""
        return self.heads * V_HEAD_DIM

    @property
    def e_rank(self) -> int:
        """Router logits a rank computes: experts [e_rank r, e_rank (r + 1))."""
        return EXPERTS // self.tp

    @property
    def inter_s(self) -> int:
        """The shared expert's intermediate on this rank (unpadded)."""
        return INTER_FULL * SHARED_EXPERTS // self.tp

    @property
    def inter_valid(self) -> int:
        """A routed expert's real intermediate on this rank."""
        return INTER_FULL // self.tp

    @property
    def inter_r(self) -> int:
        """A routed expert's intermediate as loaded: padded with zeros to
        ``ROUTED_PAD_ALIGN``."""
        return padded(self.inter_valid, ROUTED_PAD_ALIGN)

    @property
    def own_cols(self) -> int:
        """The hidden columns whose all-reduce this rank owns."""
        return HIDDEN // self.tp

    @property
    def own_tasks(self) -> int:
        return self.own_cols // COLS

    @property
    def nt(self) -> int:
        """AR + norm column tasks over the whole hidden."""
        return HIDDEN // COLS

    # ---- the routed planes (MXFP4, two values a byte) as FusedMoE leaves them

    @property
    def w13_rows(self) -> int:
        """One expert's gate_up rows: the gate block, then the up block."""
        return 2 * self.inter_r

    @property
    def w13_bytes(self) -> int:
        return self.w13_rows * HIDDEN // 2

    @property
    def w13_scale_cols(self) -> int:
        return HIDDEN // MX_GROUP

    @property
    def w13_scale_bytes(self) -> int:
        return self.w13_rows * self.w13_scale_cols

    @property
    def w2_bytes(self) -> int:
        return HIDDEN * self.inter_r // 2

    @property
    def w2_scale_cols(self) -> int:
        return self.inter_r // MX_GROUP

    @property
    def w2_scale_bytes(self) -> int:
        return HIDDEN * self.w2_scale_cols

    # ---- the shared expert (native MXFP8 1x32)

    @property
    def sh_gate_up(self) -> tuple[int, int]:
        """gate_up's (rows, K): [gate | up] over the hidden."""
        return 2 * self.inter_s, HIDDEN

    @property
    def sh_down(self) -> tuple[int, int]:
        return HIDDEN, self.inter_s


def _hf(atom_config):
    return getattr(atom_config.hf_config, "text_config", atom_config.hf_config)


def _sparse(hf) -> dict:
    return getattr(hf, "sparse_attention_config", None) or {}


def _speculative_width(atom_config) -> int | None:
    """Rows a verify takes a sequence (k + 1), or None without a drafter."""
    spec = getattr(atom_config, "speculative_config", None)
    if spec is None:
        return None
    return int(spec.num_speculative_tokens) + 1


def config_refusal(atom_config) -> str | None:
    """Why this deployment cannot use Pro mono phase 1, or None."""
    hf = _hf(atom_config)
    sparse = _sparse(hf)
    spec = getattr(atom_config, "speculative_config", None)
    width = _speculative_width(atom_config)
    rotary = int(
        (getattr(hf, "head_dim", 0) or 0)
        * (getattr(hf, "partial_rotary_factor", 1.0) or 1.0)
    )
    checks = [(envs.ATOM_MONO_ENABLE, "switched off")]
    # Decode CP is harmless: the attention, the only reader of the KV split,
    # stays on the production path (``common_refusals``' docstring).
    checks += common_refusals(
        atom_config, ProDims, SUPPORTED_TP, refuse_decode_cp=False
    ) + resident_grid_refusals(atom_config)
    level = getattr(getattr(atom_config, "compilation_config", None), "level", None)
    checks += [
        # The gate is Python in the decoder layer's forward, which a compiled
        # level would trace (Pro's recipe runs level 0; absent: no compile).
        (
            level in (None, CompilationLevel.NO_COMPILATION),
            f"compilation level {level}",
        ),
        # P1 reads no KV. Page 16 is refused as a scope limit, not a numeric
        # one: it is the only Pro deployment P1 is built and measured for.
        (
            atom_config.kv_cache_block_size == PAGE,
            f"block size {atom_config.kv_cache_block_size}",
        ),
        # the checkpoint's widths, which the kernels bake
        ((h := getattr(hf, "hidden_size", None)) == HIDDEN, f"hidden {h}"),
        (
            (n := getattr(hf, "num_attention_heads", None)) == Q_HEADS,
            f"{n} attention heads",
        ),
        (
            (vd := sparse.get("value_head_dim") or HEAD_DIM - rotary) == V_HEAD_DIM,
            f"value head dim {vd}",
        ),
        ((e := getattr(hf, "num_local_experts", None)) == EXPERTS, f"{e} experts"),
        ((k := getattr(hf, "num_experts_per_tok", None)) == TOPK, f"top-{k}"),
        (
            (i := getattr(hf, "intermediate_size", None)) == INTER_FULL,
            f"intermediate {i}",
        ),
        (
            (sh := getattr(hf, "n_shared_experts", 0)) == SHARED_EXPERTS,
            f"{sh} shared experts",
        ),
        # The router stage bakes sigmoid scoring with a correction bias,
        # renormalized top-4 scaled by 2.0, no expert groups: production reads
        # these off the config, so a checkpoint that changed one would be
        # served by the wrong router with no error at all.
        (
            (sf := getattr(hf, "scoring_func", "sigmoid")) == "sigmoid",
            f"scoring {sf!r}",
        ),
        (bool(getattr(hf, "use_routing_bias", False)), "no routing bias"),
        (
            float(getattr(hf, "routed_scaling_factor", 1.0)) == ROUTE_SCALE,
            f"routed scaling {getattr(hf, 'routed_scaling_factor', 1.0)}",
        ),
        ((getattr(hf, "n_group", None) or 1) == 1, "grouped top-k"),
        (
            float(getattr(hf, "swiglu_alpha", SWIGLU_ALPHA)) == SWIGLU_ALPHA,
            "SwiGLU alpha",
        ),
        (
            float(getattr(hf, "swiglu_limit", SWIGLU_LIMIT)) == SWIGLU_LIMIT,
            "SwiGLU limit",
        ),
        # the routed and shared stages bake ``u + 1``
        (float(getattr(hf, "swiglu_beta", 1.0)) == 1.0, "SwiGLU beta"),
        # The lookahead norm the kernel writes is layer L + AHEAD's input.
        (
            int(sparse.get("sparse_indexer_lookahead_num_layers") or 0)
            in (0, LOOKAHEAD_AHEAD),
            "lookahead distance",
        ),
        # A served forward is a plain decode or a DSpark verify. Any other
        # method would be admitted on a path never built or measured for it.
        (spec is None or spec.method == "dspark", "not DSpark"),
        # A verify of one sequence must fit, or no verify would ever take mono.
        (
            width is None or width <= MAX_ROWS,
            f"verify width {width} past {MAX_ROWS} rows",
        ),
    ]
    return first_refusal(checks)


def rows_refusal(rows: int) -> str | None:
    """Why a ``mono_ffn_rows`` value cannot be latched, or None (0 is "off",
    not a refusal)."""
    if not isinstance(rows, int) or isinstance(rows, bool):
        return f"mono_ffn_rows {rows!r} is not an integer"
    if not 0 <= rows <= MAX_ROWS:
        return f"mono_ffn_rows {rows} outside 0 .. {MAX_ROWS}"
    return None


# the routed experts' activation formats (``layout.ROUTED_ACTS``: plain
# strings here, so the gate imports nothing of the kernels)
ROUTED_ACTS = ("fp4", "fp8")


def act_refusal(act) -> str | None:
    """Why a ``mono_routed_act`` value cannot be latched, or None."""
    if act not in ROUTED_ACTS:
        return f"mono_routed_act {act!r} not in {ROUTED_ACTS}"
    return None


def latch_rows(rows, group, act="fp8") -> tuple[int, str | None]:
    """``(the row limit every rank serves, why it is 0 or None)``.

    ``rows``: this rank's ``ProVariants.mono_ffn_rows`` at runner creation,
    ``act`` its ``mono_routed_act`` (the runner builds with it).
    ``group``: the TP CPU (gloo) group, None for one rank. Collective: every
    rank must call it at the same point. A value one rank refuses, or ranks
    that read different values (a variants file edited between their reads),
    turn the phase off on every rank."""
    from atom.distributed.group_agreement import group_max_each

    why = rows_refusal(rows) or act_refusal(act)
    local = 0 if why is not None else rows
    ai = 0 if why is not None else ROUTED_ACTS.index(act)
    # bad flag, max, -min of the rows and of the format in one collective
    bad, hi, neg_lo, ahi, neg_alo = group_max_each(
        (int(why is not None), local, -local, ai, -ai), group
    )
    if bad:
        return 0, why or "another TP rank refused mono_ffn_rows or mono_routed_act"
    if hi != -neg_lo:
        return 0, f"TP ranks read mono_ffn_rows {-neg_lo} .. {hi}"
    if ahi != -neg_alo:
        return 0, (
            f"TP ranks read mono_routed_act {ROUTED_ACTS[-neg_alo]!r} and"
            f" {ROUTED_ACTS[ahi]!r}"
        )
    return hi, None if hi else "mono_ffn_rows 0"


def gate_rows(rows: int, latched: int) -> bool:
    """Whether a forward of ``rows`` graph rows (pad rows included, so it is
    TP-uniform) is in the latched phase-1 range."""
    return 1 <= rows <= latched <= MAX_ROWS
