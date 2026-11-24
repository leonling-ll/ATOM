# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Which Pro MoE layers of a forward run their FFN segment on the phase-1
kernel: the gate ``MiniMaxM3ProDecoderLayer.forward`` asks at each MoE layer.

Every input of the decision is TP-uniform, because the kernel waits in-kernel
on every peer rank: the forward's graph rows (pad rows included), whether it is
a prefill or split into micro-batches, and the latched row limit. Never a
per-rank flag such as ``is_dummy_run`` (the attention returns zeros on every
rank there, so mono runs on zeros alike).

The limit is latched once, at the first MoE layer of the first forward (every
rank runs that forward), and agreed over the TP group (``config.latch_rows``):
``ATOM_M3PRO_VARIANTS`` is re-read per rank about once a second, so a file
edited while serving could otherwise give ranks different limits in one
forward. The variants the segment's hand-off depends on (the lookahead's
source and the MXFP8 norms, ``variants_refusal``) are checked at the same
point and turn the phase off if they do not hold; later edits of them are not
seen by the mono layers. A graph captured before the runner existed (or at a
width that did not build) runs production, which the gate keeps uniform.

Plain Python: no aiter or FlyDSL import (the runner is created through
``create``, which imports them), so the gate is checked on the CPU.
"""

import logging

from atom.models.minimax_m3.pro_mono.config import (
    ROUTED_ACT_NAMES,
    gate_rows,
    latch_rows,
    rows_refusal,
)
from atom.mono.runtime.route import LazyRunner
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")

WHAT = "MiniMax-M3.1-Pro mono FFN"


def variants_refusal(v, mx_norm: bool) -> str | None:
    """Why ``ProVariants`` ``v`` (with ``mx_norm``: this aiter's MXFP8 norms)
    cannot take the phase-1 segment, or None.

    The kernel writes what production's post-attention segment hands on under
    these variants only: the MXFP8 FFN input (``fuse_mxfp8_quant``, the MXFP8
    norm kernels) and the lookahead as the reading layer's input norm of the
    residual, MXFP8 (``ProLookaheadState.put``'s ``resid_in`` branch). Its
    in-kernel AR #1 sums in the Lamport all-reduce's order, so production's
    AR #1 must be the Lamport one at every width mono serves
    (``ar_lamport_rows``), or h_mid would differ from production's. It reads
    the attention's bf16 output, so the day0 decode, which hands o_proj E4M3
    rows from ``day0_decode_rows`` rows, must start above the row limit."""
    checks = [
        (not v.diag, "the diag variant"),
        (v.fuse_mxfp8_quant, "fuse_mxfp8_quant off"),
        (mx_norm, "aiter without the MXFP8 norms"),
        (v.lookahead, "lookahead off"),
        (v.lookahead_src == "resid_in", f"lookahead_src {v.lookahead_src!r}"),
        (
            # a value ``latch_rows`` refuses is reported there
            rows_refusal(v.mono_ffn_rows) is not None
            or v.ar_lamport_rows >= v.mono_ffn_rows,
            f"ar_lamport_rows {v.ar_lamport_rows} below {v.mono_ffn_rows}",
        ),
        (
            rows_refusal(v.mono_ffn_rows) is not None
            or not 0 < v.day0_decode_rows <= v.mono_ffn_rows,
            f"day0_decode_rows {v.day0_decode_rows} within {v.mono_ffn_rows}",
        ),
    ]
    return next((why for ok, why in checks if not ok), None)


class ProMonoRoute:
    """One model's gate and lazily created runner.

    ``refusal``: ``config.config_refusal`` of the deployment (None: served).
    ``create()``: the runner (collective; raises ``MonoUnsupported``).
    ``read()``: this rank's ``(ProVariants, mx_norm)``, read fresh.
    ``group()``: the TP CPU (gloo) group."""

    def __init__(self, refusal: str | None, create, read, group) -> None:
        self._read, self._group = read, group
        self._mono = LazyRunner(create, WHAT)
        self.rows = 0
        # the routed experts' activations the runner builds with (latched)
        self.routed_act = "bf16"
        self.check = False
        self._latched = False
        if refusal is not None:
            logger.info("%s off: %s", WHAT, refusal)
            self._mono.enabled = False

    @property
    def runner(self):
        return self._mono.runner

    def _latch(self) -> None:
        """Collective: every rank, at the first MoE layer of the first forward."""
        self._latched = True
        v, mx_norm = self._read()
        why = variants_refusal(v, mx_norm)
        # a refused variant set latches 0 here, which turns every rank off
        act = v.mono_routed_act
        self.rows, why = latch_rows(
            0 if why else v.mono_ffn_rows, self._group(), act=act
        )
        if not self.rows:
            if v.mono_ffn_rows:
                logger.warning("%s off: %s", WHAT, why)
            self._mono.enabled = False
            return
        self.routed_act = act
        # ATOM_MONO_CHECK is the process's environment, the same on every rank
        self.check = envs.ATOM_MONO_CHECK
        logger.info(
            "%s: forwards of 1 .. %d rows, routed experts %s%s",
            WHAT,
            self.rows,
            ROUTED_ACT_NAMES[act],
            ", checked against production" if self.check else "",
        )

    def take(self, layer_id: int, rows: int):
        """The runner, when this MoE layer of a ``rows``-row forward runs on
        it, else None. Collective on the first call and on a width's first
        step (the runner, then that width's kernels, built on every rank)."""
        if not self._mono.enabled:
            return None
        if not self._latched:
            self._latch()
            if not self._mono.enabled:
                return None
        if not gate_rows(rows, self.rows):
            return None
        fwd = get_forward_context()
        if (
            fwd.context is None
            or fwd.context.is_prefill
            or fwd.ubatch_slices is not None
        ):
            return None
        if not self._mono.ready(rows):
            return None
        runner = self._mono.runner
        return runner if runner.serves(layer_id) else None
