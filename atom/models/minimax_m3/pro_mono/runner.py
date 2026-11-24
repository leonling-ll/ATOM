# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The phase-1 runner: every MoE layer's operands bound in place, the peer
buffer and mailboxes, a build of the layer kernel per step width, and the
segment a mono MoE layer runs instead of production's post-attention one.

A mono layer's segment (``segment``), Option A::

    core (attention output, before o_proj) + residual
      -> layer kernel: o_proj, AR #1, + residual, Gemma norm (+ MXFP8), the
         lookahead norm of layer L + 2 (MXFP8), then (upto = "ffn") the router,
         the routed experts (A16W4, W4A8 or W4A4: ``routed_act``) and the shared
         expert (MXFP8) -> this rank's FFN partial (``ffn_out``)
      -> (FFN partial, residual_mid): what production's segment returns, so
         the next layer's AR #2 + input norm runs unchanged (Option A)

The lookahead goes where ``ProLookaheadState.put`` would put it
(``resid_in[L + 2]``, MXFP8), so layer L + 2's indexer reads it as before.

``UPTO`` names the build: the "ffn" build writes the FFN partial itself
(``ffn_out`` in its ABI) and the segment skips production's MoE; a "norm"
build hands its (x, x8) to production's MoE instead. Switching is this one
constant. An "ffn" build reads each layer's FFN weights through ``moe_w``, a
device table of their addresses and the layer's work-queue word
(``layout.MOE_TABLE`` order) made once, past the scratch allocation, and
stages the routed mids (and, W4A4, x as MXFP4) in ``plain`` (and, A16W4, the
K halves' gate / up partials in ``scratch``).

Steps: ``begin_step`` (the mailboxes zeroed, the peer fence) once a forward, at
its first MoE layer, on the current stream, so a decode graph records it.
Every region is single-buffered with per-layer tags: production's AR #2
between two mono layers holds every rank until every rank finished the
earlier one. A layer's outputs are new tensors (a graph's pool under capture),
as production's are, so a residual the next layer's fused add updates in
place is never a buffer the kernel writes again.

Check mode (``ATOM_MONO_CHECK=1``, eager steps; a captured step runs mono
alone): each mono layer first runs production's segment at that width (its
own o_proj, then ``moe_route_glue``'s launch where it serves, else AR #1 +
norm; the lookahead put and the MoE) on a copy of the residual, then the
kernel, and rank 0 logs ``compare()`` of every output the two hand on; the
forward goes on from mono's.
"""

import logging
from dataclasses import dataclass, replace

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.utility.dtypes import fp8

from atom.models.minimax_m3.pro_mono.config import EPS, HIDDEN, MX_GROUP
from atom.models.minimax_m3.pro_mono.kernels.layer import ABI, build
from atom.models.minimax_m3.pro_mono.layout import (
    DIMS,
    MOE_WEIGHTS,
    REGIONS,
    P1Build,
    peer_bytes,
    plain_bytes,
    queue_word_offset,
    scratch_bytes,
    scratch_layout,
)
from atom.models.minimax_m3.pro_mono.weights import (
    ProMoeLayer,
    bind_layer,
    lookahead_reader,
    native_a8,
)
from atom.mono.plan.execution import BLOCKS
from atom.mono.runtime.compare import compare
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.debug import (
    DIAG_BYTES,
    given_up_waits,
    raise_if_given_up,
    region_namer,
)
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes
from atom.mono.runtime.uncached import uncached_bytes
from atom.mono.runtime.widths import WidthBuilds
from atom.utils import envs
from atom.utils.forward_context import pass_scheduled_tokens_gpu

logger = logging.getLogger("atom")

WHAT = "MiniMax-M3.1-Pro mono FFN"
# the build every mono layer runs (``layout.UPTO``)
UPTO = "ffn"

BF16 = torch.bfloat16
U8 = torch.uint8
# a kernel output by ABI name: (dtype, columns); a build allocates those its
# ABI has (``la8`` / ``la8s`` only on a layer with a lookahead reader)
OUTPUTS = {
    "h_mid": (BF16, HIDDEN),
    "x": (BF16, HIDDEN),
    "x8": (fp8, HIDDEN),
    "x8s": (U8, HIDDEN // MX_GROUP),
    "la8": (fp8, HIDDEN),
    "la8s": (U8, HIDDEN // MX_GROUP),
    "ffn_out": (BF16, HIDDEN),
}
LOOKAHEAD_OUTPUTS = ("la8", "la8s")


@dataclass(frozen=True)
class MonoLayer:
    """A bound MoE layer: its operands, the layer whose indexer reads its
    lookahead (None: no reader), the device table of its FFN operands'
    addresses and its queue word (``moe_w``; None until the scratch is
    allocated) and its build's kind."""

    w: ProMoeLayer
    reader: int | None
    moe_w: torch.Tensor | None

    @property
    def kind(self) -> tuple:
        """What a build of this layer bakes besides the width."""
        return (self.reader is not None, self.w.eps, self.w.eps_ahead or EPS)


def _real_rows_ptr(rows: int) -> int:
    """The device i32 holding the pass's real row count when it covers these
    rows (the forward context's ``scheduled_tokens_gpu``, what production's
    routing masks a graph's padding rows by), else 0: every row is routed."""
    count = pass_scheduled_tokens_gpu(rows)
    if count is None:
        return 0
    if count.dtype != torch.int32:
        raise RuntimeError(
            f"scheduled_tokens_gpu is {count.dtype}, the kernel reads i32"
        )
    return count.data_ptr()


class ProMonoRunner:
    """Owns the bound layers, the peer buffer, scratch and the per-width
    builds. Construction is collective over the TP group (bind, agree, then
    the peer handshake)."""

    def __init__(self, layers, state, check: bool, routed_act: str = "bf16") -> None:
        group = get_tp_group().cpu_group
        self.check = check
        # the routed experts' activations (``layout.ROUTED_ACTS``), TP-uniform
        # (``config.latch_rows``)
        self.routed_act = routed_act
        self.debug = envs.ATOM_MONO_DEBUG
        bind_agreed(lambda: self._bind(layers, state), group)
        self._allocate(group)
        for lid, lw in self.layers.items():
            queue = self.scratch.data_ptr() + queue_word_offset(self.scratch_key, lid)
            self.layers[lid] = replace(lw, moe_w=moe_table(lw.w, self.dev, queue))
        self.builds = WidthBuilds(
            self._build_width,
            lambda built: [(launch, ABI) for launch in built.values()],
            group,
            WHAT,
        )

    # ------------------------------------------------------------- binding

    def _bind(self, layers, state) -> None:
        tp = get_tensor_model_parallel_world_size()
        if tp != DIMS.tp:
            raise MonoUnsupported(f"TP {tp}")
        self.rank = get_tensor_model_parallel_rank()
        self.dev = torch.device("cuda", torch.cuda.current_device())
        cus = torch.cuda.get_device_properties(self.dev).multi_processor_count
        if cus != BLOCKS:
            raise MonoUnsupported(f"{cus} CUs: the kernel needs all {BLOCKS}")
        self.state = state
        self.layers: dict[int, MonoLayer] = {}
        for layer in layers:
            if not getattr(layer, "is_moe_layer", False):
                continue
            lid = layer.layer_num
            reader = lookahead_reader(lid, state)
            if reader is not None and not native_a8(state.index_linears[reader]):
                # production puts a bf16 lookahead there; the kernel writes MXFP8
                raise MonoUnsupported(
                    f"layer {reader}'s indexer projection is not native MXFP8"
                )
            ahead = state.norms[reader] if reader is not None else None
            w = bind_layer(layer, DIMS, self.rank, ahead)
            self.layers[lid] = MonoLayer(w, reader, None)
        if not self.layers:
            raise MonoUnsupported("no MoE layer")
        self.first = min(self.layers)
        self.last = max(self.layers)
        self.kinds = sorted({lw.kind for lw in self.layers.values()})

    def _allocate(self, group) -> None:
        key = P1Build(tokens=1, upto=UPTO, debug=self.debug, routed_act=self.routed_act)
        self.peers, self._finalizer = owned_peer_buffer(
            self, peer_bytes(), group, self.rank, DIMS.tp, self.dev
        )
        self.scratch_key = key
        self.scratch = uncached_bytes(scratch_bytes(key), self.dev)
        self.plain = uncached_bytes(plain_bytes(key), self.dev)
        self.diag = (
            self.scratch[: scratch_layout(key)["diag"][1]] if self.debug else None
        )
        self.mailboxes = StepMailboxes(self.peers, self.scratch, debug=self.debug)

    def close(self) -> None:
        """Free the peer mappings now (once no rank can launch another step)."""
        self._finalizer()

    # -------------------------------------------------------------- builds

    def _build_width(self, rows: int) -> dict:
        return {
            kind: build(
                P1Build(
                    tokens=rows, upto=UPTO, lookahead=kind[0], eps=kind[1],
                    eps_ahead=kind[2], debug=self.debug,
                    routed_act=self.routed_act,
                )
            )
            for kind in self.kinds
        }  # fmt: skip

    def prepare(self, rows: int) -> bool:
        """Whether every rank holds the ``rows``-row builds (collective)."""
        return self.builds.prepare(rows)

    def serves(self, layer_id: int) -> bool:
        return layer_id in self.layers

    # ------------------------------------------------------------- a layer

    def launch(self, layer_id: int, core: torch.Tensor, residual: torch.Tensor):
        """The layer kernel on ``core`` [S, 3072] and ``residual`` [S, 7168]:
        its outputs by ABI name (``OUTPUTS``)."""
        n = residual.shape[0]
        lw = self.layers[layer_id]
        o_k = DIMS.o_k
        if core.dtype != BF16 or core.numel() != n * o_k:
            raise RuntimeError(
                f"layer {layer_id}: core {core.dtype} {tuple(core.shape)},"
                f" expected bf16 [{n}, {o_k}]"
            )
        core = core.reshape(n, o_k).contiguous()
        residual = residual.contiguous()
        out = {
            name: torch.empty(n, cols, dtype=dtype, device=residual.device)
            for name, (dtype, cols) in OUTPUTS.items()
            if name in ABI.names
            and (lw.reader is not None or name not in LOOKAHEAD_OUTPUTS)
        }
        ptrs = {name: 0 for name in OUTPUTS if name in ABI.names}
        ptrs.update({name: t.data_ptr() for name, t in out.items()})
        w = lw.w
        args = {
            "core": core.data_ptr(), "w_o": w.w_o.data_ptr(), "s_o": w.s_o.data_ptr(),
            "res": residual.data_ptr(), "g_post": w.g_post.data_ptr(),
            "g_ahead": 0 if w.g_ahead is None else w.g_ahead.data_ptr(),
            "moe_w": lw.moe_w.data_ptr(), **ptrs,
            "scratch": self.scratch.data_ptr(), "plain": self.plain.data_ptr(),
            "nvalid": _real_rows_ptr(n),
            "peers": self.peers.addresses.data_ptr(), "rank": self.rank,
            "layer": layer_id, "tl": 0,
        }  # fmt: skip
        if layer_id == self.first:
            self.mailboxes.begin_step()
        stream = torch.cuda.current_stream()
        _run_compiled(self.builds[n][lw.kind], *ABI.pack(args), stream)
        if layer_id == self.last and self.debug and not capturing():
            self._raise_on_given_up_waits(n)
        return out

    def segment(self, layer, core: torch.Tensor, residual: torch.Tensor, reference):
        """Decoder layer ``layer``'s post-attention segment on the kernel:
        ``(FFN partial, residual_mid)``. ``reference(residual)``: production's
        segment of this layer's o_proj of ``core`` at this width (the
        ``moe_route_glue`` launch where it serves, else AR #1 + norm and the
        MoE), its lookahead put included -> ``(FFN partial, residual_mid, x,
        x8, x8s)`` (check mode only)."""
        lid = layer.layer_num
        ref = None
        if self.check and not capturing():
            ref = self._production(layer, residual, reference)
        out = self.launch(lid, core, residual)
        reader = self.layers[lid].reader
        if reader is not None:
            self.state.resid_in[reader] = (
                out["la8"],
                out["la8s"].view(torch.float8_e8m0fnu),
            )
        ffn = out.get("ffn_out")
        if ffn is None:
            ffn = layer._ffn(out["x"], out["x8"], out["x8s"], fuse=True)
        if ref is not None:
            self._log_check(lid, out, ffn, ref)
        return ffn, out["h_mid"]

    # ---------------------------------------------------------- check mode

    def _production(self, layer, residual, reference) -> dict:
        """Production's segment on a copy of ``residual`` (its fused add may
        update the residual in place): what it hands on, by output name. The
        lookahead it put is taken back out (the kernel's goes on)."""
        ffn, h_mid, x, x8, x8s = reference(residual.clone())
        ref = {"h_mid": h_mid, "x": x, "x8": x8, "x8s": x8s, "ffn": ffn}
        reader = self.layers[layer.layer_num].reader
        if reader is not None:
            # absent when the live variants left the latched ones
            put = self.state.resid_in.pop(reader, None)
            if isinstance(put, tuple):
                ref["la8"], ref["la8s"] = put
        return ref

    def _log_check(self, lid: int, out: dict, ffn: torch.Tensor, ref: dict) -> None:
        if self.rank != 0:
            return
        got = {**out, "ffn": ffn}
        parts = [
            f"{name} {compare(_values(got[name]), _values(want))}"
            for name, want in ref.items()
            if name in got
        ]
        logger.info(
            "%s check layer %d rows %d: %s",
            WHAT,
            lid,
            ffn.shape[0],
            " | ".join(parts),
        )

    def _raise_on_given_up_waits(self, rows: int) -> None:
        """A debug build's step: raise with every mailbox wait that gave up."""
        assert self.diag.numel() == DIAG_BYTES
        waits = given_up_waits(self.diag, region_namer(REGIONS))
        raise_if_given_up(WHAT, self.rank, rows, waits, self.mailboxes)


def moe_table(w: ProMoeLayer, dev: torch.device, queue: int) -> torch.Tensor:
    """The kernel's ``moe_w``: ``w``'s FFN operand addresses, then the
    layer's queue word ``queue``, i64 in ``layout.MOE_TABLE`` order (the
    tensors are the model's parameters and the word is in the runner's
    scratch, so the addresses hold for the runner's life)."""
    return torch.tensor(
        [getattr(w, name).data_ptr() for name in MOE_WEIGHTS] + [queue],
        dtype=torch.int64,
        device=dev,
    )


def _values(t: torch.Tensor) -> torch.Tensor:
    """A tensor ``compare`` can take: E8M0 scale codes as their integers."""
    if t.dtype in (U8, torch.float8_e8m0fnu):
        return t.view(U8).float()
    return t.float()


def capturing() -> bool:
    return torch.cuda.is_current_stream_capturing()
