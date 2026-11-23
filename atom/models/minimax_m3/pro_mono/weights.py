# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The tensors one Pro MoE layer's phase-1 kernel reads, bound in place.

Every binding is the production parameter itself or a view of it: no clone, no
requantization, no re-shuffle, no re-padding (owner ruling: an optimization
never changes the model's weights). The kernel reads exactly the bytes the
production path reads -- o_proj's and the shared expert's native MXFP8 1x32
(16x16-preshuffled weight, row-major E8M0 scale on gfx950), the bf16 router
(the checkpoint's fp32 cast once at load) with its fp32 correction bias, and
the routed experts' MXFP4 planes as ``FusedMoE`` left them after the recipe's
online NVFP4 -> MXFP4 conversion (lane-shuffled, padded 640 -> 768 with
zeros; ``_routed``).

Each layout assumption is checked here once, so a model loaded any other way is
refused (``MonoUnsupported``) instead of being misread. ``bind_layer`` runs
under ``consensus.bind_agreed``, so one rank's refusal is every rank's.

Nothing here imports aiter: the checks read torch dtypes and the attributes
ATOM's loaders set (``native_a8_group_rows``, ``runs_bf16``, the Parameter's
``is_shuffled``).
"""

from dataclasses import dataclass

import torch

from atom.models.minimax_m3.pro_mono.config import (
    EXPERTS,
    HIDDEN,
    MX_GROUP,
    TOPK,
    ProDims,
)
from atom.mono.runtime.consensus import MonoUnsupported

BF16 = torch.bfloat16
F32 = torch.float32
FP8 = torch.float8_e4m3fn
# an E8M0 scale plane is held as uint8 or as torch's e8m0 dtype
E8M0 = (torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8))
# packed MXFP4 (two E2M1 a byte) is held as torch's fp4x2 dtype or as uint8
FP4 = (torch.uint8, getattr(torch, "float4_e2m1fn_x2", torch.uint8))


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _mxfp8(linear, rows: int, cols: int, name: str):
    """(weight, scale) of a native MXFP8 1x32 linear: the checkpoint's bytes,
    the weight (16, 16)-preshuffled, the scale row-major [rows, cols / 32].

    Row-major because aiter's ``shuffle_mxfp8_bpreshuffle_scale`` returns its
    input unchanged on every arch but gfx1250 (``aiter/ops/shuffle.py``); this
    is safe only because ``common_refusals`` refuses any arch but gfx950."""
    _need(linear is not None, f"{name}: missing")
    _need(
        getattr(linear, "native_a8_group_rows", None) == 1,
        f"{name}: not the checkpoint's native 1x32 MXFP8",
    )
    _need(
        not getattr(linear, "runs_bf16", False),
        f"{name}: dequantized to bf16 at load (no group32 FP8 GEMM)",
    )
    _need(not getattr(linear, "is_output_padded", False), f"{name}: output padded")
    # the Parameter, not .data: ``is_shuffled`` is an attribute of the Parameter
    w = getattr(linear, "weight", None)
    _need(w is not None, f"{name}: no weight")
    _need(
        w.dtype == FP8 and tuple(w.shape) == (rows, cols) and w.is_contiguous(),
        f"{name}: weight {w.dtype} {tuple(w.shape)}, expected e4m3 {(rows, cols)}",
    )
    _need(getattr(w, "is_shuffled", False), f"{name}: weight not preshuffled")
    s = getattr(linear, "weight_scale", None)
    _need(
        s is not None
        and s.dtype in E8M0
        and s.numel() == rows * (cols // MX_GROUP)
        and s.is_contiguous(),
        f"{name}: scale {None if s is None else (s.dtype, tuple(s.shape))}",
    )
    return w.data, s.data


def _bf16_vec(t, n: int, name: str) -> torch.Tensor:
    t = getattr(t, "data", t)
    _need(
        t is not None and t.dtype == BF16 and t.numel() == n and t.is_contiguous(),
        f"{name}: {None if t is None else (t.dtype, tuple(t.shape))}",
    )
    return t


def _norm(norm, name: str) -> tuple[torch.Tensor, float]:
    """A Gemma RMSNorm's raw weight (the kernel applies 1 + w, as aiter's
    ``gemma_norm`` does: no folded copy) and its epsilon."""
    _need(norm is not None, f"{name}: missing")
    eps = getattr(norm, "variance_epsilon", None)
    _need(eps is not None, f"{name}: no epsilon")
    return _bf16_vec(norm.weight, HIDDEN, name), float(eps)


def _plane(t, dtypes, shape, name: str) -> torch.Tensor:
    """``t``'s bytes as a uint8 view (no copy), if it is ``shape`` of one of
    ``dtypes`` and contiguous."""
    _need(t is not None, f"{name}: missing")
    _need(
        t.dtype in dtypes and tuple(t.shape) == shape and t.is_contiguous(),
        f"{name}: {t.dtype} {tuple(t.shape)}, expected {shape} contiguous",
    )
    return t.data.view(torch.uint8)


def _routed(moe, dims: ProDims, name: str):
    """``FusedMoE``'s four MXFP4 planes where production left them, as uint8
    views: (w13, w13 scale, w2, w2 scale).

    The recipe's ``{"*.experts": "mxfp4"}`` converts the NVFP4 checkpoint at
    load (``FusedMoE._online_quant``): ``Mxfp4MoEMethod.create_weights`` sizes
    the planes with the rank's intermediate 640 padded to ``dims.inter_r``
    (``pad_align`` 256) and zero-fills them; the conversion writes gate rows
    [0, 640), up rows [768, 1408) and w2 K bytes [0, 320) (scale groups
    [0, 20)), so the pad is zero values with E8M0 code 0. ``MiniMaxM3MoE`` pins
    ``intermediate_pad = 0``, so production computes the padded 768 too. The
    lane-shuffled prep (``_process_weight_layout_after_loading`` branch C,
    gfx950) then applies aiter's ``shuffle_weight`` (16, 16) to w13 / w2 and
    ``shuffle_scale`` to each scale flattened to [experts x rows, groups]:
    no row or group padding at these widths (rows a multiple of 256, groups of
    8), so each plane is exactly its logical size."""
    experts = getattr(moe, "experts", None)
    _need(experts is not None, f"{name}: no routed experts")
    method = experts.quant_method
    _need(getattr(experts, "top_k", None) == TOPK, f"{name}: top-{experts.top_k}")
    _need(
        not getattr(experts, "source_is_nvfp4", False),
        f"{name}: still the checkpoint's NVFP4, not converted to MXFP4"
        ' (no --online_quant_config {"*.experts": "mxfp4"})',
    )
    _need(
        not getattr(experts, "num_fused_shared_experts", 0),
        f"{name}: the shared expert is fused into the routed tables",
    )
    _need(
        not getattr(method, "is_guinterleave", True),
        f"{name}: gate/up interleaved; the kernel reads the gate block, then up",
    )
    _need(
        not getattr(method, "fp8_activations", False),
        f"{name}: FP8-activation MXFP4 prep (another weight layout)",
    )
    _need(
        getattr(method, "hidden_pad", None) == 0,
        f"{name}: hidden_pad {getattr(method, 'hidden_pad', None)}",
    )
    _need(
        getattr(method, "intermediate_pad", None) == 0,
        f"{name}: intermediate_pad {getattr(method, 'intermediate_pad', None)};"
        " production must compute the zero pad as the kernel does",
    )
    # ``create_weights`` sets it to None on every layer; only the Triton prep
    # gives it a value
    _need(
        getattr(experts, "w13_swizzle_layout", None) is None,
        f"{name}: the Triton MXFP4 weight prep ran, not the lane-shuffled one",
    )
    _need(
        getattr(experts, "_mega_w1", None) is None,
        f"{name}: MegaMoE owns the experts and has released the raw planes",
    )
    w13, w2 = experts.w13_weight, experts.w2_weight
    _need(
        getattr(w13, "is_shuffled", False) and getattr(w2, "is_shuffled", False),
        f"{name}: expert planes are not preshuffled",
    )
    return (
        _plane(w13, FP4, (EXPERTS, dims.w13_rows, HIDDEN // 2), f"{name} w13"),
        _plane(
            experts.w13_weight_scale,
            E8M0,
            (EXPERTS * dims.w13_rows, dims.w13_scale_cols),
            f"{name} w13 scale",
        ),
        _plane(w2, FP4, (EXPERTS, HIDDEN, dims.inter_r // 2), f"{name} w2"),
        _plane(
            experts.w2_weight_scale,
            E8M0,
            (EXPERTS * HIDDEN, dims.w2_scale_cols),
            f"{name} w2 scale",
        ),
    )


def _router(moe, dims: ProDims, rank: int, name: str):
    """(this rank's band of the bf16 router weight -- experts [e_rank r,
    e_rank (r + 1)), a contiguous row view --, the fp32 correction bias over
    all 384)."""
    gate = getattr(moe, "gate", None)
    _need(gate is not None, f"{name}: no router")
    w = getattr(gate, "weight", None)
    _need(
        w is not None
        and w.dtype == BF16
        and tuple(w.shape) == (EXPERTS, HIDDEN)
        and w.is_contiguous(),
        f"{name} router: {None if w is None else (w.dtype, tuple(w.shape))},"
        f" expected bf16 {(EXPERTS, HIDDEN)}",
    )
    _need(getattr(gate, "bias", None) is None, f"{name} router: has a bias")
    bias = getattr(moe, "e_score_correction_bias", None)
    _need(
        bias is not None and bias.dtype == F32 and bias.numel() == EXPERTS,
        f"{name}: correction bias "
        f"{None if bias is None else (bias.dtype, tuple(bias.shape))}",
    )
    band = w.data[dims.e_rank * rank : dims.e_rank * (rank + 1)]
    return band, bias.data


@dataclass(frozen=True)
class ProMoeLayer:
    """One MoE layer's kernel operands, every tensor production's own storage."""

    layer_id: int
    # o_proj: e4m3 [7168, 3072] preshuffled, e8m0 [7168, 96]
    w_o: torch.Tensor
    s_o: torch.Tensor
    # post_attention_layernorm (raw w; the kernel applies 1 + w)
    g_post: torch.Tensor
    eps: float
    # input_layernorm of layer + AHEAD, whose indexer input this layer's norm
    # also writes; None past the last reader (layers 66 and 67)
    g_ahead: torch.Tensor | None
    eps_ahead: float | None
    # the router: this rank's 48 rows of the bf16 [384, 7168] gate, fp32 bias
    w_gate: torch.Tensor
    bias: torch.Tensor
    # The routed MXFP4 planes, uint8 views of FusedMoE's (``_routed``). Per
    # expert e, logical rows: w13 [1536, 7168 / 2] gate [0, 640), zero
    # [640, 768), up [768, 1408), zero [1408, 1536); w2 [7168, 768 / 2], K
    # [640, 768) zero. Two E2M1 a byte, the even k in the low nibble.
    # w13 / w2: aiter ``shuffle_weight`` (16, 16): 16-row x 16-byte tiles,
    # row-major inside, tiles K-fastest (``mono.device.mx.fp4_tile_load``).
    # s13 [384 x 1536, 224] / s2 [384 x 7168, 24]: E8M0, aiter ``shuffle_scale``
    # over the flattened rows (``mono.device.mx.scale_index``); pad code 0.
    w13: torch.Tensor
    s13: torch.Tensor
    w2: torch.Tensor
    s2: torch.Tensor
    # the shared expert: gate_up e4m3 [1280, 7168] + [1280, 224], down
    # e4m3 [7168, 640] + [7168, 20]
    w_sgu: torch.Tensor
    s_sgu: torch.Tensor
    w_sd: torch.Tensor
    s_sd: torch.Tensor


def bind_layer(layer, dims: ProDims, rank: int, ahead_norm=None) -> ProMoeLayer:
    """A ``MiniMaxM3ProDecoderLayer``'s phase-1 operands, or ``MonoUnsupported``.

    ``ahead_norm``: the ``input_layernorm`` of the layer that reads this one's
    lookahead (``ProLookaheadState.norms[L + AHEAD]``), None when none does."""
    layer_id = getattr(layer, "layer_num", None)
    name = f"layer {layer_id}"
    _need(getattr(layer, "is_moe_layer", False), f"{name}: not a MoE layer")
    _need(0 <= rank < dims.tp, f"{name}: rank {rank} of TP {dims.tp}")
    attn = layer.self_attn
    w_o, s_o = _mxfp8(attn.o_proj, HIDDEN, dims.o_k, f"{name} o_proj")
    g_post, eps = _norm(layer.post_attention_layernorm, f"{name} post-attn norm")
    g_ahead = eps_ahead = None
    if ahead_norm is not None:
        g_ahead, eps_ahead = _norm(ahead_norm, f"{name} lookahead norm")
    moe = layer.block_sparse_moe
    w_gate, bias = _router(moe, dims, rank, name)
    w13, s13, w2, s2 = _routed(moe, dims, f"{name} experts")
    sh = getattr(moe, "shared_experts", None)
    _need(sh is not None, f"{name}: no shared expert of its own")
    w_sgu, s_sgu = _mxfp8(sh.gate_up_proj, *dims.sh_gate_up, f"{name} shared gate_up")
    w_sd, s_sd = _mxfp8(sh.down_proj, *dims.sh_down, f"{name} shared down_proj")
    return ProMoeLayer(
        layer_id=layer_id,
        w_o=w_o,
        s_o=s_o,
        g_post=g_post,
        eps=eps,
        g_ahead=g_ahead,
        eps_ahead=eps_ahead,
        w_gate=w_gate,
        bias=bias,
        w13=w13,
        s13=s13,
        w2=w2,
        s2=s2,
        w_sgu=w_sgu,
        s_sgu=s_sgu,
        w_sd=w_sd,
        s_sd=s_sd,
    )
