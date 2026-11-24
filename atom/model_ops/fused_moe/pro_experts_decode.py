# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's routed experts at decode sizes on aiter's
``minimax_m3_moe_experts_decode`` (``ProVariants.moe_decode`` "day0").

A MoE call of at most 64 rows runs amd-kernels-day0's experts as ported into aiter: a
one-launch expert sort sized by the call's rows, the FlyDSL a16w4 GEMMs at a decode tile
config (block_m 16, slice-K over 4 waves in gemm1), and the 4 slots of a row summed in fp32
in slot order (no atomics). Weights are read where they lie: the same MXFP4 bytes and E8M0
scales, the same BF16 activations, the same SwiGLU clamp at 7. Routing is aiter's
``topk_gating``, as on the ``fused_moe`` path, with one difference: every row is routed,
padding rows too, because the sort takes no -1 ids; their outputs are not read.

``FusedMoE.forward_impl`` hands a call over when ``runs`` says so: a TP layer (no EP, no
DP gather) with the plain sigmoid routing and the routed scale in the weights, separated
gate/up MXFP4 weights on the FlyDSL path, no biases, nothing to all-reduce, and a call
``minimax_m3_moe_experts_decode_supported`` serves (its shapes, at most 64 rows, gfx950).
Every other call keeps ``fused_moe``.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger("atom")

TOP_K = 4
SWIGLU_LIMIT = 7.0


def _layer_qualifies(layer) -> bool:
    """What the decode experts need of a layer, apart from a call's rows."""
    from aiter import ActivationType

    from atom.model_ops.moe import Mxfp4MoEMethod

    method = layer.quant_method
    parallel = layer.moe_parallel_config
    return (
        isinstance(method, Mxfp4MoEMethod)
        and method.fused_experts is None
        and not method.is_guinterleave
        and not method.use_triton
        and not method.fp8_activations
        and not parallel.use_ep
        and parallel.dp_size == 1
        and parallel.dp_logical_ratio == 1
        and not layer.reduce_results
        and layer.top_k == TOP_K
        and layer.num_fused_shared_experts == 0
        and layer.scoring_func == "sigmoid"
        and not layer.use_grouped_topk
        and layer.custom_routing_function is None
        and layer.renormalize
        and getattr(layer, "fold_routed_scaling_factor", False)
        and layer.activation == ActivationType.Swiglu
        and float(getattr(layer, "swiglu_limit", 0.0)) == SWIGLU_LIMIT
        and getattr(layer, "w13_bias", None) is None
        and getattr(layer, "w2_bias", None) is None
    )


class ProDecodeExperts:
    """The routed experts of a decode-sized call (``FusedMoE.pro_decode_experts``)."""

    def __init__(self, experts, supported, topk_gating):
        self._experts = experts
        self._supported = supported
        self._topk_gating = topk_gating
        self._qualifies: dict[int, bool] = {}

    @classmethod
    def build(cls) -> ProDecodeExperts | None:
        """None when this aiter does not have the op."""
        try:
            from aiter import topk_gating
            from aiter.ops.flydsl import (
                minimax_m3_moe_experts_decode,
                minimax_m3_moe_experts_decode_supported,
            )
        except (ImportError, AttributeError) as e:
            logger.warning(
                "M3-Pro moe_decode=day0: aiter has no minimax_m3_moe_experts_decode "
                "(%s); the routed experts stay on fused_moe",
                e,
            )
            return None
        return cls(
            minimax_m3_moe_experts_decode,
            minimax_m3_moe_experts_decode_supported,
            topk_gating,
        )

    @classmethod
    def from_variants(cls) -> ProDecodeExperts | None:
        """The decode experts when ``ProVariants.moe_decode`` is "day0", else None."""
        from atom.model_ops.minimax_m3.pro_attn import refresh_variants

        choice = refresh_variants(force=True).moe_decode
        if choice == "aiter":
            return None
        if choice != "day0":
            raise ValueError(
                f"ATOM_M3PRO_VARIANTS moe_decode: {choice!r}; known: 'aiter', 'day0'"
            )
        return cls.build()

    def runs(self, layer, hidden_states: torch.Tensor) -> bool:
        key = id(layer)
        if key not in self._qualifies:
            self._qualifies[key] = _layer_qualifies(layer)
        return self._qualifies[key] and bool(
            self._supported(hidden_states, layer.w13_weight, layer.w2_weight)
        )

    def forward(
        self, layer, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        rows = hidden_states.shape[0]
        device = hidden_states.device
        topk_ids = torch.empty(rows, TOP_K, dtype=torch.int32, device=device)
        topk_weights = torch.empty(rows, TOP_K, dtype=torch.float32, device=device)
        # Every row, padding rows too: no num_valid_rows (the sort takes no -1 ids).
        self._topk_gating(
            topk_weights,
            topk_ids,
            router_logits,
            layer.e_score_correction_bias,
            True,
            layer.routed_scaling_factor,
            score_func="sigmoid",
        )
        return self._experts(
            hidden_states,
            topk_weights,
            topk_ids,
            layer.w13_weight,
            layer.w13_weight_scale,
            layer.w2_weight,
            layer.w2_weight_scale,
        )
