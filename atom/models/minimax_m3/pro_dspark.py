# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3-Pro's DSpark block drafter: the MiniMax-M3.1 draft (``dspark.py``)
at Pro's shapes, read out of the target's own checkpoint.

Pro ships its draft inline -- ``language_model.model.dspark.*`` in the target's
shards -- and describes it in the target's text config: ``dflash_config`` holds
the layer count, the ``dspark_*`` fields the rest
(``config._normalize_minimax_m3_pro_dspark_config``). Against the M3.1 draft:

  * MQA: 128 q heads over one K and one V head, all of head_dim 256 (V too,
    where the target's is 192), Gemma q/k norms of 256, neox RoPE on the first
    64 lanes. Dense SwiGLU-OAI FFN of 20480; the confidence head reads
    ``[hidden; markov]``.
  * The pool pages at the target's block, 16, which is the page the per-row
    decode reads: no page-16 re-view, and a table entry is one page.
  * The block rows decode through aiter's gluon paged decode, the target's own
    attention kernel; the M3.1 draft's per-row ASM decode is head_dim 128 only.
  * RoPE is the target's YaRN at the magnitude transformers gives it.

Checkpoint layout (MXFP8 [1, 32] linears, BF16 rest; 5 layers):
  language_model.model.dspark.fc.weight                       [7168, 5 * 7168]
  language_model.model.dspark.{hidden_norm,final_norm}.weight [7168]
  language_model.model.dspark.layers.{0..4}.decoder_layer.
      {input_layernorm,post_attention_layernorm}.weight
      self_attn.q_proj.weight(_scale_inv)                     [32768, 7168]
      self_attn.{k,v}_proj.weight(_scale_inv)                 [256, 7168]
      self_attn.o_proj.weight(_scale_inv)                     [7168, 32768]
      self_attn.{q,k}_norm.weight                             [256]
      mlp.{gate,up}_proj / down_proj                          20480 wide
  language_model.model.dspark.markov_head.markov_w{1,2}.weight [200064, 256]
  language_model.model.dspark.confidence_head.proj.{weight,bias} [1, 7424]
"""

import logging
import math
from typing import ClassVar

import torch

from atom.model_loader.weight_names import WeightsMapper
from atom.model_ops.layernorm import (
    fused_allreduce_gemma_rms_norm,
    fused_allreduce_gemma_rms_norm_mxfp8,
    gemma_rms_norm_mxfp8,
)
from atom.model_ops.minimax_m3.pro_attn import variants
from atom.model_ops.minimax_m3.pro_kernels import attention_aiter
from atom.models.minimax_m3.dspark import (
    M3DSparkAttention,
    M3DSparkDecoderLayer,
    MiniMaxM3DSpark,
)
from atom.models.minimax_m3.pro import _aiter_mx_norm, _native_a8

logger = logging.getLogger("atom")

DRAFT_PREFIX = "language_model.model.dspark."


def _yarn_mscale(factor: float) -> float:
    return 0.1 * math.log(factor) + 1.0 if factor > 1 else 1.0


class _DraftTensorsOnly(WeightsMapper):
    """Drops every tensor of the checkpoint but the draft's, by name, before it
    is read: the checkpoint is the whole target."""

    def _map_name(self, key: str) -> str | None:
        return key if key.startswith(DRAFT_PREFIX) else None


class M3ProDSparkAttention(M3DSparkAttention):
    """MQA over head_dim 256, on the target's page-16 pool and decode kernel."""

    def _rope_scaling(self, config) -> dict | None:
        """The target's YaRN at the config's ``attention_factor``, where aiter
        scales by ``0.1 ln(factor) + 1`` times its own ``attn_factor``."""
        scaling = dict(getattr(config, "rope_scaling", None) or {})
        if scaling.get("rope_type", scaling.get("type")) != "yarn":
            return scaling or None
        attention_factor = scaling.pop("attention_factor", None)
        if attention_factor is not None:
            scaling["attn_factor"] = float(attention_factor) / _yarn_mscale(
                float(scaling["factor"])
            )
        return scaling

    def _bound_cache(self):
        kv = self.kv
        if not kv.bound:
            return None
        return kv.k_cache, kv.v_cache, kv.k_scale, kv.v_scale

    def _block_attention_paged(self, q: torch.Tensor, md, cache) -> torch.Tensor:
        """``[context | block]`` attention as ``B * T`` q_len=1 rows over the
        tables ``build_paged_block`` built (sliding window applied there)."""
        k_cache, v_cache, k_scale, v_scale = cache
        rows = q.shape[0]
        q = q.view(rows, self.num_heads, self.head_dim)
        out = torch.empty_like(q)
        attention_aiter(
            q,
            k_cache,
            v_cache,
            md.dspark_bt16_rows,
            md.dspark_ctx_rows,
            out,
            self.scaling,
            k_scale,
            v_scale,
            value_dim=self.head_dim,
        )
        return out.view(rows, -1)


class M3ProDSparkDecoderLayer(M3DSparkDecoderLayer):
    attention_cls = M3ProDSparkAttention

    def __init__(self, atom_config, config, quant_config, layer_num, prefix=""):
        super().__init__(atom_config, config, quant_config, layer_num, prefix)
        # The target's layer epilogues (pro.py): the TP all-reduce folded into
        # the next Gemma RMSNorm, which also writes the next GEMM's MXFP8 input
        # (the bytes quantize_fp8 of its bf16 output gives), and SwiGLU-OAI
        # writing down_proj's MXFP8 input. Same weights, same bf16 all-reduce,
        # same MXFP8 activation format: 16 -> 11 launches a layer. Gated like
        # the target's fused norms (fuse_mxfp8_quant, not under the diag
        # variant, native MXFP8 linears, an aiter with the MXFP8 norms), else
        # the unfused layer. Decided here because reduce_results is
        # structural.
        v = variants()
        self.fused_epilogues = bool(
            v.fuse_mxfp8_quant
            and not v.diag
            and _native_a8(self.self_attn.qkv_proj, self.mlp.gate_up_proj)
            and _aiter_mx_norm()
        )
        if self.fused_epilogues:
            self.self_attn.o_proj.reduce_results = False
            self.mlp.down_proj.reduce_results = False

    def forward(self, positions, hidden_states, residual=None):
        """Returns the UN-reduced ``(mlp_out, residual)`` when fused: the next
        layer's input norm, or the model's final norm, all-reduces it."""
        if not self.fused_epilogues:
            return super().forward(positions, hidden_states, residual)
        lamport_rows = variants().ar_lamport_rows
        if residual is None:
            residual = hidden_states
            _, xq, xs, _ = gemma_rms_norm_mxfp8(
                hidden_states, None, self.input_layernorm, write_bf16=False
            )
        else:
            _, xq, xs, residual = fused_allreduce_gemma_rms_norm_mxfp8(
                hidden_states, residual, self.input_layernorm, lamport_rows
            )
        h = self.self_attn(positions, (xq, xs))
        _, xq, xs, residual = fused_allreduce_gemma_rms_norm_mxfp8(
            h, residual, self.post_attention_layernorm, lamport_rows
        )
        return self.mlp(xq, xs, quant_down=True), residual


class MiniMaxM3ProDSpark(MiniMaxM3DSpark):
    """The DSpark draft inside a MiniMax-M3-Pro checkpoint."""

    layer_cls = M3ProDSparkDecoderLayer

    def __init__(self, atom_config, layer_offset: int = 0) -> None:
        super().__init__(atom_config, layer_offset)
        logger.info(
            "M3-Pro DSpark draft layers: %s epilogues",
            "fused AR+norm+MXFP8" if self.layers[0].fused_epilogues else "unfused",
        )

    def block_final_norm(self, hidden, residual):
        """The last layer's all-reduce folded into ``final_norm`` when the
        layers run the fused epilogues."""
        if not self.layers[0].fused_epilogues:
            return super().block_final_norm(hidden, residual)
        hidden, _ = fused_allreduce_gemma_rms_norm(hidden, residual, self.final_norm)
        return hidden

    weights_mapper: ClassVar[WeightsMapper] = _DraftTensorsOnly()
    # Drafts its trained block (`dspark_block_size`, 7) whole when
    # --num-speculative-tokens is smaller; the verify reads the first k.
    drafts_trained_block: ClassVar[bool] = True
