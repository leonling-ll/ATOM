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

import math
from typing import ClassVar

import torch

from atom.model_loader.weight_names import WeightsMapper
from atom.model_ops.minimax_m3.pro_kernels import attention_aiter
from atom.models.minimax_m3.dspark import (
    M3DSparkAttention,
    M3DSparkDecoderLayer,
    MiniMaxM3DSpark,
)

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
        tables ``prepare_paged_block`` built (sliding window applied there)."""
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


class MiniMaxM3ProDSpark(MiniMaxM3DSpark):
    """The DSpark draft inside a MiniMax-M3-Pro checkpoint."""

    layer_cls = M3ProDSparkDecoderLayer
    weights_mapper: ClassVar[WeightsMapper] = _DraftTensorsOnly()
    # Drafts its trained block (`dspark_block_size`, 7) whole when
    # --num-speculative-tokens is smaller; the verify reads the first k.
    drafts_trained_block: ClassVar[bool] = True
