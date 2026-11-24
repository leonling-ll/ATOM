# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3-Pro (text): 68 layers, 384 experts, one shared-KV head of 256 dims.

Against M3.1-Flash, only the attention differs (the MoE / dense MLP / norms / loader
are the Flash modules, with Pro's sizes read from the config):

* q_proj (128 heads x 256, TP-sharded), k_proj (1 head x 256, replicated): no v_proj
  -- V is K with the 64 rope dims dropped, so ``o_proj`` takes 128 x 192;
* a 16-head x 128 indexer (``index_q_proj`` / ``index_k_proj``, replicated) whose
  inputs, from layer 2 on, are the state of the layer two back
  (``sparse_indexer_lookahead_*``, source ``pre_ffn_norm``: its residual before the FFN
  norm, here put through the reading layer's input norm), and which scores 16-token
  blocks;
* the four projections stored as one weight ``[q; k; index q; index k]`` (``qkv_proj``):
  up to ``FRONT_ROWS`` rows one GEMM computes them from ``[x; indexer input]``, above it
  ``qk_proj`` / ``index_qk_proj`` (row views of it) run on their own inputs;
* ``index_compress_query`` per layer (softmax-pool compression of the index keys).

The attention math is ``model_ops.minimax_m3.pro_attn`` (torch, correctness first).
"""

import logging
import re
from functools import cache, partial
from typing import ClassVar

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from aiter.rotary_embedding import get_rope
from torch import nn
from transformers import PretrainedConfig

from atom.config import Config, QuantizationConfig
from atom.model_loader.weight_names import WeightsMapper
from atom.model_ops.base_attention import Attention
from atom.model_ops.eplb import enable_expert_count
from atom.model_ops.layernorm import (
    GemmaRMSNorm,
    fused_allreduce_gemma_rms_norm,
    fused_allreduce_gemma_rms_norm_mxfp8,
    fused_allreduce_gemma_rms_norm_mxfp8_writes_out,
    gemma_rms_norm_mxfp8,
)
from atom.model_ops.linear import (
    MinimaxM3QKVParallelLinearWithIndexer,
    RowParallelLinear,
)
from atom.model_ops.minimax_m3.pro_attn import (
    ProSparseAttentionImpl,
    diag_path,
    refresh_variants,
    variants,
)
from atom.model_ops.utils import atom_parameter
from atom.models.minimax_m3 import moe_route_glue
from atom.models.minimax_m3.model import (
    MiniMaxM3DecoderLayer,
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGenerationTextOnly,
    _rope_theta,
)
from atom.models.minimax_m3.pro_mono.config import config_refusal
from atom.models.minimax_m3.pro_mono.route import ProMonoRoute
from atom.models.minimax_m3.pro_mono.runner import ProMonoRunner
from atom.models.minimax_m3.vl import MiniMaxM3SparseForConditionalGenerationVL

logger = logging.getLogger("atom")

# Rows up to which a layer's q, k, index q and index k come from one GEMM over
# ``[x; indexer input]`` (2 x rows): at 8 rows one 16-row MFMA tile holds both halves.
FRONT_ROWS = 16


def _quantized_once(x: torch.Tensor, x_scale: torch.Tensor | None, linears) -> tuple:
    """``x`` (and its scale) for ``linears`` that share it: quantised here, once, when
    every one runs the native group32 FP8 GEMM, which would quantise it the same way
    itself (``quantize_fp8``); as it is otherwise."""
    if x_scale is None and all(
        getattr(m, "native_a8_group_rows", None) is not None for m in linears
    ):
        from atom.model_ops.blockscale import quantize_fp8

        return quantize_fp8(x)
    return x, x_scale


def _native_a8(*linears) -> bool:
    """Whether every one of ``linears`` runs the native group32 FP8 GEMM, i.e. takes an
    MXFP8 ``(x, x_scale)`` (and would ``quantize_fp8`` a bf16 ``x`` itself)."""
    return all(getattr(m, "native_a8_group_rows", None) is not None for m in linears)


@cache
def _aiter_mx_norm() -> bool:
    """Whether this aiter's norm kernels can write MXFP8 (``add_rmsnorm_mxfp8`` and the
    E8M0 mode of ``fused_qk_rmsnorm_group_quant`` come in together); without them the
    fuse_mxfp8_quant norms keep the unfused path."""
    import aiter
    from aiter.dist import communication_op

    ok = hasattr(aiter, "add_rmsnorm_mxfp8")
    logger.info(
        "M3-Pro fuse_mxfp8_quant: aiter MXFP8 norms %s, all-reduce MXFP8 epilogue %s",
        ok,
        hasattr(
            communication_op,
            "tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant",
        ),
    )
    return ok


class ProLookaheadState:
    """The states the lookahead indexers of later layers read.

    Layer ``L >= start`` feeds its indexer from layer ``L - ahead``'s state (see
    ``ProVariants.lookahead_src`` for which one); one forward's layers run in order, so a
    dict of the last ``ahead`` layers is the whole state (no cross-forward lifetime: a
    layer's entry is written before it is read).

    Up to ``FRONT_ROWS`` rows the MXFP8 indexer inputs and attention inputs live in a
    ring of ``ahead + 1`` slots (``front_rows``), slot ``reader % (ahead + 1)``: the
    lookahead norm writes rows ``[T, 2T)`` of its reader's slot, the reader's input norm
    rows ``[0, T)``, and the reader's front GEMM reads the slot's ``2T`` rows as one A.
    Fixed buffers, so a CUDA graph reads and writes the same addresses every replay.
    """

    def __init__(self, config: PretrainedConfig):
        sparse_cfg = config.sparse_attention_config
        self.ahead = int(sparse_cfg.get("sparse_indexer_lookahead_num_layers") or 0)
        self.start = int(sparse_cfg.get("sparse_indexer_lookahead_start_layer") or 0)
        self.source = sparse_cfg.get("sparse_indexer_lookahead_source")
        self.ffn_in: dict[int, torch.Tensor] = {}
        self.resid: dict[int, torch.Tensor] = {}
        # every layer's input_layernorm (the decoder layers register them): with them the
        # ``resid_in`` source is normed when it is put, keyed by the layer that reads it
        self.norms: dict[int, GemmaRMSNorm] = {}
        # and their attentions' indexer projection: a ``resid_in`` entry for a reader whose
        # projection is native group32 FP8 is put as MXFP8 ``(x, x_scale)`` under fuse_mxfp8_quant
        self.index_linears: dict[int, tuple[nn.Module, ...]] = {}
        self.resid_in: dict[int, torch.Tensor | tuple[torch.Tensor, torch.Tensor]] = {}
        self._ring: tuple[torch.Tensor, torch.Tensor] | None = None

    def front_rows(
        self, reader: int, rows: int, like: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``reader``'s ring slot as ``2 * rows`` MXFP8 rows ``(q, q_scale)`` (rows
        ``[0, rows)`` its input, ``[rows, 2 * rows)`` its indexer input), or None past
        ``FRONT_ROWS``. ``like``: an activation of the layer (hidden size, device)."""
        if not 0 < rows <= FRONT_ROWS:
            return None
        hidden = like.shape[-1]
        if (
            self._ring is None
            or self._ring[0].shape[-1] != hidden
            or self._ring[0].device != like.device
        ):
            # allocated once, by the first eager forward of at most FRONT_ROWS rows
            # (a capture is preceded by an eager forward of its size)
            shape = (self.ahead + 1, 2 * FRONT_ROWS, hidden)
            self._ring = (
                torch.empty(shape, dtype=torch.float8_e4m3fn, device=like.device),
                torch.empty(
                    (*shape[:2], hidden // 32),
                    dtype=torch.float8_e8m0fnu,
                    device=like.device,
                ),
            )
        slot = reader % (self.ahead + 1)
        return self._ring[0][slot, : 2 * rows], self._ring[1][slot, : 2 * rows]

    @staticmethod
    def front_input(
        front: tuple[torch.Tensor, torch.Tensor] | None,
        rows: int,
        x: tuple[torch.Tensor, torch.Tensor | None],
        index_src: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """The A of a layer's one front GEMM, read where the norms wrote it: None (two
        GEMMs) unless ``x``, the input norm's MXFP8 ``(q, q_scale)``, is rows ``[0, rows)``
        of ``front`` (``front_rows``); then those rows when the indexer reads ``x`` too
        (``index_src`` None), all ``2 * rows`` when ``index_src`` is the slot's rows
        ``[rows, 2 * rows)``, None for any other ``index_src``."""

        def written_at(t, q, q_scale) -> bool:
            return (
                isinstance(t, tuple)
                and t[1] is not None
                and t[0].data_ptr() == q.data_ptr()
                and t[1].data_ptr() == q_scale.data_ptr()
            )

        if front is None or not written_at(x, *front):
            return None
        q, q_scale = front
        if index_src is None:
            return q[:rows], q_scale[:rows]
        if written_at(index_src, q[rows:], q_scale[rows:]):
            return front
        return None

    @staticmethod
    def _normed(norm: GemmaRMSNorm, x: torch.Tensor) -> torch.Tensor:
        """``x`` through ``norm`` as a plain (torch) Gemma RMSNorm: the module's own call may be
        the fused all-reduce / quantising variant, which this is not."""
        return GemmaRMSNorm.forward_static(
            norm.weight.data, norm.variance_epsilon, x, None
        )

    def source_for(
        self, layer: int, norm: GemmaRMSNorm | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """The indexer input of ``layer`` (``None``: its own attention input), bf16 or
        MXFP8 ``(x, x_scale)``. ``norm`` is the layer's input_layernorm, which the
        ``resid_in`` source goes through."""
        v = variants()
        if not (self.ahead and v.lookahead and layer >= self.start):
            return None
        if v.lookahead_src == "ffn_in":
            return self.ffn_in.get(layer - self.ahead)
        if v.lookahead_src == "resid_in" and layer in self.resid_in:
            return self.resid_in.pop(layer)
        src = self.resid.get(layer - self.ahead)
        if src is not None and v.lookahead_src == "resid_in":
            src = self._normed(norm, src)
        return src

    def diag_sources(
        self, layer: int, norm: GemmaRMSNorm | None = None
    ) -> dict[str, torch.Tensor | None]:
        """Every candidate lookahead input of ``layer`` (the ``pro_diag`` developer tool)."""
        if not (self.ahead and layer >= self.start):
            return {}
        resid = self.resid.get(layer - self.ahead)
        return {
            "ffn": self.ffn_in.get(layer - self.ahead),
            "res": resid,
            "resn": (
                None if resid is None or norm is None else self._normed(norm, resid)
            ),
        }

    def put(self, layer: int, ffn_in: torch.Tensor, resid: torch.Tensor) -> None:
        if not self.ahead:
            return
        # Copies: the next layer's fused add updates ``resid`` in place, and the FFN
        # may reuse ``ffn_in``; the lookahead reads them two layers later.
        v = variants()
        reader = layer + self.ahead
        if v.diag:
            self.resid[layer] = resid.clone()
            self.ffn_in[layer] = ffn_in.clone()
        elif v.lookahead_src == "resid_in" and self.norms:
            # the reader's input norm now, one kernel writing a new tensor: no copy
            if v.lookahead and reader >= self.start and reader in self.norms:
                norm = self.norms[reader]
                linears = self.index_linears.get(reader)
                if (
                    v.fuse_mxfp8_quant
                    and linears is not None
                    and _native_a8(*linears)
                    and _aiter_mx_norm()
                ):
                    rows = resid.shape[0]
                    front = self.front_rows(reader, rows, resid)
                    _, q, q_scale, _ = gemma_rms_norm_mxfp8(
                        resid,
                        None,
                        norm,
                        write_bf16=False,
                        q_out=None if front is None else front[0][rows:],
                        q_scale_out=None if front is None else front[1][rows:],
                    )
                    self.resid_in[reader] = (q, q_scale)
                else:
                    self.resid_in[reader] = norm.forward_cuda(resid)
        elif v.lookahead_src in ("resid", "resid_in"):
            self.resid[layer] = resid.clone()
        else:
            self.ffn_in[layer] = ffn_in.clone()
        stale = layer - self.ahead
        self.ffn_in.pop(stale, None)
        self.resid.pop(stale, None)


class MiniMaxM3ProAttention(nn.Module):
    """Pro's MQA attention with a 16-head lightning indexer."""

    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        cache_config: str = "bf16",
        index_cache_config: str = "auto",
    ) -> None:
        super().__init__()
        if index_cache_config not in ("auto", "bf16", "fp8", "nvfp4"):
            raise NotImplementedError(
                "MiniMax-M3-Pro keeps a bf16, fp8 or nvfp4 index cache, got "
                f"--index-cache-dtype {index_cache_config}"
            )
        sparse_cfg = config.sparse_attention_config
        self.layer_num = layer_id
        self.hidden_size = config.hidden_size
        tp = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        if self.total_num_heads % tp:
            raise ValueError("num_attention_heads must be divisible by the TP size")
        self.num_heads = self.total_num_heads // tp
        if config.num_key_value_heads != 1:
            raise ValueError("MiniMax-M3-Pro is MQA: num_key_value_heads must be 1")
        self.head_dim = config.head_dim
        rotary_dim = int(self.head_dim * getattr(config, "partial_rotary_factor", 1.0))
        self.value_dim = self.head_dim - rotary_dim
        declared = sparse_cfg.get("value_head_dim") or self.value_dim
        if declared != self.value_dim:
            raise ValueError(
                f"value_head_dim {declared} is not head_dim {self.head_dim} minus the "
                f"rope dims {rotary_dim}"
            )
        self.num_idx_heads = sparse_cfg["sparse_num_index_heads"]
        self.idx_head_dim = sparse_cfg["sparse_index_dim"]
        self.scaling = self.head_dim**-0.5

        # q (TP-sharded) + the one K head (replicated, as QKV replicates a KV head
        # below the TP size), no V, then index q + index k (every index head on every
        # rank): one weight, [q; k; index q; index k]
        self.qkv_proj = MinimaxM3QKVParallelLinearWithIndexer(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            1,
            self.num_idx_heads,
            self.idx_head_dim,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
            v_head_size=0,
            replicate_index_heads=True,
        )
        self.qk_sizes = [self.num_heads * self.head_dim, self.head_dim]
        self.index_qk_sizes = [
            self.num_idx_heads * self.idx_head_dim,
            self.idx_head_dim,
        ]
        self._views = None
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.value_dim,
            self.hidden_size,
            bias=False,
            reduce_results=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.index_q_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        self.index_k_norm = GemmaRMSNorm(self.idx_head_dim, eps=config.rms_norm_eps)
        # The learned query of the index keys' softmax-pool compression.
        self.index_compress_query = atom_parameter(
            torch.empty(self.idx_head_dim, dtype=torch.bfloat16)
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=_rope_theta(config),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            1,
            kv_cache_dtype=cache_config,
            layer_num=layer_id,
            use_mla=False,
            rotary_emb=self.rotary_emb,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            prefix=f"{prefix}.attn",
            impl_cls=ProSparseAttentionImpl,
            index_q_norm=self.index_q_norm,
            index_k_norm=self.index_k_norm,
            index_compress_query=self.index_compress_query,
            num_index_heads=self.num_idx_heads,
            index_head_dim=self.idx_head_dim,
            value_head_dim=self.value_dim,
            rotary_dim=rotary_dim,
            topk=sparse_cfg["sparse_topk_blocks"],
            init_blocks=sparse_cfg.get("sparse_init_block", 0),
            local_blocks=sparse_cfg.get("sparse_local_block", 0),
            block_size=sparse_cfg["sparse_block_size"],
        )
        # o_proj takes MXFP8 rows (a group32 FP8 GEMM, not its BF16 fallback): the decode
        # attention may write them itself (ProSparseAttentionImpl._mxfp8_out)
        self.attn.impl.mx_out = _native_a8(self.o_proj) and not self.o_proj.runs_bf16

    @property
    def qk_proj(self) -> nn.Module:
        """q + k: rows ``[0, sum(qk_sizes))`` of ``qkv_proj``."""
        return self._row_views()[0]

    @property
    def index_qk_proj(self) -> nn.Module:
        """index q + index k: the rest of ``qkv_proj``."""
        return self._row_views()[1]

    def _row_views(self) -> tuple[nn.Module, nn.Module]:
        # Built at the first forward, once the loader has shuffled the weight and its
        # scale (it does so after this module's own hooks); a tuple, so not submodules.
        if self._views is None:
            n = sum(self.qk_sizes)
            self._views = (
                self.qkv_proj.make_row_view(0, n),
                self.qkv_proj.make_row_view(n, sum(self.index_qk_sizes)),
            )
        return self._views

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None = None,
        index_src: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        diag_srcs: dict[str, torch.Tensor | None] | None = None,
        front: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """``front``: MXFP8 ``[x; indexer input]`` rows (``ProLookaheadState.front_input``;
        ``x`` alone when the indexer reads it) for one ``qkv_proj`` GEMM: q, k from the
        first ``n`` rows, index q, index k from the last ``n``."""
        out = self._attend(
            positions, hidden_states, hidden_states_scale, index_src, diag_srcs, front
        )
        if out.dtype == torch.float8_e4m3fn:
            # the decode attention wrote o_proj's MXFP8 input (impl.mx_out)
            return self.o_proj(out, x_scale=self.attn.impl.take_mx_scale())
        return self.o_proj(out)

    def forward_core(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor | None = None,
        index_src: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None = None,
        diag_srcs: dict[str, torch.Tensor | None] | None = None,
        front: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """The attention up to o_proj as bf16 [M, heads x 192], what the mono layer
        kernel reads (``front`` as in ``forward``). The Triton / aiter decode
        attention writes this call's rows as bf16 where ``forward``'s would be
        o_proj's MXFP8 input (``ProSparseAttentionImpl.bf16_out``). The day0
        decode's epilogue still writes E4M3 from ``day0_decode_rows`` rows, so mono
        refuses a row limit that reaches it (``pro_mono.route.variants_refusal``)."""
        impl = self.attn.impl
        impl.bf16_out = True
        try:
            return self._attend(
                positions,
                hidden_states,
                hidden_states_scale,
                index_src,
                diag_srcs,
                front,
            )
        finally:
            impl.bf16_out = False

    def _attend(
        self, positions, hidden_states, hidden_states_scale, index_src, diag_srcs, front
    ) -> torch.Tensor:
        """The projections and the attention: o_proj's input, bf16, or the decode
        attention's E4M3 rows with their scales in ``attn.impl`` (``take_mx_scale``)."""
        if front is not None:
            n = hidden_states.shape[0]
            y = self.qkv_proj(*front)
            q, k = y[:n, : sum(self.qk_sizes)].split(self.qk_sizes, dim=-1)
            iq, ik = y[-n:, sum(self.qk_sizes) :].split(self.index_qk_sizes, dim=-1)
        else:
            own = [self.qk_proj]
            if index_src is None:
                own.append(self.index_qk_proj)
            x, xs = _quantized_once(hidden_states, hidden_states_scale, own)
            q, k = self.qk_proj(x, x_scale=xs).split(self.qk_sizes, dim=-1)
            if index_src is None:
                iq, ik = self._index_qk(x, xs)
            else:
                ix, ixs = (
                    index_src
                    if isinstance(index_src, tuple)
                    else _quantized_once(index_src, None, [self.index_qk_proj])
                )
                iq, ik = self._index_qk(ix, ixs)
        if diag_srcs is not None:
            # raw indexer projections of every candidate input, for `pro_diag`
            own = (
                (iq, ik)
                if index_src is None
                else self._index_qk(hidden_states, hidden_states_scale)
            )
            raw = {"own": own}
            for name, src in diag_srcs.items():
                if src is not None:
                    raw[name] = self._index_qk(src)
            self.attn.impl._diag_raw = raw
        # the impl reads the four projections where they are (column slices of the two
        # GEMM outputs): index k rides in ``qkv``
        return self.attn(q, k, iq, positions, qkv=ik)

    def _index_qk(self, x, x_scale=None) -> tuple[torch.Tensor, torch.Tensor]:
        return tuple(
            self.index_qk_proj(x, x_scale=x_scale).split(self.index_qk_sizes, dim=-1)
        )


class MiniMaxM3ProDecoderLayer(MiniMaxM3DecoderLayer):
    """A Flash decoder layer with Pro's attention and the lookahead hand-over."""

    # mono phase 1's gate (``pro_mono.route``), shared by the model's layers; None (a
    # layer built without one too) runs production's segment only
    _mono: ProMonoRoute | None = None

    def __init__(
        self,
        *args,
        state: ProLookaheadState,
        mono: ProMonoRoute | None = None,
        **kwargs,
    ) -> None:
        # plain attributes, set before `nn.Module.__init__` runs inside super()
        self._pro_state = state
        self._mono = mono
        super().__init__(*args, **kwargs)
        state.norms[self.layer_num] = self.input_layernorm
        state.index_linears[self.layer_num] = (self.self_attn.qkv_proj,)

    def _make_attention(
        self,
        config,
        layer_num,
        prefix,
        cache_config,
        index_cache_config,
        quant_config,
    ):
        return MiniMaxM3ProAttention(
            config=config,
            layer_id=layer_num,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
            cache_config=cache_config,
            index_cache_config=index_cache_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        capture_aux: bool = False,
    ):
        # Plain (unfused-quant) norms: the lookahead needs the bf16 FFN input of
        # every layer, which a fused norm + per-token-FP8 quant does not produce.
        # fuse_mxfp8_quant: the norms whose output a native group32 FP8 GEMM reads
        # also write it as MXFP8 (the same bytes its quantize_fp8 would), the bf16
        # output kept where anything else reads it.
        v = variants()
        fuse = v.fuse_mxfp8_quant and not v.diag
        mx_norm = fuse and _aiter_mx_norm()
        attn = self.self_attn
        state = self._pro_state
        fuse_attn_in = mx_norm and _native_a8(attn.qkv_proj)
        rows = hidden_states.shape[0]
        # up to FRONT_ROWS rows the input norm writes its MXFP8 rows into the ring slot
        # the lookahead norm filled two layers back, for one front GEMM over both
        front = (
            state.front_rows(self.layer_num, rows, hidden_states)
            if fuse_attn_in
            and not attn.qkv_proj.runs_bf16
            and v.lookahead_src == "resid_in"
            and fused_allreduce_gemma_rms_norm_mxfp8_writes_out(
                rows, self.input_layernorm, v.ar_lamport_rows
            )
            else None
        )
        x_out = (None, None) if front is None else (front[0][:rows], front[1][:rows])
        hidden_states_scale = None
        if residual is None:
            residual = hidden_states
            if fuse_attn_in:
                _, hidden_states, hidden_states_scale, _ = gemma_rms_norm_mxfp8(
                    hidden_states,
                    None,
                    self.input_layernorm,
                    write_bf16=False,
                    q_out=x_out[0],
                    q_scale_out=x_out[1],
                )
            else:
                hidden_states = self.input_layernorm(hidden_states)
        elif fuse_attn_in:
            _, hidden_states, hidden_states_scale, residual = (
                fused_allreduce_gemma_rms_norm_mxfp8(
                    hidden_states,
                    residual,
                    self.input_layernorm,
                    v.ar_lamport_rows,
                    q_out=x_out[0],
                    q_scale_out=x_out[1],
                )
            )
        else:
            hidden_states, residual = fused_allreduce_gemma_rms_norm(
                hidden_states, residual, self.input_layernorm
            )
        aux_hidden_state = residual.clone() if capture_aux else None

        # A MoE layer of a decode / verify forward within the latched row limit
        # runs o_proj .. the FFN on the mono kernel (TP-uniform: ``take``).
        mono = (
            self._mono.take(self.layer_num, rows)
            if self._mono is not None and self.is_moe_layer
            else None
        )
        index_src = state.source_for(self.layer_num, self.input_layernorm)
        attn_in = {
            "positions": positions,
            "hidden_states": hidden_states,
            "hidden_states_scale": hidden_states_scale,
            "index_src": index_src,
            "diag_srcs": (
                state.diag_sources(self.layer_num, self.input_layernorm)
                if v.diag
                else None
            ),
            "front": state.front_input(
                front, rows, (hidden_states, hidden_states_scale), index_src
            ),
        }
        if mono is not None:
            # production's segment of this o_proj is check mode's reference
            core = attn.forward_core(**attn_in)
            hidden_states, residual = mono.segment(
                self,
                core,
                residual,
                lambda res: self._segment(attn.o_proj(core), res, v, mx_norm, fuse),
            )
        else:
            # Neither the attention's inputs nor o_proj's output is held across the
            # FFN: a prefill's [tokens, hidden] rows would raise the warmup peak the
            # KV pool is sized from.
            o = attn(**attn_in)
            del attn_in, index_src, hidden_states, hidden_states_scale
            x, x8, x8_scale, residual, routes = self._post_attention(
                o, residual, v, mx_norm
            )
            del o
            hidden_states = self._ffn(x, x8, x8_scale, fuse, routes)
        if aux_hidden_state is not None:
            return hidden_states, residual, aux_hidden_state
        return hidden_states, residual

    def _segment(self, hidden_states, residual, v, mx_norm: bool, fuse: bool):
        """Production's post-attention segment on o_proj's output (``forward``'s
        two steps): ``(ffn_out, residual, x, x8, x8_scale)``."""
        x, x8, x8_scale, residual, routes = self._post_attention(
            hidden_states, residual, v, mx_norm
        )
        return self._ffn(x, x8, x8_scale, fuse, routes), residual, x, x8, x8_scale

    def _post_attention(self, hidden_states, residual, v, mx_norm: bool):
        """AR #1 + the post-attention norm and the lookahead put, as
        ``moe_route_glue``'s one launch on a MoE layer it serves:
        ``(x, x8, x8_scale, residual, routes)``, ``routes`` the glue's outputs
        (else None) for ``_ffn``."""
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        if (
            self.is_moe_layer
            and v.moe_route_glue
            and mx_norm
            and moe_route_glue.applies(ffn, hidden_states.shape[0])
            and _native_a8(ffn.shared_experts.gate_up_proj)  # it reads the MXFP8 copy
        ):
            # all-reduce + norm (+ MXFP8 copy) + router + top-k + sort in one launch
            r = moe_route_glue.route(
                hidden_states, residual, self.post_attention_layernorm, ffn
            )
            self._pro_state.put(self.layer_num, r["normed"], r["residual_out"])
            return r["normed"], r["x_q"], r["x_scale"], r["residual_out"], r
        x, x8, x8_scale, residual = self._post_attention_norm(
            hidden_states, residual, v, mx_norm
        )
        self._pro_state.put(self.layer_num, x, residual)
        return x, x8, x8_scale, residual, None

    def _post_attention_norm(self, hidden_states, residual, v, mx_norm: bool):
        """AR #1 + the post-attention norm: ``(x, x8, x8_scale, residual)``,
        the MXFP8 pair None where the FFN's first GEMM does not read it."""
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        # the MoE's router and routed experts read the bf16 output; its shared expert
        # (or the dense MLP) can read the MXFP8 one
        ffn_gate_up = (
            None
            if self.is_moe_layer and ffn.shared_experts is None
            else (ffn.shared_experts if self.is_moe_layer else ffn).gate_up_proj
        )
        if mx_norm and ffn_gate_up is not None and _native_a8(ffn_gate_up):
            return fused_allreduce_gemma_rms_norm_mxfp8(
                hidden_states,
                residual,
                self.post_attention_layernorm,
                v.ar_lamport_rows,
            )
        hidden_states, residual = fused_allreduce_gemma_rms_norm(
            hidden_states, residual, self.post_attention_layernorm
        )
        return hidden_states, None, None, residual

    def _ffn(
        self, hidden_states, ffn_x, ffn_x_scale, fuse: bool, routes=None
    ) -> torch.Tensor:
        """The FFN (its TP partial) on the post-attention norm's outputs, or on
        ``moe_route_glue``'s ``routes``."""
        ffn = self.block_sparse_moe if self.is_moe_layer else self.mlp
        if routes is not None:
            return ffn.forward_glue(routes, quant_down=True)
        if not fuse:
            return ffn(hidden_states)
        if self.is_moe_layer:
            return ffn(
                hidden_states,
                shared_input=None if ffn_x is None else (ffn_x, ffn_x_scale),
                quant_down=True,
            )
        if ffn_x is not None:
            return ffn(ffn_x, ffn_x_scale, quant_down=True)
        return ffn(hidden_states, quant_down=True)


def _mono_route(atom_config: Config, state: ProLookaheadState, layers) -> ProMonoRoute:
    """The model's mono phase-1 gate; ``layers()``: its decoder layers (the
    runner binds them on its first step)."""

    def create():
        return ProMonoRunner(layers(), state, route.check, route.routed_act)

    route = ProMonoRoute(
        config_refusal(atom_config),
        create,
        read=lambda: (refresh_variants(force=True), _aiter_mx_norm()),
        group=lambda: get_tp_group().cpu_group,
    )
    return route


_TEXT_QK = re.compile(r"(model\.layers\.\d+\.self_attn\.)([qk])_proj\.")


class _TextQKNames(WeightsMapper):
    """The text layers' ``self_attn.{q,k}_proj`` renamed ``attn_{q,k}_proj``: a packed
    mapping matches substrings, and the vision tower's attention uses the same names."""

    def _map_name(self, key: str) -> str | None:
        return _TEXT_QK.sub(r"\1attn_\2_proj.", key)


class MiniMaxM3ProForCausalLM(MiniMaxM3SparseForCausalLM):
    """The text model: Flash's ``MiniMaxM3Model`` over Pro's decoder layers."""

    # q / k into ``qkv_proj`` under the names ``weights_mapper`` gives the text layers'
    # (the vision tower's q/k/v_proj keep theirs); index q / k after them.
    packed_modules_mapping: ClassVar[dict[str, tuple[str, str | int]]] = {
        ".index_q_proj": (".qkv_proj", "index_q"),
        ".index_k_proj": (".qkv_proj", "index_k"),
        ".attn_q_proj": (".qkv_proj", "q"),
        ".attn_k_proj": (".qkv_proj", "k"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }
    weights_mapper: ClassVar[WeightsMapper] = _TextQKNames()

    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        text = getattr(atom_config.hf_config, "text_config", atom_config.hf_config)
        state = ProLookaheadState(text)
        mono = _mono_route(atom_config, state, lambda: self.model.layers)
        super().__init__(
            atom_config,
            prefix=prefix,
            layer_type=partial(MiniMaxM3ProDecoderLayer, state=state, mono=mono),
        )
        if variants().moe_route_glue:
            # the op's IPC buffers: collective, and before any decode graph is captured
            moe_route_glue.build(
                text.hidden_size, text.num_local_experts, text.num_experts_per_tok
            )
        v = refresh_variants(force=True)
        if v.expert_count and get_tensor_model_parallel_rank() == 0:
            enable_expert_count(
                [
                    layer.block_sparse_moe.experts
                    for layer in self.model.layers
                    if getattr(layer, "is_moe_layer", False)
                ],
                diag_path(f"{v.diag_tag}_experts.jsonl"),
            )


class MiniMaxM3ProForConditionalGenerationTextOnly(
    MiniMaxM3SparseForConditionalGenerationTextOnly
):
    """Text-only view of the Pro checkpoint (vision tower and DSpark head skipped)."""

    packed_modules_mapping = MiniMaxM3ProForCausalLM.packed_modules_mapping
    weights_mapper = MiniMaxM3ProForCausalLM.weights_mapper
    skip_weight_prefixes: ClassVar[list[str]] = [
        *MiniMaxM3SparseForConditionalGenerationTextOnly.skip_weight_prefixes,
        # the embedded DSpark draft (5 layers + Markov / confidence heads)
        "language_model.model.dspark.",
    ]

    def __init__(self, atom_config: Config, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = atom_config.hf_config
        self.language_model = MiniMaxM3ProForCausalLM(
            atom_config=atom_config, prefix=prefix
        )
        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors
        )


class MiniMaxM3ProForConditionalGenerationVL(
    MiniMaxM3SparseForConditionalGenerationVL,
    MiniMaxM3ProForConditionalGenerationTextOnly,
):
    """Pro with its vision tower: Flash's tower, projector and patch merger (``vl``),
    sized by Pro's vision config, over Pro's text model. Without image input
    (``image_input_refusal``) it is the text-only model."""

    packed_modules_mapping = MiniMaxM3ProForCausalLM.packed_modules_mapping
    weights_mapper = MiniMaxM3ProForCausalLM.weights_mapper
    skip_weight_prefixes: ClassVar[list[str]] = ["language_model.model.dspark."]
    # The packed mapping has no q/k/v renames: Pro's text attention keeps those names.
    fused_vision_qkv: ClassVar[bool] = False
