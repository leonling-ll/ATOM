# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3.1 DSpark semi-autoregressive block drafter for ATOM.

The same algorithm as the Kimi-K3 draft (``kimi_k3_dspark.py``) -- the target's
aux hidden states are projected once into a context, every draft layer writes
its own K/V rows for that context into a paged cache, and one non-causal block
pass over ``[anchor, MASK, ...]`` drafts the whole block before a Markov head
samples it left to right -- on a MiniMax-M3 backbone instead of MLA. The
reference is NeMo AutoModel ``speculative/dspark/draft_minimax_m3.py``:

  * Attention is plain GQA (64 q / 4 kv heads, head_dim 128) with per-head
    Gemma RMSNorm on Q and K, partial neox RoPE on the first 64 lanes, and no
    norm or RoPE on V. The K/V of the context and of the block are projected
    separately and attended together, non-causally: every block position sees
    the whole context and the whole block.
  * Every norm is Gemma style, ``x * (1 + w)``; the MLP is dense SwiGLU-OAI.
    There is no indexer and no MoE -- both belong to the target.
  * The context is ``hidden_norm(fc(concat(taps)))`` and goes straight into each
    layer's K/V projection, never through ``input_layernorm``.

KV lives in a sibling pool of its own (an MHA draft cannot share the target's
latent pool the way K3's MLA draft does): ``draft_kv_builder`` resolves this
config to the MHA backend, which owns a pool laid out like the target's and
binds it onto each layer's ``KVCacheBinding``. Both writes -- context rows and the
block's own rows -- go through ``triton_fused_norm_rope_cache`` with an explicit
slot mapping, the kernel the target's dense layers use for the same norm, RoPE
and per-token fp8 cache format.

The block attention is ``B * T`` independent q_len=1 paged decode rows --
non-causal by construction -- over each request's ``[context | block]``,
through the per-row ASM / Gluon decode the target's sparse layers run and a
page-16 table ``build_paged_block`` builds outside the graph; the block's own K/V
are written into the cache first, except at positions past the request's
allocated pages. No host sync and fixed shapes, so the block pass is CUDA-graph
captured like the other drafts. The per-row decode takes one KV head per rank,
so the draft needs TP >= its KV head count (4) to draft; a P/D producer only
writes context K/V, so it may hold several (all four on a TP1 PP stage).

Checkpoint layout (``<draft>/dspark``, MXFP8 [1, 32] linears, BF16 rest):
  language_model.model.dspark.fc.weight                       [6144, 5 * 6144]
  language_model.model.dspark.{hidden_norm,final_norm}.weight [6144]
  language_model.model.dspark.layers.{0..4}.decoder_layer.
      {input_layernorm,post_attention_layernorm}.weight
      self_attn.{q,k,v,o}_proj.weight(_scale_inv)             -> qkv_proj
      self_attn.{q,k}_norm.weight                             [128]
      mlp.{gate,up,down}_proj.weight(_scale_inv)              -> gate_up_proj
  language_model.model.dspark.markov_head.markov_w{1,2}.weight [200064, 256]
  language_model.model.dspark.confidence_head.proj.{weight,bias} [1, 6400]
The draft ships no embedding or LM head; both are the target's.
"""

from typing import TYPE_CHECKING, ClassVar

import torch
import triton
import triton.language as tl
from aiter import QuantType, dtypes
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from aiter.rotary_embedding import get_rope
from aiter.tuned_gemm import tgemm
from torch import nn

from atom.config import QuantizationConfig
from atom.kv_transfer.disaggregation import kv_config_has_producer
from atom.model_ops.blockscale import native_quant_linear
from atom.model_ops.dspark_block_meta import dspark_block_meta
from atom.model_ops.dspark_markov_stage1 import (
    BLOCK_K,
    BLOCK_V,
    markov_launch_geometry,
    markov_logits_tile_of,
)
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.linear import QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from atom.model_ops.minimax_m3.sparse_attn import (
    ASM_PAGE_SIZE,
    sparse_pa_per_row,
    to_page16_shuffle,
)
from atom.model_ops.tile_argmax import argmax_lowest_id
from atom.model_ops.triton_fused_qkv_norm_rope_cache import (
    triton_fused_norm_rope_cache,
)
from atom.models.dspark_draft import (
    AllocationCap,
    DSparkConfidenceHead,
    DSparkDraftModel,
    DSparkGQADecoderLayer,
    KVCacheBinding,
    draft_quant_config,
    linear_out,
)
from atom.models.minimax_m3.model import MiniMaxM3MLP
from atom.models.minimax_m3.packed_argmax import (
    EMPTY,
    EMPTY_K,
    pack_max_id,
    ranks_best,
    unpack_id,
)
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

if TYPE_CHECKING:
    from atom.config import Config


def _draft_rope_theta(config) -> float:
    # transformers 5 moves rope_theta into rope_parameters on the M3 VL text
    # config class this draft loads as; the plain-dict target keeps it flat.
    params = getattr(config, "rope_parameters", None) or {}
    theta = params.get("rope_theta") or getattr(config, "rope_theta", None)
    if not theta:
        raise ValueError("MiniMax-M3 DSpark draft config names no rope_theta.")
    return float(theta)


class M3DSparkAttention(nn.Module):
    """GQA attention for one draft layer, with a second entry point for the
    target-derived context rows."""

    def __init__(
        self,
        atom_config: "Config",
        config,
        quant_config: QuantizationConfig,
        layer_num: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_heads % tp_size:
            raise ValueError(
                f"DSpark draft has {self.total_num_heads} attention heads, which "
                f"is not divisible by TP size {tp_size}."
            )
        # A P/D producer never runs the block pass
        # (`Drafter.context_kv_only`); it only writes context K/V.
        producer = kv_config_has_producer(atom_config.kv_transfer_config)
        if self.total_num_kv_heads > tp_size and not producer:
            raise ValueError(
                "The MiniMax-M3.1 DSpark draft decodes its block per row with one "
                f"KV head per rank, so it needs TP >= {self.total_num_kv_heads} "
                f"(its KV head count); got TP {tp_size}."
            )
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.kv_cache_dtype = atom_config.kv_cache_dtype

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            reduce_results=True,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        rotary_dim = int(
            getattr(config, "rotary_dim", 0)
            or self.head_dim * getattr(config, "partial_rotary_factor", 1.0)
        )
        # neox (half-split) rotation of the first `rotary_dim` lanes -- the
        # gpt_oss rotary the reference uses, and what the fused kernel applies.
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=rotary_dim,
            max_position=config.max_position_embeddings,
            base=_draft_rope_theta(config),
            rope_scaling=self._rope_scaling(config),
        )
        self.kv = KVCacheBinding(layer_num, page_tokens=ASM_PAGE_SIZE)

    def _rope_scaling(self, config) -> dict | None:
        """The ``rope_scaling`` handed to ``get_rope``: the M3.1 draft's RoPE is
        unscaled."""
        return None

    def _bound_cache(self):
        """This layer's KV cache in the layout every access of it uses.

        The block reads through the page-16 SHUFFLE views the per-row decode
        kernels take, so the cache must also be WRITTEN through them: a page-128
        SHUFFLE view puts the 128 in-page rows inside the head-dim tiles, and
        reading that memory as eight 16-row pages scrambles it. Slot numbers are
        the same in both (page * 128 + off == (page * 8 + off // 16) * 16 +
        off % 16), only the view differs.
        """
        kv = self.kv
        if not kv.bound:
            return None
        return to_page16_shuffle(kv.k_cache, kv.v_cache, kv.k_scale, kv.v_scale)

    def _norm_rope_write(self, q, k, v, num_heads, positions, slot_mapping, cache):
        k_cache, v_cache, k_scale, v_scale = cache
        return triton_fused_norm_rope_cache(
            q,
            k,
            v,
            positions,
            q_norm=self.q_norm,
            k_norm=self.k_norm,
            rotary_emb=self.rotary_emb,
            num_heads=num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            k_cache=k_cache,
            v_cache=v_cache,
            k_scale=k_scale,
            v_scale=v_scale,
            slot_mapping=slot_mapping,
            kv_cache_dtype=self.kv_cache_dtype,
        )

    def write_context_kv(
        self,
        ctx_hidden: torch.Tensor,  # [N, hidden] projected + normed target hidden
        positions: torch.Tensor,  # [N]
        slot_mapping: torch.Tensor,  # [N] the target forward's slots
    ) -> None:
        """K/V rows for the target context: k_norm + RoPE on K, V as is.

        The q third of the fused projection is computed and dropped; the kernel
        writes K/V only when told it has zero query heads.
        """
        cache = self._bound_cache()
        if cache is None:
            return
        qkv = linear_out(self.qkv_proj(ctx_hidden))
        _, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        self._norm_rope_write(k, k, v, 0, positions, slot_mapping, cache)

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor):
        """The block pass: non-causal attention over ``[context | block]``.

        Write the block's K/V at the block slots the proposer masked to the
        request's allocated pages -- an unallocated column reads as page 0, a
        page another request owns (under prefix caching, a shared prefix) --
        then decode per row.
        """
        if self.num_kv_heads != 1:
            # Before any K/V is written: only a P/D producer, which never
            # drafts, is built with more than one KV head per rank.
            raise RuntimeError(
                f"The block pass decodes one KV head per rank; this rank holds "
                f"{self.num_kv_heads} of {self.total_num_kv_heads}"
            )
        # An (x, x_scale) pair is the MXFP8 input a fused norm already wrote.
        x, x_scale = (
            hidden_states if isinstance(hidden_states, tuple) else (hidden_states, None)
        )
        qkv = linear_out(self.qkv_proj(x, x_scale=x_scale))
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        cache = self._bound_cache()
        if cache is None:
            # warmup_model() runs before the pool exists; the pass only sizes
            # activations then, so shapes are all that must be right.
            out = q.new_zeros(q.shape[0], self.q_size)
            return linear_out(self.o_proj(out))

        md = get_forward_context().attn_metadata
        # The block's K/V go into the cache at the block's slots (positions
        # past the allocated pages are -1, dropped), then every block position
        # decodes as its own q_len=1 row.
        q, _ = self._norm_rope_write(
            q, k, v, self.num_heads, positions, md.slot_mapping, cache
        )
        out = self._block_attention_paged(q, md, cache)
        return linear_out(self.o_proj(out))

    def _block_attention_paged(self, q: torch.Tensor, md, cache) -> torch.Tensor:
        """``[context | block]`` attention as ``B * T`` q_len=1 paged decode rows.

        A length-1 query sees its whole context, so per-row decode is the
        non-causal block attention by construction. The rows read the request's
        page-16 table and capped length that ``build_paged_block`` built
        (``dspark_bt16_rows`` / ``dspark_ctx_rows``, sliding window applied
        there), through the same per-row ASM / Gluon decode the target's sparse
        layers run -- no host sync, fixed shapes, so the block can be captured.
        The context is read once per block position, one KV head per rank.
        """
        k16, v16, ks16, vs16 = cache  # page-16 views, see _bound_cache
        rows = q.shape[0]
        q = q.view(rows, self.num_heads, self.head_dim)
        out = torch.empty_like(q)
        sparse_pa_per_row(
            q,
            k16,
            v16,
            md.dspark_bt16_rows,
            md.dspark_ctx_rows,
            num_kv_heads=self.num_kv_heads,
            sm_scale=self.scaling,
            output=out,
            k_scale=ks16,
            v_scale=vs16,
        )
        return out.view(rows, -1)


@triton.jit
def _markov_step_atomic(
    base_ptr,  # this position's base logits, this rank's vocab shard, rows strided
    prev_ids_ptr,  # FIRST: [B] the anchors
    gathered_ptr,  # not FIRST: [W, B] int64, every rank's packed best of the previous position
    prev_packed_ptr,  # not FIRST: [B] int64, this rank's packed best of the previous position (reset here)
    out_prev_ptr,  # not FIRST: the previous position's column of the [B, T] ids
    w1_ptr,
    w2_ptr,
    packed_ptr,  # [B] int64: this position's best on this rank (reset by the next launch)
    num_rows,
    vocab_size,
    rank,
    stride_base_row,
    stride_prev,
    stride_out,
    stride_w1_vocab,
    stride_w2_vocab,
    vocab_offset,
    FIRST: tl.constexpr,
    W: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """One block position of the sharded greedy Markov block in one launch: the
    previous position's cross-rank winner (tile 0 also writes it into the
    previous column and resets this rank's previous packed best), the bias
    GEMV of `markov_logits_tile_of`, and each tile's (max, lowest id) folded
    into the row's packed best by ``atomic_max``: `dspark_markov_argmax_tp`'s
    stage-2 and cross-rank pick launches folded into this one. The same values
    and tie rule, so the same ids."""
    tile = tl.program_id(0)
    offs_row = tl.program_id(1) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    offs_v = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    row_mask = offs_row < num_rows
    v_mask = offs_v < vocab_size
    if FIRST:
        prev = tl.load(prev_ids_ptr + offs_row * stride_prev, mask=row_mask, other=0)
    else:
        prev = unpack_id(ranks_best(gathered_ptr, offs_row, row_mask, num_rows, W))
        if tile == 0:
            tl.store(out_prev_ptr + offs_row * stride_out, prev, mask=row_mask)
            tl.store(
                prev_packed_ptr + offs_row,
                tl.full([BLOCK_ROW], EMPTY_K, tl.int64),
                mask=row_mask,
            )
    vals = markov_logits_tile_of(
        prev, base_ptr, w1_ptr, w2_ptr, base_ptr, tile, offs_row, offs_v, row_mask,
        v_mask, rank, stride_base_row, stride_w1_vocab, 0, stride_w2_vocab,
        vocab_offset, BLOCK_ROW, BLOCK_V, BLOCK_K, False,
    )  # fmt: skip
    tile_max, tile_idx = argmax_lowest_id(
        vals, offs_v[None, :], v_mask[None, :], 2147483647
    )
    hit = tile_idx != 2147483647
    packed = pack_max_id(tile_max, tile_idx + vocab_offset)
    tl.atomic_max(packed_ptr + offs_row, packed, mask=row_mask & hit, sem="relaxed")


@triton.jit
def _markov_last_atomic(
    gathered_ptr,
    packed_ptr,
    out_ptr,
    num_rows,
    stride_out,
    W: tl.constexpr,
    BLOCK_ROW: tl.constexpr,
):
    """The last position's cross-rank winner into its column; resets this rank's
    packed best for the next block."""
    offs_row = tl.program_id(0) * BLOCK_ROW + tl.arange(0, BLOCK_ROW)
    row_mask = offs_row < num_rows
    best = unpack_id(ranks_best(gathered_ptr, offs_row, row_mask, num_rows, W))
    tl.store(out_ptr + offs_row * stride_out, best, mask=row_mask)
    tl.store(
        packed_ptr + offs_row, tl.full([BLOCK_ROW], EMPTY_K, tl.int64), mask=row_mask
    )


def _is_native_mxfp8(lin) -> bool:
    """``lin`` runs the native MXFP8 1x32 GEMM (not a weight dequantized to
    bf16, not another weight format)."""
    weight = getattr(lin, "weight", None)
    return (
        getattr(lin, "native_a8_group_rows", None) == 1
        and not getattr(lin, "runs_bf16", False)
        and weight is not None
        and weight.dtype == torch.float8_e4m3fn
        and getattr(lin, "weight_scale", None) is not None
    )


class M3DSparkDecoderLayer(DSparkGQADecoderLayer):
    """One draft layer: pre-norm attention + pre-norm dense SwiGLU-OAI MLP."""

    attention_cls: ClassVar[type[M3DSparkAttention]] = M3DSparkAttention

    def __init__(
        self,
        atom_config: "Config",
        config,
        quant_config: QuantizationConfig,
        layer_num: int,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = self.attention_cls(
            atom_config, config, quant_config, layer_num, prefix=f"{prefix}.self_attn"
        )
        self.mlp = MiniMaxM3MLP(
            config=config,
            intermediate_size=int(config.dspark_ffn_hidden_size),
            quant_config=quant_config,
            reduce_results=True,
            prefix=f"{prefix}.mlp",
            # The checkpoint says `swigluoai`, and the reference draft MLP is
            # SwiGLU-OAI whatever the field says; transformers'
            # MiniMaxM3VLTextConfig, which this config loads as, rewrites the
            # field to its `silu` default.
            hidden_act="swigluoai",
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )


class MiniMaxM3DSpark(DSparkDraftModel):
    """Top-level standalone DSpark draft for MiniMax-M3.1 (GQA backbone +
    Markov head + confidence head).

    The block is the base ``block_backbone``, with no embedding scale: the
    reference embeds with the target's table as is.
    """

    # Per-row MHA decode; its slots, lengths and tables are built in
    # `build_paged_block`.
    block_attention_is_mla = False
    builds_paged_block = True

    layer_cls: ClassVar[type[M3DSparkDecoderLayer]] = M3DSparkDecoderLayer

    # Whether `head_and_sample` runs the confidence head it has. `__init__`
    # sets it from `confidence_schedule`, the head's only consumer: with the
    # schedule off the proposer drops the head's output.
    emits_confidence: bool = True
    # `_markov_atomic`'s per-position packed bests, `[T, max_num_seqs]`
    _mk_packed: "torch.Tensor | None" = None
    # `_context_kv_weight`: None until built, () where it cannot be
    _ctx_kv: "tuple | None" = None

    weights_mapping: ClassVar[dict[str, str]] = {
        "language_model.model.dspark.": "",
        ".decoder_layer.": ".",
    }
    packed_modules_mapping: ClassVar[dict[str, tuple[str, str | int]]] = {
        ".q_proj": (".qkv_proj", "q"),
        ".k_proj": (".qkv_proj", "k"),
        ".v_proj": (".qkv_proj", "v"),
        ".gate_proj": (".gate_up_proj", 0),
        ".up_proj": (".gate_up_proj", 1),
    }

    def __init__(self, atom_config: "Config", layer_offset: int = 0) -> None:
        super().__init__()
        if (
            atom_config.decode_context_parallel_size > 1
            and self.kv_pool_spec is None
        ):
            raise NotImplementedError(
                "The MiniMax-M3.1 DSpark draft does not support DCP."
            )
        config = atom_config.hf_config
        self.hf_config = config
        # The draft checkpoint is MXFP8 (``[1, 32]`` blocks) whatever the
        # target is -- the M3.1 targets come as MXFP8 + NVFP4 experts, Quark
        # MXFP4, or BF16.
        quant_config = draft_quant_config(
            config,
            draft="MiniMax-M3 DSpark",
            accepted={(QuantType.per_1x32, dtypes.fp8)},
        )

        self.hidden_size = config.hidden_size
        self.vocab_size = int(config.vocab_size)

        # Replicated: the aux hidden states arrive replicated, and the
        # checkpoint keeps this projection BF16 (an ignored layer).
        self.fc = ReplicatedLinear(
            self.hidden_size * len(config.dspark_target_layer_ids),
            self.hidden_size,
            bias=False,
            quant_config=None,
            prefix="fc",
        )
        self.hidden_norm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                self.layer_cls(
                    atom_config,
                    config,
                    quant_config,
                    layer_offset + i,
                    prefix=f"layers.{i}",
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        self.final_norm = GemmaRMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.init_block_fields(config)
        # 0 or absent: every block row attends the whole context.
        self.sliding_window = int(getattr(config, "sliding_window", 0) or 0)
        self.confidence_head = (
            DSparkConfidenceHead(self.hidden_size, self.markov_head.rank, bias=True)
            if getattr(config, "enable_confidence_head", False)
            else None
        )
        # A config without DSpark options has the default: no schedule.
        self.emits_confidence = bool(
            getattr(getattr(atom_config, "dspark", None), "confidence_schedule", False)
        )

        # Bound by share_with_target(); the checkpoint ships neither.
        self.embed_tokens = None
        self.lm_head = None

        # `build_paged_block`'s buffers, sized here and allocated on its
        # first call.
        self.max_num_seqs = atom_config.max_num_seqs
        self._block_rows_p = None

    # ---- drafting entry points (called by the proposer) --------------------

    def _alloc_block_rows(self, P: int, T: int, device) -> None:
        """The first call is the warmup, ahead of any capture: these addresses
        are the ones every recording bakes in."""
        max_bs = self.max_num_seqs
        self._alloc_cap = AllocationCap(max_bs, device)
        self._bt16_rows = torch.zeros(max_bs * T, P, dtype=torch.int32, device=device)
        self._ctx_rows = torch.zeros(max_bs * T, dtype=torch.int32, device=device)
        self._block_rows_p = P
        if self._mk_packed is None:
            self._alloc_markov_packed(T, device)

    def _alloc_markov_packed(self, num_draft: int, device) -> None:
        """Once, never regrown: a captured block holds its address."""
        assert not torch.cuda.is_current_stream_capturing()
        self._mk_packed = torch.full(
            (num_draft, self.max_num_seqs), EMPTY, dtype=torch.int64, device=device
        )

    def _context_kv_weight(self):
        """The K/V rows of every context layer's qkv_proj as one MXFP8 weight
        ``[layers * 2 * kv, hidden]``, its E8M0 scales and whether it is
        preshuffled: one GEMM gives every layer's context K/V, and the q rows
        each layer's projection would compute and drop are not read. The bytes
        are the layers' own, copied once after the weights load
        (``make_row_view`` cuts whole shuffle tiles); a weight updated in place
        later is not seen. None where a projection is not native MXFP8 1x32."""
        if self._ctx_kv is None:
            if torch.cuda.is_current_stream_capturing():
                # built eagerly only: a recording would put it in its pool
                return None
            projs = [layer.self_attn.qkv_proj for layer in self.context_layers]
            attn = self.context_layers[0].self_attn
            self._ctx_kv = ()
            if all(_is_native_mxfp8(p) for p in projs) and not any(
                p.bias is not None or getattr(p, "is_output_padded", False)
                for p in projs
            ):
                try:
                    views = [
                        p.make_row_view(attn.q_size, 2 * attn.kv_size) for p in projs
                    ]
                except ValueError:  # offsets off the scale tiles at this TP size
                    views = None
                if views is not None:
                    self._ctx_kv = (
                        torch.cat([v.weight.data for v in views]),
                        torch.cat([v.weight_scale.data for v in views]),
                        bool(getattr(views[0].weight, "is_shuffled", False)),
                    )
        return self._ctx_kv or None

    def project_context_mxfp8(self, taps):
        """`project_context` as the context K/V GEMM's input: ``(x, None)``,
        which the GEMM quantizes itself, or an already quantized pair."""
        return self.project_context(taps), None

    def write_context_kv(
        self, taps, positions, slot_mapping=None, pcp_num_tokens=None
    ) -> None:
        """Every context layer's K/V from one GEMM over the concatenated K/V
        rows (`_context_kv_weight`), then each layer's k_norm + RoPE + cache
        write. The per-layer path stays for a PCP-split prefill, the dummy
        run, an unbound cache and a projection that is not native MXFP8."""
        kv_weight = None
        if (
            pcp_num_tokens is None
            and not get_forward_context().context.is_dummy_run
            and self.context_layers[0].self_attn.kv.bound
        ):
            kv_weight = self._context_kv_weight()
        if kv_weight is None:
            return super().write_context_kv(
                taps, positions, slot_mapping, pcp_num_tokens
            )
        weight, scale, shuffled = kv_weight
        x, x_scale = self.project_context_mxfp8(taps)
        kv = native_quant_linear(
            x,
            weight,
            scale,
            x_scale=x_scale,
            weight_group_rows=1,
            weight_preshuffled=shuffled,
        )
        for i, layer in enumerate(self.context_layers):
            attn = layer.self_attn
            k = kv[:, 2 * i * attn.kv_size : (2 * i + 1) * attn.kv_size]
            v = kv[:, (2 * i + 1) * attn.kv_size : (2 * i + 2) * attn.kv_size]
            attn._norm_rope_write(
                k, k, v, 0, positions, slot_mapping, attn._bound_cache()
            )

    def build_paged_block(
        self,
        attn_metadata,
        block_tables: torch.Tensor,  # [running_bs, W] ids of entry_tokens pages
        anchor_positions: torch.Tensor,  # [running_bs] staged, pad rows 0
        block_positions: torch.Tensor,  # [running_bs, T] out
        slots: torch.Tensor,  # [running_bs * T] out, the block's KV write slots
        ctx_lens: torch.Tensor,  # [running_bs] out, anchor + 1 + T
        allocated_pages,  # [>= scheduled_bs] host, pages each request holds
        scheduled_bs: int,
        entry_tokens: int,  # tokens one page id spans, the slots' unit
        pad_ctx_len: int,  # a pad row's context, on page 0
    ) -> None:
        """The block's positions, KV write slots and context lengths, and each
        block row's page-16 table and length, in one launch
        (`dspark_block_meta`), into buffers sized at ``max_num_seqs`` and only
        sliced, so a captured block replays against this step's numbers.

        The block is attended as ``running_bs * T`` independent q_len=1 decode
        rows -- non-causal by construction -- each over its request's
        ``[context | block]`` through a page-16 block table, the layout the
        target's sparse decode kernels read. A row at position p attends the
        keys from ``p - window + 1`` on, from the page-16 that holds that key,
        so it also reads up to 15 keys before it; its length ends it within
        its own request's table.

        Block positions can run past the pages the scheduler has allocated for
        the request (it allocates through the verify window, the block reaches
        up to T past the accepted anchor). A table column that was never
        allocated reads as page 0, which is some other request's page -- under
        prefix caching, a prefix every request shares. Such positions are
        neither written (slot -1) nor read (the row's context is capped at what
        is allocated): their draft rows simply miss a few of their own keys.
        ``ctx_lens`` (``attn_metadata.context_lens``) stays uncapped.
        """
        running_bs, T = block_positions.shape
        P = (entry_tokens // ASM_PAGE_SIZE) * block_tables.shape[1]
        if self._block_rows_p != P:
            self._alloc_block_rows(P, T, block_tables.device)
        alloc_pages = self._alloc_cap.stage_pages(
            allocated_pages, scheduled_bs, running_bs
        )
        rows = running_bs * T
        dspark_block_meta(
            anchor_positions,
            block_tables,
            alloc_pages,
            block_positions,
            slots,
            ctx_lens,
            self._bt16_rows[:rows],
            self._ctx_rows[:rows],
            scheduled_bs=scheduled_bs,
            entry_tokens=entry_tokens,
            window=self.sliding_window,
            pad_ctx_len=pad_ctx_len,
        )
        attn_metadata.dspark_bt16_rows = self._bt16_rows[:rows]
        attn_metadata.dspark_ctx_rows = self._ctx_rows[:rows]

    def _markov_atomic(self, hidden, anchor_ids, num_draft):
        """The greedy block off this rank's vocab shard with no confidence
        head: per position one launch (`_markov_step_atomic`) and one ``[B]``
        int64 all-gather, then one closing launch, where
        `_sample_block_off_the_vocab_shard` runs stage 1, stage 2, the
        all-gather and the cross-rank pick. The same ids. Each position's
        packed bests live in the persistent ``[T, max_num_seqs]`` buffer and
        are back at ``EMPTY`` between blocks (each is reset by the launch after
        the all-gather that read it), so no launch clears it."""
        head, markov = self.lm_head, self.markov_head
        bs = anchor_ids.shape[0]
        base_logits = tgemm.mm(hidden, head.weight).view(bs, num_draft, -1)
        v_local = base_logits.shape[-1]
        w1, w2 = markov.markov_w1.weight, markov.markov_w2.weight
        out_ids = anchor_ids.new_empty(bs, num_draft)
        tp_group = get_tp_group()
        if self._mk_packed is None:
            # a forward ahead of the block buffers (the memory-profile pass)
            self._alloc_markov_packed(num_draft, base_logits.device)
        assert num_draft <= self._mk_packed.shape[0]
        assert bs <= self._mk_packed.shape[1]
        geometry = markov_launch_geometry(bs, v_local)
        gathered = None
        for k in range(num_draft):
            base_k = base_logits[:, k]
            packed = self._mk_packed[k, :bs]
            _markov_step_atomic[(geometry.v_tiles, geometry.row_tiles)](
                base_k, anchor_ids, packed if gathered is None else gathered,
                self._mk_packed[max(k - 1, 0), :bs], out_ids[:, max(k - 1, 0)],
                w1, w2, packed, bs, v_local, w2.shape[1], base_k.stride(0),
                anchor_ids.stride(0), out_ids.stride(0), w1.stride(0), w2.stride(0),
                head.vocab_start_idx, FIRST=k == 0, W=tp_group.world_size,
                BLOCK_ROW=geometry.block_row, BLOCK_V=BLOCK_V, BLOCK_K=BLOCK_K,
                num_warps=geometry.num_warps, num_stages=2,
            )  # fmt: skip
            # the int64 words cross as float32 pairs, as the (max, id) pairs do
            gathered = (
                tp_group.all_gather(
                    packed.view(torch.float32).view(bs, 2),
                    dim=0,
                    use_custom=envs.ATOM_USE_CUSTOM_ALL_GATHER,
                )
                .view(torch.int64)
                .view(-1)
            )
        block_row = min(1024, triton.next_power_of_2(bs))
        _markov_last_atomic[(triton.cdiv(bs, block_row),)](
            gathered, self._mk_packed[num_draft - 1, :bs], out_ids[:, num_draft - 1],
            bs, out_ids.stride(0), W=tp_group.world_size, BLOCK_ROW=block_row,
            num_warps=1,
        )  # fmt: skip
        return out_ids

    def head_and_sample(self, hidden, anchor_ids, num_draft):
        bs = anchor_ids.shape[0]
        emit = self.emits_confidence and self.confidence_head is not None
        if self._samples_off_the_vocab_shard(self.lm_head):
            # Under TP: no logits all-gather, V / tp rows of markov_w2 a
            # position; the same ids and confidences.
            if not emit:
                return self._markov_atomic(hidden, anchor_ids, num_draft), None
            out_ids, markov_embeds = self._sample_block_off_the_vocab_shard(
                hidden, self.lm_head, anchor_ids, num_draft
            )
            return out_ids, self.confidence_head(
                hidden.view(bs, num_draft, -1), markov_embeds
            )
        base_logits = self.lm_head(hidden).view(bs, num_draft, -1)
        out_ids = anchor_ids.new_empty(bs, num_draft + 1)
        out_ids[:, 0] = anchor_ids
        markov_embeds = []
        for k in range(num_draft):
            # Greedy: sampling temperature is the target verify's business.
            markov_embeds.append(
                self.markov_head.sample_next(
                    out_ids[:, k], base_logits[:, k], out_ids[:, k + 1]
                )
            )
        confidence = None
        if emit:
            confidence = self.confidence_head(
                hidden.view(bs, num_draft, -1), torch.stack(markov_embeds, dim=1)
            )
        return out_ids[:, 1:], confidence
