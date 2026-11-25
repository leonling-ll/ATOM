# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3-Pro attention: a plain-torch reference, correctness first.

What Pro's attention is, against M3 / M3.1 (see ``models/minimax_m3/pro.py``):

* one KV head (MQA) of ``head_dim`` 256 whose V *is* K with the rope dims dropped
  (``sparse_enable_kv_share`` / ``sparse_kvshare_value_drop_rope_dim``): no V
  projection, ``value_head_dim`` = 256 - 64 = 192. The V cache the builder allocates
  anyway (256 wide) holds that V in its first 192 rows, for aiter's paged attention,
  which reads K and V from separate caches;
* a lightning indexer of 16 heads x 128 dims scoring 16-token blocks of one shared
  index-key head, top-128 blocks per query with 1 init + 9 local blocks forced;
* the indexer's inputs may come from two layers back (lookahead, built by the model).

None of aiter's / ATOM's M3 kernels fits that (they are page-128, one index head per
KV head, separate V), so this impl keeps the cache layouts the standard builder
already binds -- the K cache in its SHUFFLE shape, the index cache as
``(blocks, 16, 128)`` -- and does the rest in torch on a fp32 path:

    norm -> partial rope -> write K / index-K -> (indexer -> block select) -> attention

At most 2048 tokens (128 blocks of 16) selection is the identity, so short prompts
(GSM8K) never exercise the indexer; ``ATOM_M3PRO_VARIANTS`` names the alternatives
for the parts of the formulas the checkpoint does not pin down.
"""

import functools
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, fields, replace

import torch

from atom.model_ops.attention_mha import PagedAttentionImpl
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.utils.forward_context import get_forward_context

try:  # the Triton kernels; the torch reference works without them
    from atom.model_ops.minimax_m3 import pro_kernels
except ImportError:  # pragma: no cover
    pro_kernels = None

logger = logging.getLogger("atom")

_NEG_INF = float("-inf")
# On the K-only cache aiter's gluon decode reads V from the K rows about 3 us a layer slower
# than from a V cache, and the Triton attention wins up to 16 rows (opt/kv/test_vrows.py, MI355X:
# bs1 16.0 vs 13.9 us, b16 16.1 vs 15.5, b32 at 8k 16.7 vs 20.7, b64 22.4 vs 30.3)
_K_ROWS_AITER_MIN = 17
# Forced blocks are pinned above every real score, init above local (as the M3
# selector's 1e30 / 1e29): a block that is both ends up local.
_PIN_INIT = 1e30
_PIN_LOCAL = 1e29


@dataclass
class ProVariants:
    """The parts of Pro's attention the checkpoint does not fix, one switch each.

    Read from ``ATOM_M3PRO_VARIANTS`` (a JSON object, or the path of a JSON file that
    is re-read when it changes, at the first layer of a forward so all of a forward's
    layers agree). The defaults are the best reading of the config (the indexer's that of
    a PyTorch reference implementation of the model); the others are what is tried, one
    at a time, against GSM8K / long-context retrieval.
    """

    # YaRN magnitude. "hf" (default): the config's attention_factor taken as the whole
    # magnitude (1.0 -> none), which is what transformers' reference computes for
    # this config; "aiter": aiter's 0.1 ln(factor) + 1 on cos/sin (it multiplies
    # attention_factor into that default instead of replacing it); or a number.
    # Measured (GSM8K, 5-shot, BOS, 200 hard docs): hf 150 vs aiter 117 correct.
    mscale: str = "hf"
    # The indexer switches (head_reduce .. pool_scale below) are all 12 / 12 on a
    # 14k-token key-value retrieval (dense attention too); only a selector blind to
    # the scores (init_blocks=128) fails it (0 / 12). So that test shows the indexer
    # works, not which formula is the checkpoint's; the defaults (pool4 block scores, max
    # over heads, pooling temperature 1/sqrt(128)) are the reference implementation's. How
    # well the block scores predict the dense attention (``pro_diag``; real weights, 9.5k-14.5k
    # tokens, layers 2-11 of four documents, 1920 queries) prefers lse / lse: the share of the
    # oracle's headroom (over the forced blocks) the top-128 captures is 0.951 for tokmax
    # summed over heads with the lookahead from the FFN norm's output, 0.960 for tokmax
    # without lookahead and 0.978 for lse / lse; with 32 blocks 0.900 / 0.906 / 0.953. (The
    # lookahead itself is a toss-up against the layer's own input: 0.979 / 0.958.) GSM8K
    # (5-shot, all 1319) is within noise between pool4 / max and lse / lse: 0.927 / 0.936.
    # How the 16 index heads' block scores become one: "max" | "lse" | "sum" | "mean".
    head_reduce: str = "max"
    # Block score inside a head: "pool4": softmax-pool the block's 4 groups of 4 tokens with
    # ``index_compress_query`` into centroids and take the best centroid's score; "lse":
    # log-sum-exp over the 16 token scores / sqrt(index_head_dim) (the log of the attention
    # mass the block would get if the index scores were attention logits: with rope on and
    # relu off its softmax is calibrated to the dense block mass, KL 0.010 nats at layers 0-1,
    # where no pooled reading got below 0.021); "max" over the 16 token scores (M3's).
    block_score: str = "pool4"
    # Which 192 of the 256 K dims are V: "tail" drops the leading (rope) dims.
    v_slice: str = "tail"
    # What V is cut from: "post" the K after k_norm (what is cached as K: the model
    # keeps one 256-wide K per token and no V), "pre" the raw k_proj output (kept in
    # the otherwise unused V cache, no norm applied). "pre" generates only newlines
    # (0 / 200 on the GSM8K subset): V comes from the normed K.
    v_src: str = "post"
    # Multiplier on the softmax scale (head_dim ** -0.5). GSM8K is flat in 0.85 .. 1.6
    # (150 .. 165 / 200 on the hard subset, noise about +-7), so the natural 1.0 stays.
    scale_mult: float = 1.0
    # Partial-rope pairing: "neox" (halves) or "gptj" (even / odd interleaved).
    rope_style: str = "neox"
    # Attend every causal key whatever the indexer selects (the oracle).
    dense_only: bool = False
    # Rope on the indexer's q / k (as M3's indexer; off to test).
    index_rope: bool = True
    # Where a lookahead layer's indexer input is taken from (the layer two back): "resid_in"
    # the residual stream before that layer's FFN norm (the config's ``pre_ffn_norm``), put
    # through the *reading* layer's input_layernorm as the q / k / v inputs are; "ffn_in" that
    # layer's FFN norm output (worst measured: its channel gains are the FFN's, not the
    # attention's); "resid" the raw residual (massive activations: unusable). Every source
    # normed with the reading layer's input norm (or its own input) is within noise of the
    # others, so this is the config-faithful one, not a proven one.
    lookahead_src: str = "resid_in"
    # Lookahead on / off (off: every indexer reads its own layer's input).
    lookahead: bool = True
    # Forced-block counts; None keeps the config's (1 init / 9 local).
    init_blocks: int | None = None
    local_blocks: int | None = None
    # Scale of the index token scores (q . k), before the block score and the head
    # reduction (a temperature for "lse" scores and the head "lse"; the "max" / "pool4"
    # block scores with "sum" / "max" / "mean" heads are insensitive to it).
    index_scale: float = 1.0
    # "pool4": the softmax logits of the 4-token pooling are multiplied by this
    # (1/sqrt of the index head dim).
    pool_scale: float = 128**-0.5
    # Developer diagnostic (``pro_diag``): run dense attention, and on sampled queries of a
    # long prefill score each reading of the indexer by the share of the dense attention
    # it selects. ``diag_tag`` names the output file.
    diag: bool = False
    diag_tag: str = "x"
    # The Triton kernels (``pro_kernels``) instead of the torch reference: same math, a
    # handful of launches per layer instead of ~100, no host sync past the lengths.
    # Every other switch above either is honoured by them or sends the layer back to torch.
    kernels: bool = False
    # The prep of the kernel path (norm + rope of q / k / index q / index k, the K / V /
    # index-K cache writes): "triton" (``pro_kernels.prep``) or "aiter"
    # (``pro_kernels.prep_aiter``: aiter's fused_qk_norm_rope_cache_mqa, the same math in
    # one HIP launch, any number of local heads, Triton's at most 32). "aiter" runs
    # Triton's where this aiter lacks it.
    prep: str = "triton"
    # The attention of the kernel path: "triton" (``pro_kernels.attention``) or "aiter":
    # aiter's paged-attention decode (gluon, as ATOM's M3 runs it) with every token a
    # sequence of its own over a page table of its selected blocks, V from the V cache.
    attn: str = "triton"
    # With an fp8 KV cache, "aiter" runs the dots in fp8 too (else in the activation dtype).
    attn_fp8_compute: bool = True
    # The kernels' block selection: "torch" (torch.topk over the block table's width),
    # "aiter" (aiter's per-row top-k over each token's causal blocks only) or, for the
    # "aiter" attention, "fused": selection and page tables in one kernel (select_rows),
    # "split": the same tables with each row split over several programs
    # (select_rows_split; its cost follows the context, not the block table's width).
    topk: str = "torch"
    # Kernel launch shape: programs per token over the selected blocks (0: by batch size)
    # and blocks per step.
    attn_splits: int = 0
    attn_nblk: int = 4
    # MXFP8 activation quant (E4M3, group-32 E8M0 scales, ``quantize_fp8``'s bytes) written
    # by the ops that produce the activations -- the norms after the all-reduces, the
    # lookahead norm, the SwiGLU -- instead of a ``quantize_fp8`` launch before each native
    # group32 FP8 GEMM (``pro.py``), and the decode / verify attention below the day0
    # decode writes o_proj's (``_mxfp8_out``: the Triton attention always, aiter's where its
    # gluon decode takes ``output_scale``; prefill keeps bf16).
    # Off under ``diag``; the norms keep the unfused path on an aiter without the MXFP8
    # norm outputs.
    fuse_mxfp8_quant: bool = True
    # A MoE layer's post-attention all-reduce, norm (+ MXFP8 copy), router, sigmoid
    # top-k and moe_sorting as one aiter launch (``models/minimax_m3/moe_route_glue``),
    # up to 512 rows; six launches a MoE layer fewer. Read when the model is built
    # (the op's IPC buffers) and at each forward.
    moe_route_glue: bool = False
    # Prefill indexer split over the TP ranks: each rank scores and selects only its share
    # of a long prefill's rows (``row_split_index``), then the page tables are all-gathered.
    # Every row is computed as before, so the selection is the same bit for bit. Needs the
    # kernel path with "topk": "fused" or "split"; only prefills of at least _SPLIT_MIN_TOKENS rows
    # that select (more than ``topk`` blocks) split.
    index_split: bool = False
    # No effect: index q and k share one packed GEMM (index_qk_proj), so every row's
    # index q is computed; kept so variants files that set it still parse.
    index_split_proj: bool = True
    # Decode / speculative-verify indexer split over the TP ranks: in a graph of at least
    # this many rows (0: off), a multiple of the TP size, on aiter's page tables, each rank
    # scores and selects one contiguous share of the rows and the [page table | ctx] rows are
    # all-gathered back in order. Bit for bit the same selection, an eighth of the work.
    index_split_decode: int = 0
    # The score_tq kernel (prefill, speculative verify) leaves the tiles past a program's
    # rows unwritten instead of -inf, as the selections that read only causal blocks allow
    # (fused / split / aiter): its cost follows the context, not the block table's width.
    score_tq_causal: bool = False
    # vLLM's compressed index cache: every 4 index keys softmax-pooled into one bf16 row
    # (ic_c [pages, 4, 128]), written after prep; the scorer reads those rows only.
    index_compress: bool = False
    # Prefill and speculative-verify scores take ``score_tq`` rows per program (a power of
    # two; 0: one): the rows of a sequence read each tile of block keys once, one dot for
    # all their query heads. A plain decode (one row a sequence) keeps one.
    score_tq: int = 0
    # The decode / verify block scorer of the kernel path: "triton" (``scores``) or
    # "fused" (``scores_fused``: aiter's FlyDSL scorer, each tile of keys read once
    # for up to 8 rows of a sequence). "fused" takes the static (graph) decode with
    # pool4 / max scores (with "index_compress", over the bf16 centroids) and a
    # selection other than "torch"; everything else, and an aiter or GPU without it,
    # keeps "triton".
    index: str = "triton"
    # Rows (tokens) at or below which those fused all-reduce + norm sites run as aiter's
    # Lamport all-reduce + ``add_rmsnorm_mxfp8`` (two launches) instead of the fused
    # 1-stage kernel: 7.5 vs 9.6 us at 1 row, 9.7 vs 12.6 at 16 (TP8, 7168). The sum and
    # the residual are the same bytes; the norm rounds on its own, so a bf16 output can
    # move by one ulp (one element at 16 rows).
    # Needs the Lamport path on (AITER_CUSTOM_AR_DISABLE_LAMPORT unset); 0 turns it off.
    ar_lamport_rows: int = 0
    # Rows (tokens) at or below which the same sites run aiter's one-launch Lamport
    # two-shot all-reduce + residual add + RMSNorm + MXFP8 (the fused 1-stage kernel's
    # bytes) instead; before ``ar_lamport_rows``. Needs the Lamport path on and an aiter
    # with ``tensor_model_parallel_fused_allreduce_rmsnorm_lamport`` (else the sites keep
    # their route); 0 turns it off.
    ar_fused_rows: int = 0
    # A MoE layer hands its routed and shared-expert outputs over unsummed, and the next
    # layer's input all-reduce adds them (``routed + shared`` rounded to bf16, the bytes
    # of the separate add): inside the ``ar_fused_rows`` kernel up to its rows, as a
    # separate add before the all-reduce otherwise. Not on the last layer.
    moe_tail_ar: bool = False
    # Rows (tokens) at or below which a MoE layer runs its shared expert on a side
    # stream while its router and routed experts run on the current one, joined before
    # the layer returns; a CUDA graph records the fork and the join. The same kernels on
    # the same inputs: the same bytes. It pays with ``moe_tail_ar`` (no add after the
    # join); 0 turns it off.
    moe_two_stream_rows: int = 0
    # The K-only cache (``ProKOnlyLayout``): one 256-wide row a token, V read from its tail,
    # no V cache (776 -> 520 B a token and layer with an fp8 KV and a bf16 index cache).
    # The attention is then the Triton one. Read once, when the layers are built: the
    # pool's layout cannot change while it serves.
    k_only: bool = False
    # An fp8 K cache as a plain (clamped) cast, its scale plane all 1.0: what moo's and
    # amd-kernels-day0's kernels read. The normed K is O(1), within e4m3's range. The aiter
    # prep writes it where its op takes ``kv_unit_scale``, else the Triton prep runs.
    k_plain_fp8: bool = False
    # The prefill attention of the K-only cache: "triton", or "day0" (amd-kernels-day0's
    # FlyDSL fp8_sparse_prefill as ported into aiter, aiter.ops.flydsl's
    # minimax_m3_fp8_sparse_prefill; needs k_plain_fp8, 16 local heads and an aiter that
    # has it, else "triton").
    prefill_attn: str = "triton"
    # Decode / verify batches of at least this many rows attend with amd-kernels-day0's FlyDSL
    # sparse decode (day0#4, one workgroup a row: ahead of the paged attention from ~24 rows)
    # on the K-only plain fp8 cache, the picks select_logical's; 0: off. Without the package
    # (or day0#4 in it) the batch keeps the other attention.
    day0_decode_rows: int = 24
    # ... and its epilogue writes o_proj's MXFP8 input (E4M3 + E8M0 of the bf16-rounded rows,
    # quantize_fp8's bytes) where o_proj is a native group32 FP8 GEMM: no separate quant.
    day0_decode_mxfp8: bool = True
    # Mono decode phase 1 (``models/minimax_m3/pro_mono``): the most graph rows a decode /
    # verify forward may have for its MoE layers' o_proj .. FFN segment to run as one
    # persistent kernel; 0 turns it off, at most ``pro_mono.config.max_rows``. Wider
    # forwards keep production's segment (``moe_route_glue`` where it is on). The layers
    # mono takes get the decode attention's output as bf16. On where the deployment and
    # the variants allow it (``pro_mono.config.config_refusal``,
    # ``pro_mono.route.variants_refusal``: TP8 with FlyDSL, the Lamport AR #1 up to
    # these rows, the MXFP8 norms and the resid_in lookahead, a ``day0_decode_rows``
    # above the limit); elsewhere it logs why and production serves. Latched once, at
    # the first forward (``pro_mono.route``, agreed over the TP group): a later edit of
    # the file does not change it. The accuracy it was turned on with (A16W4, the
    # Agentic recipe's variants) is in recipes/MiniMax-M3.1-Pro-Accuracy-Switches.md.
    mono_ffn_rows: int = 16
    # The mono FFN's routed experts' activations: "bf16" (A16W4: bf16 x and mids against
    # the MXFP4 experts dequantized in the kernel, production's a16w4 numerics at these
    # widths), "fp8" (W4A8: the norm's MXFP8 x and MXFP8 mids, gfx950's f8f6f4 MFMA) or
    # "fp4" (W4A4, aiter a4w4 numerics); the last two quantize the activations, a
    # numerics change. Latched with ``mono_ffn_rows`` and agreed over the TP group.
    mono_routed_act: str = "bf16"
    # Developer profile, not an attention switch (the attention path never reads it): count
    # the distinct routed experts each MoE layer reads in every forward (TP rank 0) into
    # ``<diag_tag>_experts.jsonl`` (``eplb.ExpertCountProfile``). A counting run, not a
    # timing run. Read once, when the model is built: the count runs inside the captured
    # decode graphs.
    expert_count: bool = False
    # The routed experts of a MoE call of at most 64 rows: "aiter" (``fused_moe``) or
    # "day0": aiter's ``minimax_m3_moe_experts_decode`` (amd-kernels-day0's one-launch
    # sort, the a16w4 GEMMs at its decode tiles, the 4 slots summed in fp32 in slot order;
    # ``fused_moe/pro_experts_decode.py``), "aiter" where this aiter lacks it. Read once,
    # when the layers are built: a captured graph keeps the kernels it recorded.
    moe_decode: str = "aiter"


# Below it, on an fp8 index cache, Triton's query-blocked scores are as fast or faster
# than aiter's fused scorer (MI355X, 16 rows: 19.6 vs 27.3 us at 32k context, 32.9 vs
# 28.2 at 128k; 64 rows: 33.3 vs 26.2 at 32k), so "index": "fused" keeps Triton there.
_FUSED_MIN_ROWS_FP8 = 32

# Below it the all-gather costs more than the scoring it saves (moo's threshold).
_SPLIT_MIN_TOKENS = 1024


def row_split_index(n: int, tp: int, rank: int, device=None):
    """``rank``'s rows of ``n``, cut into ``2 * tp`` segments of ``seg``: segment ``rank`` and
    its mirror ``2 * tp - 1 - rank`` (causal cost grows along a prefill, so the pairs even
    out). ``(idx [2 * seg] int64, seg)``; the last segments' rows past ``n`` repeat ``n - 1``.
    """
    seg = -(-n // (2 * tp))
    s = torch.arange(seg, device=device)
    idx = torch.cat([rank * seg + s, (2 * tp - 1 - rank) * seg + s])
    return idx.clamp_(max=n - 1), seg


def assemble_rows(gathered: torch.Tensor, seg: int, n: int, tp: int) -> torch.Tensor:
    """The ranks' ``row_split_index`` rows, all-gathered ``[tp * 2 * seg, ...]``, back in row
    order ``[n, ...]``."""
    g = gathered.view(tp, 2, seg, *gathered.shape[1:])
    return torch.cat([g[:, 0], g[:, 1].flip(0)]).flatten(0, 1)[:n]


_variants = ProVariants()
_variants_src = None  # (string or path, mtime) the live `_variants` came from
_variants_checked = 0.0


def _parse_variants(obj: dict) -> ProVariants:
    known = {f.name for f in fields(ProVariants)}
    unknown = sorted(set(obj) - known)
    if unknown:
        raise ValueError(
            f"ATOM_M3PRO_VARIANTS: unknown keys {unknown}; known: {sorted(known)}"
        )
    return ProVariants(**obj)


def refresh_variants(force: bool = False) -> ProVariants:
    """Re-read ``ATOM_M3PRO_VARIANTS`` (at most once a second)."""
    global _variants, _variants_src, _variants_checked
    src = os.environ.get("ATOM_M3PRO_VARIANTS", "").strip()
    now = time.monotonic()
    if not src:
        return _variants
    if not force and now - _variants_checked < 1.0:
        return _variants
    _variants_checked = now
    try:
        if src.startswith("{"):
            key = src
            if key != _variants_src:
                _variants = _parse_variants(json.loads(src))
                _variants_src = key
        else:
            mtime = os.stat(src).st_mtime
            key = (src, mtime)
            if key != _variants_src:
                with open(src) as f:
                    _variants = _parse_variants(json.load(f))
                _variants_src = key
                logger.info("M3-Pro variants: %s", _variants)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        logger.warning("M3-Pro variants unchanged (%s: %s)", type(e).__name__, e)
    return _variants


def variants() -> ProVariants:
    return _variants


def diag_path(name: str) -> str:
    """``name`` in the developer diagnostics' directory (``$ATOM_M3PRO_DIAG_DIR``, else
    ``m3pro_diag`` under the system temp directory), created if missing."""
    d = os.environ.get(
        "ATOM_M3PRO_DIAG_DIR", os.path.join(tempfile.gettempdir(), "m3pro_diag")
    )
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


@dataclass(frozen=True)
class ProKOnlyLayout:
    """Pro's K-only pages as an ``MhaKvPool`` layout: per token one 256-wide K row, the
    page row-major (``[blocks, 1, 1, page, 256]``: the SHUFFLE view at ``x`` = 256, which
    the prep kernels and the Triton attention address alike), V its last 192 dims. An fp8
    cache keeps one per-token K scale plane. ``kv_views`` gives K for V too: nothing
    writes a V."""

    kv_dtype: torch.dtype
    tag = "konly"

    def check(self, block_size: int, head_dim: int, blocks_per_entry: int) -> None:
        if head_dim != 256:
            raise ValueError(f"the Pro K-only page is 256 wide, not {head_dim}")

    def cache_fields(self, layers, entry_tokens, num_kv_heads, head_dim):
        from atom.model_ops.attentions.mha_kv_pool import mha_kv_fields

        return mha_kv_fields(
            layers=layers,
            entry_tokens=entry_tokens,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            kv_dtype=self.kv_dtype,
        )[:1]

    def scale_fields(self, layers, entry_tokens, num_kv_heads):
        from atom.model_ops.attentions.mha_kv_pool import mha_kv_scale_fields

        return mha_kv_scale_fields(
            layers=layers,
            entry_tokens=entry_tokens,
            num_kv_heads=num_kv_heads,
            kv_dtype=self.kv_dtype,
        )[:1]

    def kv_views(self, pool, layer):
        nh, hd, bs = pool.num_kv_heads, pool.head_dim, pool.block_size
        k = pool.field_view("k", layer, self.kv_dtype, (-1, nh, 1, bs, hd))
        return k, k


def _index_rows(ik: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Index keys as the index cache holds them: an fp8 cache a plain cast clamped to its
    range (M3's fp8 index cache; the keys are normed), as ``pro_kernels.prep`` writes.
    """
    if dtype.itemsize == 1:
        fmax = torch.finfo(dtype).max
        return ik.float().clamp(-fmax, fmax).to(dtype)
    return ik.to(dtype)


def _reduce_heads(bmax: torch.Tensor, how: str, scale: float) -> torch.Tensor:
    """``[..., NI, NB]`` per-head block scores -> ``[..., NB]``."""
    if scale != 1.0:
        bmax = bmax * scale
    if how == "sum":
        return bmax.sum(dim=-2)
    if how == "mean":
        return bmax.mean(dim=-2)
    if how == "max":
        return bmax.amax(dim=-2)
    if how == "lse":
        return torch.logsumexp(bmax, dim=-2)
    raise ValueError(f"unknown head_reduce {how!r}")


class ProSparseAttentionImpl(PagedAttentionImpl):
    """MiniMax-M3-Pro's MQA + lightning-indexer attention, in torch.

    Plugged into ``Attention`` via ``impl_cls=``: the generic op's ``query`` / ``key`` /
    ``value`` / ``qkv`` carry the module's raw q, k, index q and index k projections (the
    module owns the projections, the lookahead choice of the indexer input, and ``o_proj``).
    """

    # whether o_proj takes MXFP8 rows (the model sets it), and the scales of this forward's
    # MXFP8 output (take_mx_scale)
    mx_out = False
    _mx_scale = None
    # set for one call by ``MiniMaxM3ProAttention.forward_core``: the mono layer kernel
    # reads the attention's output as bf16, so the Triton / aiter rows below the day0
    # decode stay bf16 (the day0 decode's epilogue keeps ``day0_decode_mxfp8``)
    bf16_out = False

    is_indexed_sparse_attention = True
    k_only = False  # the pool is ``ProKOnlyLayout`` (no V cache)
    # The decode context parallel group's size: each rank holds every W-th 16-token block
    # of the KV and index caches (``pro_kernels.dcp_local_positions``).
    dcp_world = 1

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="bf16",
        logits_soft_cap=None,
        attn_type=None,
        kv_sharing_target_layer_name=None,
        layer_num=0,
        mla_modules=None,
        sinks=None,
        rotary_emb=None,
        q_norm=None,
        k_norm=None,
        index_q_norm=None,
        index_k_norm=None,
        index_compress_query=None,
        num_index_heads=16,
        index_head_dim=128,
        value_head_dim=192,
        rotary_dim=64,
        topk=128,
        init_blocks=1,
        local_blocks=9,
        block_size=16,
        **kwargs,
    ):
        if kv_cache_dtype not in ("bf16", "fp8"):
            raise NotImplementedError(
                "MiniMax-M3-Pro keeps a bf16 or fp8 (per-token scaled) KV cache, "
                f"got --kv_cache_dtype {kv_cache_dtype!r}"
            )
        super().__init__(
            num_heads=num_heads,
            head_dim=head_dim,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            layer_num=layer_num,
            mla_modules=mla_modules,
            sinks=sinks,
            rotary_emb=rotary_emb,
            q_norm=q_norm,
            k_norm=k_norm,
            **kwargs,
        )
        assert num_kv_heads == 1, "Pro is MQA: one (shared K/V) head"
        self.index_q_norm = index_q_norm
        self.index_k_norm = index_k_norm
        self.index_compress_query = index_compress_query
        self.num_idx_heads = num_index_heads
        self.index_head_dim = index_head_dim
        self.value_head_dim = value_head_dim
        self.rotary_dim = rotary_dim
        self.topk = topk
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.page = block_size
        # Bound by `AiterAttentionMetadataBuilder.build_kv_cache_tensor`:
        # the index-key cache, (blocks, page, index_head_dim).
        self.index_cache: torch.Tensor | None = None
        self.k_only = refresh_variants(force=True).k_only
        from atom.config import get_current_atom_config

        self.dcp_world = get_current_atom_config().decode_context_parallel_size
        if self.k_only:
            from atom.config import kv_cache_torch_dtype

            self.kv_layout = ProKOnlyLayout(kv_cache_torch_dtype(kv_cache_dtype))

    # ------------------------------------------------------------------ rope --
    def _cos_sin(self, positions: torch.Tensor, v: ProVariants):
        """``[T, rotary_dim/2]`` fp32 cos / sin, the YaRN magnitude set by the variant."""
        rope = self.rotary_emb
        cos = rope.cos_cache.view(rope.cos_cache.shape[0], -1)
        sin = rope.sin_cache.view(rope.sin_cache.shape[0], -1)
        cos = cos.index_select(0, positions).float()
        sin = sin.index_select(0, positions).float()
        aiter_m = float(getattr(rope, "mscale", 1.0))
        if v.mscale == "aiter":
            return cos, sin
        target = 1.0 if v.mscale == "hf" else float(v.mscale)
        ratio = target / aiter_m
        return cos * ratio, sin * ratio

    def _rope(
        self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, style="neox"
    ):
        """Partial rope on the leading ``rotary_dim`` dims of ``[T, n, d]``."""
        r = self.rotary_dim
        h = r // 2
        xf = x.float()
        c, s = cos.unsqueeze(1), sin.unsqueeze(1)
        if style == "gptj":
            xr = xf[..., :r].reshape(*xf.shape[:-1], h, 2)
            x1, x2 = xr[..., 0], xr[..., 1]
            rot = torch.stack([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1).flatten(-2)
            out = torch.cat([rot, xf[..., r:]], dim=-1)
        else:
            x1, x2 = xf[..., :h], xf[..., h:r]
            out = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s, xf[..., r:]], dim=-1)
        return out.to(x.dtype)

    # -------------------------------------------------------- block selection --
    def _select_blocks(self, bmax: torch.Tensor, qblock: torch.Tensor, v: ProVariants):
        """Selected blocks, ``[..., NB]`` bool, from per-head block scores.

        ``bmax`` is ``[..., NI, NB]`` (``-inf`` where a block holds no visible key),
        ``qblock`` the block index of each query (``[...]``).
        """
        score = _reduce_heads(bmax, v.head_reduce, 1.0)
        init = self.init_blocks if v.init_blocks is None else v.init_blocks
        local = self.local_blocks if v.local_blocks is None else v.local_blocks
        return self._select_from_score(score, qblock, init, local)

    def _select_from_score(
        self,
        score: torch.Tensor,
        qblock: torch.Tensor,
        init: int,
        local: int,
        topk: int | None = None,
    ):
        """Top-``topk`` (default ``self.topk``) valid blocks of ``score`` (``[..., NB]``),
        ``init`` first and ``local`` blocks ending at the query's own forced in:
        ``[..., NB]`` bool."""
        nb = score.shape[-1]
        j = torch.arange(nb, device=score.device)
        qb = qblock.unsqueeze(-1)
        valid = j <= qb
        score = torch.where(
            valid & (j < init), torch.full_like(score, _PIN_INIT), score
        )
        score = torch.where(
            valid & (j >= qb + 1 - local), torch.full_like(score, _PIN_LOCAL), score
        )
        score = torch.where(valid, score, torch.full_like(score, _NEG_INF))
        k = min(self.topk if topk is None else topk, nb)
        idx = score.topk(k, dim=-1).indices
        sel = torch.zeros_like(valid)
        sel.scatter_(-1, idx, True)
        return sel & valid

    def _centroids(self, ik: torch.Tensor, v: ProVariants) -> torch.Tensor:
        """Softmax-pool each 4 consecutive index keys (weights from their dot with the
        learned ``index_compress_query``): ``[..., L, DI] -> [..., L / 4, DI]``."""
        cq = self.index_compress_query.data.float()
        g = ik.float().reshape(*ik.shape[:-2], ik.shape[-2] // 4, 4, ik.shape[-1])
        w = torch.softmax((g * cq).sum(-1) * v.pool_scale, dim=-1)
        return (w.unsqueeze(-1) * g).sum(-2)

    def _block_max(self, iq: torch.Tensor, ik: torch.Tensor, visible: torch.Tensor, v):
        """``[R, NI, NB]`` per-head block scores of ``R`` queries against ``ik``.

        ``iq`` is ``[R, NI, DI]``, ``ik`` ``[R, L, DI]`` (or ``[L, DI]`` shared by all
        rows) with ``L`` a multiple of the page, ``visible`` the ``[R, L]`` bool of keys
        a query may see.
        """
        scale = v.index_scale
        if v.block_score == "pool4":
            # 4 centroids a block; blocks past the query's are masked by the selector
            # and the query's own block is forced, so no token-level mask is needed.
            cent = self._centroids(ik, v)
            if cent.dim() == 2:
                dots = torch.einsum("rnd,cd->rnc", iq.float(), cent)
            else:
                dots = torch.einsum("rnd,rcd->rnc", iq.float(), cent)
            r, ni, c = dots.shape
            return (
                (dots * scale)
                .view(r, ni, c // (self.page // 4), self.page // 4)
                .amax(dim=-1)
            )
        if ik.dim() == 2:
            dots = torch.einsum("rnd,ld->rnl", iq.float(), ik.float())
        else:
            dots = torch.einsum("rnd,rld->rnl", iq.float(), ik.float())
        if v.block_score == "lse":
            scale = scale * self.index_head_dim**-0.5
        dots = (dots * scale).masked_fill(~visible.unsqueeze(1), _NEG_INF)
        r, ni, ll = dots.shape
        dots = dots.view(r, ni, ll // self.page, self.page)
        if v.block_score == "lse":
            return torch.logsumexp(dots, dim=-1)
        return dots.amax(dim=-1)

    # -------------------------------------------------------------- attention --
    def _v_of(self, kf: torch.Tensor, v: ProVariants) -> torch.Tensor:
        """V from K (the shared projection): drop the rope dims."""
        dv = self.value_head_dim
        if v.v_slice == "tail":
            return kf[..., self.head_dim - dv :]
        if v.v_slice == "head":
            return kf[..., :dv]
        raise ValueError(f"unknown v_slice {v.v_slice!r}")

    def _v_view(self, v_cache: torch.Tensor) -> torch.Tensor:
        """The unused V cache as ``[blocks, page, value_head_dim]`` (the first
        ``page * value_head_dim`` elements of each block): where the raw (pre-norm)
        V of the ``v_src="pre"`` variant lives."""
        n = v_cache.shape[0]
        flat = v_cache.view(n, -1)
        return flat[:, : self.page * self.value_head_dim].view(
            n, self.page, self.value_head_dim
        )

    def _read_k(self, kc4, ks, blk, off):
        """K at (block, offset) index tensors, ``[..., D]`` fp32; ``ks`` the per-token
        scales of an fp8 cache (None for bf16)."""
        k = kc4[blk, :, off, :].reshape(*blk.shape, self.head_dim).float()
        if ks is not None:
            k = k * ks[blk, 0, off].unsqueeze(-1)
        return k

    def _write_kv(self, layer, blk, off, k, v: ProVariants):
        """New tokens' K ``[n, D]`` into the K cache and V (K's tail) into the V cache, an
        fp8 cache quantized per token as ``pro_kernels.prep`` does. The ``v_src="pre"``
        experiment keeps its own V in the V cache instead."""
        kc4 = layer.k_cache[:, 0]
        n, d, dv = k.shape[0], self.head_dim, self.value_head_dim
        x = kc4.shape[-1]
        kr = k.reshape(n, d)
        vr = kr[:, d - dv :]
        write_v = v.v_src == "post" and not self.k_only
        if kc4.element_size() == 1:
            from atom.model_ops.minimax_m3.pro_kernels import quant_fp8_rows

            kr, ksc = quant_fp8_rows(kr, kc4.dtype, plain=v.k_plain_fp8)
            layer.k_scale[blk, 0, off] = ksc
            if write_v:
                vr, vsc = quant_fp8_rows(vr, kc4.dtype, plain=v.k_plain_fp8)
                layer.v_scale[blk, 0, off] = vsc
        kc4[blk, :, off, :] = kr.reshape(n, d // x, x)
        if write_v:
            vc = layer.v_cache[:, 0]  # [N, page/x, 256, x]
            xv = vc.shape[-1]
            vc[blk, off // xv, :dv, off % xv] = vr

    def _gather_k(self, kc4, ks, ic, vc, blocks, length):
        """K ``[L, D]`` (fp32), index-K ``[L, DI]`` and (``vc`` given) raw V ``[L, DV]``
        of one sequence from its block list."""
        page = self.page
        pos = torch.arange(length, device=blocks.device)
        blk = blocks[pos // page]
        off = pos % page
        k = self._read_k(kc4, ks, blk, off)
        ik = ic[blk, off] if ic is not None else None
        vraw = vc[blk, off] if vc is not None else None
        return k, ik, vraw

    def _attend_prefill_seq(self, q, iq, qpos, k, ik, v, vraw=None):
        """One sequence's ``tq`` queries over its ``L`` keys (``[L, D]``)."""
        tq = q.shape[0]
        length = k.shape[0]
        h = self.num_heads
        kf = k.float()
        vf = vraw.float() if vraw is not None else self._v_of(kf, v)
        nb = (length + self.page - 1) // self.page
        sparse = (not v.dense_only) and nb > self.topk
        out = q.new_empty(tq, h, self.value_head_dim)
        kpos = torch.arange(length, device=q.device)
        if sparse:
            lpad = nb * self.page
            ikp = torch.zeros(
                lpad, self.index_head_dim, device=q.device, dtype=ik.dtype
            )
            ikp[:length] = ik
            kpos_pad = torch.arange(lpad, device=q.device)
        step = max(1, (1 << 26) // (h * length))
        if sparse:
            step = max(1, min(step, (1 << 26) // (self.num_idx_heads * lpad)))
        for s in range(0, tq, step):
            e = min(tq, s + step)
            qc, pc = q[s:e].float(), qpos[s:e]
            causal = kpos[None, :] <= pc[:, None]
            if sparse:
                visible = kpos_pad[None, :] <= pc[:, None]
                bmax = self._block_max(iq[s:e], ikp, visible, v)
                sel = self._select_blocks(bmax, pc // self.page, v)
                tok = sel.repeat_interleave(self.page, dim=-1)[:, :length]
                mask = causal & tok
            else:
                mask = causal
            scores = torch.einsum("thd,ld->thl", qc, kf) * (self.scale * v.scale_mult)
            scores = scores.masked_fill(~mask.unsqueeze(1), _NEG_INF)
            p = torch.softmax(scores, dim=-1)
            out[s:e] = torch.einsum("thl,lv->thv", p, vf).to(q.dtype)
        return out.reshape(tq, h * self.value_head_dim)

    def _attend_decode(self, q, iq, md, rows, kc4, ks, ic, vc, v):
        """One query per sequence for ``rows`` (``[(token index, kv length, row)]``)."""
        dev = q.device
        r = len(rows)
        h = self.num_heads
        tok = torch.tensor([x[0] for x in rows], device=dev)
        ctx = torch.tensor([x[1] for x in rows], device=dev)
        brow = torch.tensor([x[2] for x in rows], device=dev)
        lmax = max(x[1] for x in rows)
        page = self.page
        nb = (lmax + page - 1) // page
        nblocks = kc4.shape[0]
        bt = md.block_tables[brow, :nb].long().clamp_(0, nblocks - 1)  # [R, NB]
        pos = torch.arange(lmax, device=dev)
        blk_tok = bt.repeat_interleave(page, dim=1)[:, :lmax]
        off = (pos % page).unsqueeze(0).expand(r, -1)
        k = self._read_k(kc4, ks, blk_tok, off)
        vv = vc[blk_tok, off].float() if vc is not None else self._v_of(k, v)
        visible = pos[None, :] < ctx[:, None]
        if (not v.dense_only) and nb > self.topk:
            ikall = ic[bt].reshape(r, nb * page, self.index_head_dim)
            vis_full = torch.arange(nb * page, device=dev)[None, :] < ctx[:, None]
            bmax = self._block_max(iq[tok], ikall, vis_full, v)
            sel = self._select_blocks(bmax, (ctx - 1) // page, v)
            visible = visible & sel.repeat_interleave(page, dim=-1)[:, :lmax]
        scores = torch.einsum("rhd,rld->rhl", q[tok].float(), k) * (
            self.scale * v.scale_mult
        )
        scores = scores.masked_fill(~visible.unsqueeze(1), _NEG_INF)
        p = torch.softmax(scores, dim=-1)
        o = torch.einsum("rhl,rlv->rhv", p, vv)
        return tok, o.reshape(r, h * self.value_head_dim).to(q.dtype)

    # ------------------------------------------------------------ kernel path --
    def _aiter_prep(self, v: ProVariants) -> bool:
        """aiter's prep: it writes no NVFP4 index rows, nor the keys the compressed index
        cache pools; the plain fp8 K needs its ``kv_unit_scale``."""
        return (
            v.prep == "aiter"
            and not v.index_compress
            and not self._index_nvfp4()
            and pro_kernels.aiter_prep_available(k_unit=v.k_plain_fp8)
        )

    def _index_nvfp4(self) -> bool:
        """``--index-cache-dtype nvfp4``: uint8 pages the Triton kernels alone read."""
        ic = self.index_cache
        return ic is not None and ic.dtype == torch.uint8

    def _ring_buffers(self, rows: int, device, dtype):
        """This layer's rings of raw index keys and their rows' positions (-1: empty) for
        the compressed index cache, allocated on first use (the warmup forward, before
        the KV budget is measured and any graph is captured)."""
        ring = getattr(self, "_ring", None)
        if ring is None or ring.shape[0] != rows:
            self._ring = torch.zeros(
                rows, pro_kernels.IDX_DIM, dtype=dtype, device=device
            )
            self._ring_pos = torch.full((rows,), -1, dtype=torch.int32, device=device)
        return self._ring, self._ring_pos

    def _aiter_attn(self, v: ProVariants, n: int = _K_ROWS_AITER_MIN) -> bool:
        """aiter's paged attention for ``n`` rows: on the K-only cache only where it reads V
        from the K rows (``value_rows``), and from ``_K_ROWS_AITER_MIN`` rows up."""
        return (
            v.attn == "aiter"
            and self.dcp_world == 1
            and (
                not self.k_only
                or (n >= _K_ROWS_AITER_MIN and pro_kernels.aiter_value_rows_available())
            )
        )

    def _day0_decode(self, v: ProVariants, n: int, reqs: int, fp8: bool) -> bool:
        """amd-kernels-day0's FlyDSL sparse decode for a decode / verify batch of ``n`` rows
        over ``reqs`` requests: from ``day0_decode_rows`` rows, on the K-only plain fp8 cache,
        rows request-major (``n`` a multiple of ``reqs``). The shapes alone decide (graph-safe).
        """
        return (
            0 < v.day0_decode_rows <= n
            and not v.dense_only
            and self.k_only
            and fp8
            and v.k_plain_fp8
            and v.topk in ("fused", "split")
            and self.num_heads == 16
            and self.topk == pro_kernels.TOPK
            and reqs > 0
            and n % reqs == 0
            and pro_kernels.day0_decode_available()
        )

    def take_mx_scale(self) -> torch.Tensor:
        """The E8M0 scales ``[T, H x 192 / 32]`` of the MXFP8 output this forward returned (the
        model hands them to o_proj with it), dropped here."""
        scale, self._mx_scale = self._mx_scale, None
        return scale

    def _mxfp8_out(self, v: ProVariants, n: int, day0_dec: bool = False) -> bool:
        """Whether a decode / verify batch of ``n`` rows writes o_proj's MXFP8 input (E4M3
        rows, their E8M0 scales through ``take_mx_scale``) instead of bf16, where o_proj takes
        it (``mx_out``). ``day0_dec``: the batch is the day0 decode's (``_day0_decode``),
        whose epilogue writes it under ``day0_decode_mxfp8``. Below that, under
        ``fuse_mxfp8_quant``: the Triton attention always, aiter's where its gluon decode
        takes ``output_scale``, unless the call is ``bf16_out``'s. Off under decode context
        parallelism (``dcp_world`` > 1), the day0 decode included."""
        if self.dcp_world > 1:
            return False
        if not self.mx_out:
            return False
        if day0_dec:
            return v.day0_decode_mxfp8
        return (
            not self.bf16_out
            and v.fuse_mxfp8_quant
            and self.value_head_dim % 32 == 0
            and (not self._aiter_attn(v, n) or pro_kernels.aiter_mxfp8_out_available())
        )

    def _use_kernels(self, v: ProVariants) -> bool:
        """Whether the Triton kernels cover this layer and these variants."""
        return (
            pro_kernels is not None
            and v.kernels
            and not v.diag
            and v.v_src == "post"
            and v.v_slice == "tail"
            and v.rope_style == "neox"
            and v.index_rope
            and v.block_score in ("max", "pool4", "lse")
            and v.head_reduce in ("sum", "mean", "max", "lse")
            and v.prep in ("triton", "aiter")
            and self.num_heads % 16 == 0
            # Triton's prep holds all of a token's heads in one program
            and (self.num_heads <= 32 or self._aiter_prep(v))
            and (self.head_dim, self.rotary_dim, self.value_head_dim) == (256, 64, 192)
            and (self.num_idx_heads, self.index_head_dim, self.page) == (16, 128, 16)
            and v.attn_nblk in (1, 2, 4, 8)
            and v.attn in ("triton", "aiter")
            and (
                v.topk in ("torch", "aiter")
                or (
                    v.topk in ("fused", "split")
                    and (v.attn == "aiter" or self.k_only)
                    and self.topk & (self.topk - 1) == 0
                )
            )
        )

    def _fused_index(
        self, v: ProVariants, static: bool, split, ic, n: int, dsplit=None
    ) -> bool:
        """Whether the decode scores take aiter's FlyDSL scorer (``"index":
        "fused"``): graph-shaped decode rows of one rank, neither an index split
        (``split``) nor a decode split (``dsplit``), pool4 / max scores, a
        selection that reads only each row's causal blocks, and an index cache ``ic``
        it reads: fp8 (from _FUSED_MIN_ROWS_FP8 of the ``n`` rows) or NVFP4, or with
        ``index_compress`` the bf16 centroids (a bf16 token cache it does not read)."""
        if v.index != "fused":
            return False
        centroid = v.index_compress
        ok = (
            static
            and split is None
            and dsplit is None
            and (
                (centroid and ic.dtype == torch.bfloat16)
                or (
                    not centroid
                    and (
                        ic.dtype == torch.uint8
                        or (
                            ic.dtype == torch.float8_e4m3fn and n >= _FUSED_MIN_ROWS_FP8
                        )
                    )
                )
            )
            and self.dcp_world == 1
            and v.block_score == "pool4"
            and v.head_reduce == "max"
            and v.topk != "torch"
            and pro_kernels.fused_scores_available(centroid)
        )
        small_fp8 = ic.dtype == torch.float8_e4m3fn and n < _FUSED_MIN_ROWS_FP8
        if (
            static
            and not ok
            and not small_fp8
            and not getattr(self, "_warned_index", False)
        ):
            self._warned_index = True
            logger.warning(
                "M3-Pro layer %d: index 'fused' needs pool4 / max scores, a topk "
                "other than 'torch', an fp8, NVFP4 or compressed index cache, no DCP, "
                "no index split and aiter's FlyDSL Pro scorer (with the centroid "
                "format for index_compress) on gfx950; decode scores stay Triton's",
                self.layer_num,
            )
        return ok

    @staticmethod
    def _device_meta(md, n: int, device):
        """Per-forward device tensors shared by all layers: the sequence of every token
        (padding rows past the last sequence get the block table's last row).

        Cached per capture state: a CUDA graph capture runs on the metadata its warmup
        forward used, and must compute the map itself -- the warmup's tensor is not the
        graph's, and is freed (and its memory reused) once that metadata goes."""
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        cached = getattr(md, "_m3pro_dev", None)
        if cached is None or cached[0] != capturing or cached[1].shape[0] != n:
            cu = md.cu_seqlens_q
            idx = torch.arange(n, device=device, dtype=cu.dtype)
            seq = torch.searchsorted(cu[1:], idx, right=True)
            seq = seq.clamp_(max=md.block_tables.shape[0] - 1)
            md._m3pro_dev = cached = (capturing, seq)
        return cached[1]

    def _kernel_attention(
        self,
        qkv,
        positions,
        seq_id,
        slots,
        block_tables,
        layer,
        ic,
        nb: int,
        v: ProVariants,
        out,
        static: bool = False,
        split=None,
        mx_scale=None,
        dsplit=None,
        ring=None,
    ):
        """The whole layer in four launches (``prep`` / ``scores`` + top-k / ``attention``).

        ``qkv`` the raw projections of ``n`` rows (packed, or the ``(q, k, index q, index k)``
        tuple ``prep`` takes), ``positions`` / ``seq_id`` / ``slots`` ``[n]``,
        ``layer`` the layer's caches, ``nb`` the most blocks any sequence of the batch
        holds, ``out`` ``[n, H * 192]``. ``static``: ``nb`` is only a bound (the block
        table's width), so the selection runs even where every block would fit.
        ``split``: this forward's ``index_split`` plan; the indexer scores this rank's
        rows only (the other rows' index q may be unset). ``mx_scale`` ``[n, H x 192 / 32]``
        uint8 (with ``out`` E4M3): the decode attention writes o_proj's MXFP8 input
        (``_mxfp8_out``). ``dsplit``:
        a decode's ``(first, end)`` rows this rank selects (``decode_split``). ``ring``:
        ``(each batch row's first ring row, the ring rows)`` of the compressed index cache
        (``index_compress``; ``ic`` is then its ``[pages, 4, 128]`` centroids).
        """
        kc4 = layer.k_cache[:, 0]
        vc4 = None if self.k_only else layer.v_cache[:, 0]
        fp8 = kc4.element_size() == 1
        ks = layer.k_scale if fp8 else None
        vs = layer.v_scale if fp8 and vc4 is not None else None
        n = positions.shape[0]
        aiter_attn = self._aiter_attn(v, n)
        rope = self.rotary_emb
        cos = rope.cos_cache.view(rope.cos_cache.shape[0], -1)
        sin = rope.sin_cache.view(rope.sin_cache.shape[0], -1)
        aiter_m = float(getattr(rope, "mscale", 1.0))
        if v.mscale == "aiter":
            ratio = 1.0
        else:
            ratio = (1.0 if v.mscale == "hf" else float(v.mscale)) / aiter_m
        eps = self.q_norm.variance_epsilon
        k_unit = fp8 and v.k_plain_fp8
        prep = functools.partial(pro_kernels.prep, k_unit=k_unit)
        if self._aiter_prep(v):
            if pro_kernels.prep_aiter_fits(qkv, self.num_heads):
                prep = functools.partial(pro_kernels.prep_aiter, k_unit=k_unit)
            elif not getattr(self, "_warned_prep", False):
                self._warned_prep = True
                logger.warning(
                    "M3-Pro layer %d: unaligned projection rows, Triton prep",
                    self.layer_num,
                )
        compress = v.index_compress
        if compress:
            dt = qkv[0].dtype if isinstance(qkv, tuple) else qkv.dtype
            prep = functools.partial(
                prep,
                ik_out=torch.empty(n, pro_kernels.IDX_DIM, dtype=dt, device=ic.device),
            )
        q, iq = prep(
            qkv,
            positions,
            slots,
            cos,
            sin,
            ratio,
            self.q_norm.weight.data,
            self.k_norm.weight.data,
            self.index_q_norm.weight.data,
            self.index_k_norm.weight.data,
            eps,
            kc4,
            ic,
            self.num_heads,
            vc4=vc4,
            k_scale=ks,
            v_scale=vs,
        )
        if self.dcp_world > 1 and not static:
            # prep wrote this rank's blocks; the prefill reads every rank's
            kc4, ks, ic, block_tables = self._dcp_prefill_caches(kc4, ks, ic)
        if compress:
            # the groups this step closes, before any score reads them (a prefill chunk's
            # later rows read its earlier, older-than-local blocks)
            base, rows = ring if ring is not None else (None, 0)
            if base is None:
                base = torch.full(
                    (block_tables.shape[0],), -1, dtype=torch.int32, device=ic.device
                )
            rbuf, rpos = self._ring_buffers(max(rows, 1), ic.device, q.dtype)
            pro_kernels.compress_insert(
                prep.keywords["ik_out"],
                positions,
                seq_id,
                slots,
                base,
                rbuf,
                rpos,
                self.index_compress_query.data,
                v.pool_scale,
                ic,
            )
        scale = self.scale * v.scale_mult
        nblk = v.attn_nblk
        # MI355X, decode over top-128 (opt/kv/bench_decode_attn.py, valid_first, nblk 4):
        # 32 / 16 / 8 splits at 1-16 / 32 / 64 rows put bs1 at aiter's 13 us (from 26)
        splits = v.attn_splits or (
            32
            if n <= 16
            else (
                16
                if n <= 32
                else 8 if n <= 64 else 4 if n <= 128 else 2 if n <= 256 else 1
            )
        )

        day0 = (
            not static
            and v.prefill_attn == "day0"
            and self.k_only
            and fp8
            and v.k_plain_fp8
            # its picks: valid first, -1 after (select_logical)
            and v.topk in ("fused", "split")
            and self.num_heads == 16
            and self.topk == pro_kernels.TOPK
            and pro_kernels.day0_prefill_available(kc4.shape[-2])
        )

        # decode / verify: FlyDSL sparse decode from day0_decode_rows rows
        day0_dec = static and self._day0_decode(v, n, block_tables.shape[0], fp8)
        assert mx_scale is None or static, "prefill attention writes bf16"

        valid_first = False  # set where the selection is select_logical's

        def attend(sel, rows=None):
            if day0_dec:
                assert rows is None, "the day0 decode takes block ids (select_logical)"
                return pro_kernels.attention_day0_decode(
                    q,
                    kc4,
                    block_tables,
                    positions,
                    slots,
                    sel,
                    out,
                    scale,
                    n // block_tables.shape[0],
                    out_scale=mx_scale,
                )
            if day0:
                assert (
                    rows is None
                ), "day0 takes block ids (select_logical), not page tables"
                return pro_kernels.attention_day0(
                    q, kc4, block_tables, seq_id, positions, sel, out, scale
                )
            if aiter_attn:
                if rows is None:
                    rows = pro_kernels.sparse_rows(
                        sel, block_tables, seq_id, positions, nb
                    )
                rows, ctx = rows
                return pro_kernels.attention_aiter(
                    q,
                    layer.k_cache,
                    layer.v_cache,
                    rows,
                    ctx,
                    out,
                    scale,
                    k_scale=ks,
                    v_scale=vs,
                    fp8_compute=v.attn_fp8_compute,
                    k_rows=self.k_only,
                    out_scale=mx_scale,
                )
            return pro_kernels.attention(
                q,
                kc4,
                block_tables,
                seq_id,
                positions,
                sel,
                out,
                scale,
                splits,
                nblk=nblk,
                n_blocks=nb if sel is None else None,
                k_scale=ks,
                valid_first=valid_first,
                out_scale=mx_scale,
            )

        if v.dense_only or (nb <= self.topk and not static):
            # every causal block is attended (the selector would keep them all anyway)
            return attend(None)
        init = self.init_blocks if v.init_blocks is None else v.init_blocks
        local = self.local_blocks if v.local_blocks is None else v.local_blocks
        pool = v.block_score == "pool4"
        block_mode = {
            "max": pro_kernels.BLOCK_TOKMAX,
            "pool4": pro_kernels.BLOCK_POOL4MAX,
            "lse": pro_kernels.BLOCK_TOKLSE,
        }[v.block_score]
        if v.index_compress:
            # ``ic`` is the compressed cache: its rows are the blocks' pool4 centroids,
            # so the block score is their max (pool4 / max on precomputed centroids)
            assert v.block_score == "pool4", "index_compress replaces the pool4 scores"
            block_mode, pool = pro_kernels.BLOCK_CENTROID, False
        head_mode = {
            "sum": pro_kernels.HEAD_SUM,
            "mean": pro_kernels.HEAD_SUM,
            "max": pro_kernels.HEAD_MAX,
            "lse": pro_kernels.HEAD_LSE,
        }[v.head_reduce]
        sc = v.index_scale * (
            self.index_head_dim**-0.5 if v.block_score == "lse" else 1.0
        )
        if self.dcp_world > 1 and static:
            score_args = (pool, init, local, sc, block_mode, head_mode)
            return self._dcp_decode(
                q, iq, positions, seq_id, block_tables, kc4, ks, ic, v, out, score_args
            )
        k = min(self.topk, nb)
        rq, rseq, rpos = iq, seq_id, positions  # the rows the indexer scores
        rps = n // block_tables.shape[0]  # rows a sequence (static: k + 1 for a verify)
        if dsplit is not None:
            # whole sequences a rank (decode_split), so no score tile spans two of them
            r0, r1 = dsplit
            rq, rseq, rpos = iq[r0:r1], seq_id[r0:r1], positions[r0:r1]
        if split is not None:
            # every row scores alone, so this rank's rows come out as they would in full
            idx, seg, n_rows, tp = split
            rq, rseq, rpos = iq[idx], seq_id[idx], positions[idx]
        if self._fused_index(v, static, split, ic, n, dsplit):
            # each group of `rows` rows is one sequence's: k + 1 a verify, so the
            # largest power of two (at most 8) dividing them
            rows = min(8, rps & -rps) if rps > 1 else 1
            score = pro_kernels.scores_fused(
                rq,
                ic,
                block_tables,
                rseq,
                rpos,
                nb,
                self.index_compress_query.data,
                init,
                local,
                sc,
                v.pool_scale,
                rows,
            )
        else:
            # decode: one row a sequence, nothing to share; a verify has k + 1 rows a
            # sequence, so the tile is the largest power of two dividing them (at most
            # score_tq): a tile never spans two sequences, which would score row by row
            # (k = 3: 4 rows, not 8). Graph-safe: the shapes alone decide.
            tq = (
                v.score_tq
                if not static
                else min(v.score_tq, rps & -rps) if rps > 1 else 0
            )
            score = pro_kernels.scores(
                rq,
                ic,
                block_tables,
                rseq,
                rpos,
                nb,
                self.index_compress_query.data if pool else None,
                init,
                local,
                sc,
                block_mode=block_mode,
                head_mode=head_mode,
                pool_temp=v.pool_scale,
                # torch.topk reads the whole width; the others only each token's causal
                # blocks (the score_tq kernel only with score_tq_causal)
                causal=v.topk != "torch" and (v.score_tq_causal or tq == 0),
                tq=tq,
            )
        # select_rows_split matches select_rows bit for bit; its cost follows the context
        select = (
            pro_kernels.select_rows_split
            if v.topk == "split"
            else pro_kernels.select_rows
        )
        # day0 (prefill and decode) and Triton take block ids, aiter pages
        logical = day0 or day0_dec or not aiter_attn
        if dsplit is not None:
            from aiter.dist.communication_op import tensor_model_parallel_all_gather

            if logical:
                # block ids for the day0 decode / Triton: each row selects alone, and the
                # shares are contiguous and in rank order, so gathered they are in place
                valid_first = True
                mine = pro_kernels.select_logical(
                    score, rpos, k, split=v.topk == "split"
                )
                return attend(
                    tensor_model_parallel_all_gather(mine, dim=0).contiguous()
                )
            # this rank's [page table | ctx] rows, written in place by the selection
            mine = torch.empty(r1 - r0, k + 1, dtype=torch.int32, device=iq.device)
            if select is pro_kernels.select_rows_split:
                select(
                    score, block_tables, rseq, rpos, k, rows=mine[:, :k], ctx=mine[:, k]
                )
            else:
                rows, ctx = select(score, block_tables, rseq, rpos, k)
                mine[:, :k], mine[:, k] = rows, ctx
            # the shares are contiguous and in rank order: gathered, the rows are in place;
            # aiter takes the page table at any row stride, the lengths contiguous
            full = tensor_model_parallel_all_gather(mine, dim=0)
            return attend(None, (full[:, :k], full[:, k].contiguous()))
        if split is not None:
            from aiter.dist.communication_op import tensor_model_parallel_all_gather

            if logical:
                mine = pro_kernels.select_logical(
                    score, rpos, k, split=v.topk == "split"
                )
                valid_first = True
                return attend(
                    assemble_rows(
                        tensor_model_parallel_all_gather(mine, dim=0), seg, n_rows, tp
                    ).contiguous()
                )
            rows, ctx = select(score, block_tables, rseq, rpos, k)
            mine = torch.cat([rows, ctx[:, None]], dim=1)
            full = assemble_rows(
                tensor_model_parallel_all_gather(mine, dim=0), seg, n_rows, tp
            )
            return attend(None, (full[:, :k].contiguous(), full[:, k].contiguous()))
        if v.topk in ("fused", "split") and k & (k - 1) == 0:
            if logical:
                valid_first = True
                return attend(
                    pro_kernels.select_logical(
                        score, positions, k, split=v.topk == "split"
                    )
                )
            return attend(None, select(score, block_tables, seq_id, positions, k))
        if v.topk == "aiter":
            sel = pro_kernels.select_blocks(score, positions, k)
        else:
            sel = score.topk(k, dim=-1, sorted=False).indices.to(torch.int32)
        return attend(sel)

    # ------------------------------------------------- decode context parallel --
    def _dcp_prefill_caches(self, kc4, ks, ic):
        """A prefill's caches under DCP: every block of its sequences, all-gathered from
        the ranks that hold them, and the block tables into those (the plan is the
        forward's, the gather each layer's)."""
        from atom.distributed.dcp_utils import get_dcp_group
        from atom.model_ops.dcp_ops import dcp_all_gather

        md = get_forward_context().attn_metadata
        group, w = get_dcp_group(), self.dcp_world
        plan = getattr(md, "_m3pro_dcp_plan", None)
        if plan is None:
            plan = pro_kernels.dcp_prefill_plan(
                md.block_tables, self._cpu_lengths(md)[1], w
            )
            md._m3pro_dcp_plan = plan
        rows, tables = plan

        def gather(x):
            return pro_kernels.dcp_gather_blocks(
                x, rows, lambda t: dcp_all_gather(group, t, 0), w
            )

        return gather(kc4), None if ks is None else gather(ks), gather(ic), tables

    def _dcp_decode(
        self, q, iq, positions, seq_id, block_tables, kc4, ks, ic, v, out, score_args
    ):
        """A decode step under DCP: every rank scores its own blocks, the scores are
        all-gathered so each rank selects from all of them as without DCP, each rank
        attends all query heads over the selected blocks it holds, and a log-sum-exp
        merge returns this rank's heads."""
        from atom.config import get_current_atom_config
        from atom.distributed.dcp_utils import get_dcp_group
        from atom.model_ops.dcp_ops import (
            dcp_all_gather,
            dcp_all_gather_query_heads,
            dcp_lse_merge,
        )

        pool, init, local, sc, block_mode, head_mode = score_args
        group = get_dcp_group()
        w, r = self.dcp_world, group.rank_in_group
        n = q.shape[0]
        md = get_forward_context().attn_metadata
        capturing = (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        )
        cached = getattr(md, "_m3pro_dcp_pos", None)
        if cached is None or cached[0] != capturing or cached[1].shape[0] != n:
            md._m3pro_dcp_pos = cached = (
                capturing,
                pro_kernels.dcp_local_positions(positions, r, w),
            )
        qpos = cached[1]
        nbl = -(-block_tables.shape[1] // w)
        score = pro_kernels.scores(
            iq,
            ic,
            block_tables,
            seq_id,
            qpos,
            nbl,
            self.index_compress_query.data if pool else None,
            0,
            0,
            sc,
            block_mode=block_mode,
            head_mode=head_mode,
            pool_temp=v.pool_scale,
            causal=True,
            tq=v.score_tq if n > block_tables.shape[0] else 0,
        )
        pro_kernels.dcp_pin(score, positions, r, w, init, local)
        score = pro_kernels.dcp_interleave(dcp_all_gather(group, score, 0), w)
        sel = pro_kernels.select_logical(
            score, positions, min(self.topk, nbl * w), split=v.topk == "split"
        )
        sel = pro_kernels.dcp_local_selection(sel, r, w)
        qf = dcp_all_gather_query_heads(group, q)
        h = qf.shape[1]
        dv = self.value_head_dim
        of = q.new_empty(n, h * dv)
        lse = torch.empty(n, h, dtype=torch.float32, device=q.device)
        # a rank holds about 1/w of the selected blocks, for w times the heads
        splits = max(
            1, (v.attn_splits or (32 if n <= 16 else 16 if n <= 32 else 8)) // w
        )
        pro_kernels.attention(
            qf,
            kc4,
            block_tables,
            seq_id,
            qpos,
            sel,
            of,
            self.scale * v.scale_mult,
            splits,
            nblk=v.attn_nblk,
            k_scale=ks,
            valid_first=True,
            lse=lse,
        )
        # the merge takes power-of-two head widths: each head's 192 as three of 64
        parts = dv // 64
        merged = dcp_lse_merge(
            of.view(n, h * parts, 64),
            lse.repeat_interleave(parts, dim=1),
            group,
            backend=get_current_atom_config().dcp_config.comm_backend,
        )
        out.copy_(merged.reshape(n, -1))
        return out

    # ---------------------------------------------------------------- forward --
    def decode_split(self, n: int, v: ProVariants, seqs: int = 0):
        """This rank's ``(first, end)`` rows of a graph's ``n`` decode rows of ``seqs``
        sequences under ``index_split_decode``, or ``None``: every rank selects every row.
        A rank's share holds whole sequences (``n / seqs`` rows each). The shapes alone
        decide, so every rank and layer agrees and a graph holds it."""
        if not v.index_split_decode or n < v.index_split_decode:
            return None
        if not self._aiter_attn(v, n) or v.topk not in ("fused", "split"):
            return None
        from atom.distributed.dcp_utils import get_dcp_world_size

        try:
            if (
                get_dcp_world_size() > 1
            ):  # DCP shards the index cache by block, not rows
                return None
        except AssertionError:  # no ATOM config (unit tests): no DCP
            pass
        from aiter.dist.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_world_size,
        )

        tp = get_tensor_model_parallel_world_size()
        rps = n // seqs if seqs else 1
        if tp == 1 or n % tp or (n // tp) % rps:
            return None
        share = n // tp
        rank = get_tensor_model_parallel_rank()
        return rank * share, (rank + 1) * share

    def index_split(self, n_tok: int):
        """This forward's ``index_split`` plan, ``(idx, seg, n, tp)`` (``row_split_index``
        of the forward's ``n`` rows for this rank), or ``None``: every rank scores every
        row. Decided once a forward (the first layer's call, before its projections) from
        the host lengths alone, so every rank and layer agrees."""
        fwd = get_forward_context()
        md = fwd.attn_metadata
        plan = getattr(md, "_m3pro_split", False)
        if plan is not False:
            return plan
        plan = None
        v = refresh_variants()
        if (
            v.index_split
            and fwd.context.is_prefill
            and not fwd.context.is_dummy_run
            and self._use_kernels(v)
            and v.attn == "aiter"
            and v.topk in ("fused", "split")
            and not v.dense_only
        ):
            from aiter.dist.parallel_state import (
                get_tensor_model_parallel_rank,
                get_tensor_model_parallel_world_size,
            )

            tp = get_tensor_model_parallel_world_size()
            cu, ctx = self._cpu_lengths(md)
            n = min(n_tok, int(cu[-1]))
            nb = (max(ctx, default=0) + self.page - 1) // self.page
            if tp > 1 and n >= _SPLIT_MIN_TOKENS and nb > self.topk:
                idx, seg = row_split_index(
                    n, tp, get_tensor_model_parallel_rank(), md.slot_mapping.device
                )
                plan = (idx, seg, n, tp)
        md._m3pro_split = plan
        return plan

    def _cpu_lengths(self, md):
        """``(cu_seqlens_q, context_lens)`` as lists, synced once per forward."""
        cached = getattr(md, "_m3pro_cpu", None)
        if cached is None:
            cu = md.cu_seqlens_q.cpu().tolist()
            ctx = md.context_lens.cpu().tolist()
            cached = (cu, ctx)
            md._m3pro_cpu = cached
        return cached

    def forward(
        self,
        query,
        key,
        value,
        kv_cache=None,
        attn_metadata=None,
        position=None,
        q_scale=None,
        qkv=None,
        output=None,
        **kwargs,
    ):
        fwd = get_forward_context()
        h, d = self.num_heads, self.head_dim
        n_tok = query.shape[0]
        if fwd.context.is_dummy_run:
            return query.new_zeros(n_tok, h * self.value_head_dim)
        parts = (query, key, value, qkv)
        if self.layer_num == 0:
            refresh_variants()
        v = variants()
        if v.diag and not v.dense_only:
            v = replace(v, dense_only=True)  # the diagnostic reads the dense trajectory

        md = fwd.attn_metadata
        if self.dcp_world > 1 and (not self._use_kernels(v) or v.dense_only):
            raise NotImplementedError(
                "MiniMax-M3-Pro under --decode-context-parallel-size needs the kernel "
                'path ({"kernels": true} in ATOM_M3PRO_VARIANTS) and the indexer'
            )
        if self._use_kernels(v):
            layer = fwd.kv_cache_data[f"layer_{self.layer_num}"]
            prep_impl = "aiter" if self._aiter_prep(v) else "triton"
            attn_impl = (
                ("aiter (K rows) past 16 rows, else triton" if self.k_only else "aiter")
                if self._aiter_attn(v)
                else "triton"
            )
            if (
                v.day0_decode_rows
                and self.k_only
                and pro_kernels.day0_decode_available()
            ):
                attn_impl += f"; day0 decode from {v.day0_decode_rows} rows" + (
                    " (MXFP8 out)"
                    if self._mxfp8_out(v, v.day0_decode_rows, day0_dec=True)
                    else ""
                )
            # the Triton / aiter rows, below the day0 decode
            mx_impl = (
                "off"
                if not (v.fuse_mxfp8_quant and self.mx_out and self.dcp_world == 1)
                else (
                    "on"
                    if self._mxfp8_out(v, _K_ROWS_AITER_MIN)
                    else (
                        "on Triton rows only (this aiter's decode writes bf16)"
                        if self.k_only
                        else "off (this aiter's decode writes bf16)"
                    )
                )
            )
            logged = (attn_impl, prep_impl, mx_impl)
            if self.layer_num == 0 and getattr(self, "_logged_kernels", None) != logged:
                self._logged_kernels = logged
                logger.info(
                    "M3-Pro attention: kernels, %s prep, %s attention, %s%s KV cache, "
                    "Triton / aiter decode MXFP8 out %s (%s)",
                    prep_impl,
                    attn_impl,
                    "K-only " if self.k_only else "",
                    layer.k_cache.dtype,
                    mx_impl,
                    v,
                )
            static = not fwd.context.is_prefill
            if static:
                # decode reads nothing back from the host (a graph can hold it): every
                # row (padding rows have slot -1 and write nothing), the block table's
                # full width, and the selection whatever the lengths
                n, nb = min(n_tok, md.slot_mapping.shape[0]), md.block_tables.shape[1]
            else:
                cu, ctx = self._cpu_lengths(md)
                n = min(n_tok, int(cu[-1]))
                if n == 0:
                    return query.new_zeros(n_tok, h * self.value_head_dim)
                nb = (max(ctx) + self.page - 1) // self.page
            # the decode attention writes o_proj's MXFP8 input where o_proj takes it (mx_out,
            # set by the model): E4M3 rows here, their E8M0 scales through take_mx_scale
            mx = static and self._mxfp8_out(
                v,
                n,
                self._day0_decode(
                    v, n, md.block_tables.shape[0], layer.k_cache.element_size() == 1
                ),
            )
            # the kernels write every row they compute in full: zero only the others
            width = h * self.value_head_dim
            out = query.new_empty(
                n_tok, width, dtype=torch.float8_e4m3fn if mx else query.dtype
            )
            mx_scale = (
                query.new_empty(n_tok, width // 32, dtype=torch.uint8) if mx else None
            )
            if n < n_tok:
                out.view(torch.uint8)[n:].zero_()  # fp8 or bf16: all-zero bytes are 0
                if mx:
                    mx_scale[n:].zero_()
            seq_id = self._device_meta(md, n, query.device)
            self._kernel_attention(
                tuple(p[:n] for p in parts),
                position[:n].long(),
                seq_id,
                md.slot_mapping[:n],
                md.block_tables,
                layer,
                self.index_cache,
                nb,
                v,
                out[:n],
                static=static,
                split=None if static else self.index_split(n_tok),
                mx_scale=None if mx_scale is None else mx_scale[:n],
                dsplit=(
                    self.decode_split(n, v, md.block_tables.shape[0])
                    if static
                    else None
                ),
                ring=(
                    getattr(md, "pro_ring_base", None),
                    getattr(md, "pro_ring_rows", 0),
                ),
            )
            if mx:
                self._mx_scale = mx_scale.view(torch.float8_e8m0fnu)
            return out

        if self._index_nvfp4() or v.index_compress:
            raise NotImplementedError(
                'an NVFP4 or compressed index cache needs the kernel path ({"kernels": true})'
            )
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                'MiniMax-M3-Pro decode graphs need {"kernels": true} in ATOM_M3PRO_VARIANTS '
                "(the torch attention reads the lengths back to the host), or --enforce-eager"
            )
        out = query.new_zeros(n_tok, h * self.value_head_dim)
        cu, ctx = self._cpu_lengths(md)
        n = min(n_tok, int(cu[-1]))
        if n == 0:
            return out
        positions = position[:n].long()
        q = query[:n].reshape(n, h, d)
        k = key[:n].reshape(n, 1, d)
        iq = value[:n].reshape(n, self.num_idx_heads, -1)
        ik = qkv[:n].reshape(n, 1, self.index_head_dim)

        k_raw = k  # the raw k_proj output, before k_norm (the "pre" V source)
        eps = self.q_norm.variance_epsilon
        q = GemmaRMSNorm.forward_static(self.q_norm.weight.data, eps, q, None)
        k = GemmaRMSNorm.forward_static(self.k_norm.weight.data, eps, k, None)
        iq = GemmaRMSNorm.forward_static(self.index_q_norm.weight.data, eps, iq, None)
        ik = GemmaRMSNorm.forward_static(self.index_k_norm.weight.data, eps, ik, None)
        cos, sin = self._cos_sin(positions, v)
        q = self._rope(q, cos, sin, v.rope_style)
        k = self._rope(k, cos, sin, v.rope_style)
        if v.index_rope:
            iq = self._rope(iq, cos, sin, v.rope_style)
            ik = self._rope(ik, cos, sin, v.rope_style)

        layer = fwd.kv_cache_data[f"layer_{self.layer_num}"]
        kc4 = layer.k_cache[:, 0]  # [N, D/x, page, x]
        ks = layer.k_scale if kc4.element_size() == 1 else None
        ic = self.index_cache
        page = self.page
        slots = md.slot_mapping[:n].long()
        blk, off = slots // page, slots % page
        if self.layer_num == 0 and not getattr(self, "_logged_cache", False):
            self._logged_cache = True
            logger.info(
                "M3-Pro caches: k %s %s %s, v %s %s %s, index %s",
                tuple(layer.k_cache.shape),
                layer.k_cache.stride(),
                layer.k_cache.dtype,
                tuple(layer.v_cache.shape),
                layer.v_cache.stride(),
                layer.v_cache.dtype,
                tuple(ic.shape),
            )
            # uninitialised (never loaded) parameters would show up as garbage here
            for name, p in (
                ("index_compress_query", self.index_compress_query),
                ("q_norm", self.q_norm.weight),
                ("k_norm", self.k_norm.weight),
                ("index_q_norm", self.index_q_norm.weight),
                ("index_k_norm", self.index_k_norm.weight),
            ):
                pf = p.data.float()
                logger.info(
                    "M3-Pro L0 %s: shape %s absmax %.4f mean %.4f",
                    name,
                    tuple(pf.shape),
                    pf.abs().max().item(),
                    pf.mean().item(),
                )
        self._write_kv(layer, blk, off, k.reshape(n, d), v)
        ic[blk, off] = _index_rows(ik.reshape(n, self.index_head_dim), ic.dtype)
        vc = None
        if v.v_src == "pre":
            if self.k_only:
                raise NotImplementedError("v_src='pre' keeps a V cache: not K-only")
            if ks is not None:
                raise NotImplementedError(
                    "v_src='pre' keeps its V unscaled: bf16 KV cache only"
                )
            vc = self._v_view(layer.v_cache)
            vc[blk, off] = k_raw.reshape(n, d)[:, d - self.value_head_dim :]

        if v.diag and fwd.context.is_prefill:
            from atom.model_ops.minimax_m3 import pro_diag

            pro_diag.collect(
                self, q, k, positions, getattr(self, "_diag_raw", None), cu, ctx, v
            )

        if (not fwd.context.is_prefill) and all(
            cu[b + 1] - cu[b] == 1 for b in range(len(ctx)) if ctx[b] > 0
        ):
            rows = [(cu[b], ctx[b], b) for b in range(len(ctx)) if ctx[b] > 0]
            if rows:
                tok, o = self._attend_decode(q, iq, md, rows, kc4, ks, ic, vc, v)
                out[tok] = o
            return out

        for b in range(len(ctx)):
            s, e = cu[b], cu[b + 1]
            if e <= s or ctx[b] <= 0:
                continue
            length = ctx[b]
            nblk = (length + page - 1) // page
            blocks = md.block_tables[b, :nblk].long()
            kk, ikk, vraw = self._gather_k(kc4, ks, ic, vc, blocks, length)
            qpos = torch.arange(length - (e - s), length, device=q.device)
            out[s:e] = self._attend_prefill_seq(q[s:e], iq[s:e], qpos, kk, ikk, v, vraw)
        return out
