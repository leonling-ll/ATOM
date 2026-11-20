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

import json
import logging
import os
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
    # The attention of the kernel path: "triton" (``pro_kernels.attention``) or "aiter":
    # aiter's paged-attention decode (gluon, as ATOM's M3 runs it) with every token a
    # sequence of its own over a page table of its selected blocks, V from the V cache.
    attn: str = "triton"
    # With an fp8 KV cache, "aiter" runs the dots in fp8 too (else in the activation dtype).
    attn_fp8_compute: bool = True
    # The kernels' block selection: "torch" (torch.topk over the block table's width),
    # "aiter" (aiter's per-row top-k over each token's causal blocks only) or, for the
    # "aiter" attention, "fused": selection and page tables in one kernel (select_rows).
    topk: str = "torch"
    # Kernel launch shape: programs per token over the selected blocks (0: by batch size)
    # and blocks per step.
    attn_splits: int = 0
    attn_nblk: int = 2


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

    is_indexed_sparse_attention = True

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
        write_v = v.v_src == "post"
        if kc4.element_size() == 1:
            from atom.model_ops.minimax_m3.pro_kernels import quant_fp8_rows

            kr, ksc = quant_fp8_rows(kr, kc4.dtype)
            layer.k_scale[blk, 0, off] = ksc
            if write_v:
                vr, vsc = quant_fp8_rows(vr, kc4.dtype)
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
            and self.num_heads % 16 == 0
            and self.num_heads <= 32  # prep holds all of a token's heads in one program
            and (self.head_dim, self.rotary_dim, self.value_head_dim) == (256, 64, 192)
            and (self.num_idx_heads, self.index_head_dim, self.page) == (16, 128, 16)
            and v.attn_nblk in (1, 2, 4, 8)
            and v.attn in ("triton", "aiter")
            and (
                v.topk in ("torch", "aiter")
                or (
                    v.topk == "fused"
                    and v.attn == "aiter"
                    and self.topk & (self.topk - 1) == 0
                )
            )
        )

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
    ):
        """The whole layer in four launches (``prep`` / ``scores`` + top-k / ``attention``).

        ``qkv`` the raw projections of ``n`` rows (packed, or the ``(q, k, index q, index k)``
        tuple ``prep`` takes), ``positions`` / ``seq_id`` / ``slots`` ``[n]``,
        ``layer`` the layer's caches, ``nb`` the most blocks any sequence of the batch
        holds, ``out`` ``[n, H * 192]``. ``static``: ``nb`` is only a bound (the block
        table's width), so the selection runs even where every block would fit.
        """
        kc4, vc4 = layer.k_cache[:, 0], layer.v_cache[:, 0]
        fp8 = kc4.element_size() == 1
        ks, vs = (layer.k_scale, layer.v_scale) if fp8 else (None, None)
        n = positions.shape[0]
        rope = self.rotary_emb
        cos = rope.cos_cache.view(rope.cos_cache.shape[0], -1)
        sin = rope.sin_cache.view(rope.sin_cache.shape[0], -1)
        aiter_m = float(getattr(rope, "mscale", 1.0))
        if v.mscale == "aiter":
            ratio = 1.0
        else:
            ratio = (1.0 if v.mscale == "hf" else float(v.mscale)) / aiter_m
        eps = self.q_norm.variance_epsilon
        q, iq = pro_kernels.prep(
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
        scale = self.scale * v.scale_mult
        nblk = v.attn_nblk
        splits = v.attn_splits or (
            8 if n <= 16 else 4 if n <= 64 else 2 if n <= 256 else 1
        )

        def attend(sel, rows=None):
            if v.attn == "aiter":
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
        head_mode = {
            "sum": pro_kernels.HEAD_SUM,
            "mean": pro_kernels.HEAD_SUM,
            "max": pro_kernels.HEAD_MAX,
            "lse": pro_kernels.HEAD_LSE,
        }[v.head_reduce]
        sc = v.index_scale * (
            self.index_head_dim**-0.5 if v.block_score == "lse" else 1.0
        )
        score = pro_kernels.scores(
            iq,
            ic,
            block_tables,
            seq_id,
            positions,
            nb,
            self.index_compress_query.data if pool else None,
            init,
            local,
            sc,
            block_mode=block_mode,
            head_mode=head_mode,
            pool_temp=v.pool_scale,
        )
        k = min(self.topk, nb)
        if v.topk == "fused" and k & (k - 1) == 0:
            return attend(
                None, pro_kernels.select_rows(score, block_tables, seq_id, positions, k)
            )
        if v.topk == "aiter":
            sel = pro_kernels.select_blocks(score, positions, k)
        else:
            sel = score.topk(k, dim=-1, sorted=False).indices.to(torch.int32)
        return attend(sel)

    # ---------------------------------------------------------------- forward --
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
        if self._use_kernels(v):
            layer = fwd.kv_cache_data[f"layer_{self.layer_num}"]
            if self.layer_num == 0 and getattr(self, "_logged_kernels", None) != v.attn:
                self._logged_kernels = v.attn
                logger.info(
                    "M3-Pro attention: kernels, %s attention, %s KV cache (%s)",
                    v.attn,
                    layer.k_cache.dtype,
                    v,
                )
            # the kernels write every row they compute in full: zero only the others
            out = query.new_empty(n_tok, h * self.value_head_dim)
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
                    return out.zero_()
                nb = (max(ctx) + self.page - 1) // self.page
            if n < n_tok:
                out[n:].zero_()
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
            )
            return out

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
