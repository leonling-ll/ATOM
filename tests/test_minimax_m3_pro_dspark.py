# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's inline DSpark draft: config routing, the weights it reads out
of the target checkpoint, its page-16 block tables and its RoPE magnitude."""

from __future__ import annotations

import functools
import math
import os
import random
import types

import pytest
import torch

pytest.importorskip("aiter", reason="the draft imports the ATOM model stack")

from aiter import QuantType, dtypes
from transformers import PretrainedConfig

from atom.config import SpeculativeConfig
from atom.model_loader.weight_names import CheckpointNameRewriter
from atom.model_ops.minimax_m3.sparse_attn import ASM_PAGE_SIZE
from atom.models.dspark_draft import AllocationCap, draft_quant_config
from atom.models.minimax_m3 import pro_dspark


def _pro_text_config(**overrides):
    fields = {
        "architectures": ["MiniMaxM3TinyForCausalLM"],
        "num_hidden_layers": 68,
        "dspark_block_size": 7,
        "dspark_markov_rank": 256,
        "dspark_noise_token_id": 200058,
        "dspark_target_layer_ids": [4, 19, 35, 51, 67],
        "dflash_config": {
            "num_hidden_layers": 5,
            "block_size": 7,
            "mask_token_id": 200058,
            "target_layer_ids": [4, 19, 35, 51, 67],
        },
        "sparse_attention_config": {
            "use_sparse_attention": True,
            "sparse_enable_kv_share": True,
            "sparse_block_size": 16,
        },
        "rope_scaling": {
            "rope_type": "yarn",
            "factor": 48.0,
            "original_max_position_embeddings": 24576,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "attention_factor": 1.0,
        },
    }
    fields.update(overrides)
    cfg = PretrainedConfig()
    for name, value in fields.items():
        setattr(cfg, name, value)
    return cfg


def test_pro_text_config_routes_to_the_inline_pro_draft():
    cfg = _pro_text_config()
    SpeculativeConfig.hf_config_override(cfg)
    assert cfg.architectures == ["DSparkMiniMaxProDraftModel"]
    assert cfg.num_hidden_layers == 5
    assert cfg.dspark_with_draft is True
    assert cfg.dspark_target_layer_ids == [4, 19, 35, 51, 67]
    assert cfg.dspark_noise_token_id == 200058
    assert cfg.dspark_markov_rank == 256
    assert cfg.sparse_attention_config is None
    # Not the DeepSeek-V4 inline draft `dspark_block_size` alone would pick.
    assert getattr(cfg, "model_type", "") != "deepseek_v4_dspark"


def test_routing_the_normalized_config_again_changes_nothing():
    cfg = _pro_text_config()
    SpeculativeConfig.hf_config_override(cfg)
    before = dict(vars(cfg))
    SpeculativeConfig.hf_config_override(cfg)
    assert dict(vars(cfg)) == before


def test_a_pro_config_without_the_draft_depth_is_refused():
    cfg = _pro_text_config(dflash_config={})
    with pytest.raises(ValueError, match="num_hidden_layers"):
        SpeculativeConfig.hf_config_override(cfg)


def test_the_draft_quantization_leaves_out_the_targets_nvfp4_experts():
    target_quant = {
        "quant_method": "mxfp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [1, 32],
        "group_size": 16,
        "moe_quant_algo": "NVFP4",
        "moe_quant_format": "nvfp4-pack-quantized",
    }
    cfg = _pro_text_config(quantization_config=target_quant)
    SpeculativeConfig.hf_config_override(cfg)
    qc = draft_quant_config(
        cfg, draft="MiniMax-M3 DSpark", accepted={(QuantType.per_1x32, dtypes.fp8)}
    )
    assert (qc.global_spec.quant_type, qc.global_spec.quant_dtype) == (
        QuantType.per_1x32,
        dtypes.fp8,
    )
    assert not qc.layer_pattern_specs
    assert target_quant["moe_quant_algo"] == "NVFP4"


def test_the_draft_reads_only_its_own_tensors_of_the_target_checkpoint():
    rewriter = CheckpointNameRewriter(
        weights_mapper=pro_dspark.MiniMaxM3ProDSpark.weights_mapper,
        weights_mapping=pro_dspark.MiniMaxM3ProDSpark.weights_mapping,
        num_hidden_layers=5,
    )
    for target in (
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.4.block_sparse_moe.experts.0.w1.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.lm_head.weight",
        "vision_tower.blocks.0.attn.qkv.weight",
    ):
        assert rewriter.rewrite(target) is None, target
    prefix = "language_model.model.dspark."
    assert (
        rewriter.rewrite(
            prefix + "layers.3.decoder_layer.mlp.down_proj.weight_scale_inv"
        )
        == "layers.3.mlp.down_proj.weight_scale"
    )
    assert rewriter.rewrite(prefix + "fc.weight") == "fc.weight"
    assert (
        rewriter.rewrite(prefix + "confidence_head.proj.bias")
        == "confidence_head.proj.bias"
    )


# ---- the paged block metadata: `build_paged_block` and its torch reference --

needs_triton = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="dspark_block_meta is a Triton kernel: a GPU, or TRITON_INTERPRET=1",
)


def _page16_table(
    block_tables: torch.Tensor,  # [rb, W] ids of pages of n page-16s each
    out: torch.Tensor,  # [rb, n * W]
    sub_pages: torch.Tensor,  # [n] = arange(n)
) -> torch.Tensor:
    """Page-16 ids of a table of ``n``-page-16 pages: page p covers
    ``p * n + 0..n-1`` (n = 8 for M3.1's page-128 pool, 1 for M3-Pro's page-16).

    Through ``copy_`` of an expanded view: an ``out=`` of the ``[rb, W, 1]``
    product would be resized to that shape rather than broadcast over the
    sub-pages, leaving ``[8p0, 8p1+1, 8p2+2, ...]``.
    """
    rb, W = block_tables.shape
    n = sub_pages.shape[0]
    out3 = out.view(rb, W, n)
    out3.copy_(block_tables.unsqueeze(-1).expand(rb, W, n))
    out3.mul_(n).add_(sub_pages)
    return out


def _window_rows(
    bt16_flat: torch.Tensor,  # [(rb + 1) * P] request tables, then P of slack
    ctx_eff: torch.Tensor,  # [rb] keys each request can read
    positions: torch.Tensor,  # [rb * T] each row's own position
    row_req: torch.Tensor,  # [rb * T] the request each row belongs to
    window: int,  # 0: attend the whole context
    real_rows: int,  # rows past this are padding
    start16: torch.Tensor,  # [rb * T] int64 scratch
    out_tables: torch.Tensor,  # [rb * T, P]
    out_lens: torch.Tensor,  # [rb * T]
) -> None:
    """Each block row's page-16 table and length under the sliding window.

    A row at position p attends the keys from ``p - window + 1`` on, as the
    reference does (the whole block is inside that window). The per-row
    decode starts a row at a page boundary, so the row starts at the page-16
    holding that key and also reads up to 15 keys before it. Its table is
    the P-wide window of its request's table from that page on, taken from
    the flat buffer; its length ends it within its own request's columns,
    so what that window runs into past them is never read.
    """
    P = out_tables.shape[1]
    if window:
        torch.sub(positions, window - 1, out=start16)
        start16.clamp_(min=0).div_(ASM_PAGE_SIZE, rounding_mode="floor")
        # Pad rows read one page.
        start16[real_rows:].zero_()
    else:
        start16.zero_()
    torch.index_select(ctx_eff, 0, row_req, out=out_lens)
    out_lens.sub_(start16, alpha=ASM_PAGE_SIZE)
    start16.add_(row_req, alpha=P)  # each row's first entry in the flat buffer
    torch.index_select(bt16_flat.unfold(0, P, 1), 0, start16, out=out_tables)


class _TorchChainDraft:
    """`MiniMaxM3DSpark.build_paged_block` spelled in torch ops, the way the
    draft built its block before `dspark_block_meta`: the proposer's non-DCP
    torch chain writes the positions, slots and lengths (this draft does not
    set ``builds_paged_block``), then ``prepare_paged_block`` caps them at the
    allocation and cuts each row's page-16 table to its window. The reference
    the one launch is checked against."""

    block_attention_is_mla = False
    builds_paged_block = False

    def __init__(self, max_num_seqs, sliding_window):
        self.max_num_seqs, self.sliding_window = max_num_seqs, sliding_window
        self._bt16 = None

    def _alloc_block_rows(self, P: int, T: int, pages: int, device) -> None:
        max_bs = self.max_num_seqs
        self._alloc_cap = AllocationCap(max_bs, device)
        # One row of slack past the last request: `_window_rows` reads every
        # row as a P-wide window of this buffer.
        self._bt16_flat = torch.zeros(
            (max_bs + 1) * P, dtype=torch.int32, device=device
        )
        self._bt16 = self._bt16_flat[: max_bs * P].view(max_bs, P)
        self._bt16_rows = torch.zeros(max_bs * T, P, dtype=torch.int32, device=device)
        self._ctx_rows = torch.zeros(max_bs * T, dtype=torch.int32, device=device)
        self._row_req = torch.arange(
            max_bs, dtype=torch.int64, device=device
        ).repeat_interleave(T)
        self._sub_pages = torch.arange(pages, dtype=torch.int32, device=device)
        self._start16 = torch.zeros(max_bs * T, dtype=torch.int64, device=device)
        self._ctx_eff = torch.zeros(max_bs, dtype=torch.int32, device=device)

    def prepare_paged_block(
        self,
        attn_metadata,
        block_tables,
        block_positions,
        slots,
        ctx_lens,
        allocated_pages,
        scheduled_bs,
        entry_tokens,
    ) -> None:
        running_bs, T = block_positions.shape
        pages = entry_tokens // ASM_PAGE_SIZE  # page-16s per table entry
        P = pages * block_tables.shape[1]
        if self._bt16 is None or self._bt16.shape[1] != P:
            self._alloc_block_rows(P, T, pages, block_tables.device)

        alloc = self._alloc_cap.apply(
            allocated_pages, scheduled_bs, block_positions, slots, entry_tokens
        )
        # The rows' own lengths; `ctx_lens` is left as the proposer built it.
        ctx_eff = self._ctx_eff[:running_bs]
        torch.minimum(ctx_lens, alloc, out=ctx_eff)

        bt16 = _page16_table(block_tables, self._bt16[:running_bs], self._sub_pages)
        if running_bs > scheduled_bs:
            # A pad row reads one page: page 0.
            bt16[scheduled_bs:, :pages].copy_(self._sub_pages)

        rows = running_bs * T
        _window_rows(
            self._bt16_flat,
            ctx_eff,
            block_positions.view(-1),
            self._row_req[:rows],
            self.sliding_window,
            scheduled_bs * T,
            self._start16[:rows],
            self._bt16_rows[:rows],
            self._ctx_rows[:rows],
        )
        attn_metadata.dspark_bt16_rows = self._bt16_rows[:rows]
        attn_metadata.dspark_ctx_rows = self._ctx_rows[:rows]


@pytest.mark.parametrize("n", [8, 1])
def test_the_reference_page16_table_expands_every_page_into_its_sub_pages(n):
    """The reference's page-16 table of a page-128 (M3.1) or page-16 (Pro)
    block table: page p -> p * n .. p * n + n - 1, in column order."""
    bt = torch.tensor([[0, 1, 0], [5, 7, 2]], dtype=torch.int32)
    out = torch.full((2, 3 * n), -1, dtype=torch.int32)
    _page16_table(bt, out, torch.arange(n, dtype=torch.int32))
    expect = (bt.repeat_interleave(n, dim=1) * n) + torch.arange(n).repeat(3)
    assert torch.equal(out, expect.to(torch.int32))


@pytest.mark.parametrize("window", [0, 35])
def test_the_reference_starts_each_row_at_its_own_window(window):
    """Each block row of the reference starts at the page-16 holding the
    first key of ITS window (position - window + 1), not one start shared by
    the block, and not moved by the allocation cap on the request's length.
    Pad rows repeat the last real anchor and read from their first page."""
    P, T, rb = 32, 3, 2
    bt16_flat = torch.arange((rb + 1) * P, dtype=torch.int32)
    # Request 0 is capped at 98 of its 100 keys; request 1 is padding.
    ctx_eff = torch.tensor([98, 8], dtype=torch.int32)
    positions = torch.tensor([97, 98, 99, 99, 99, 99])
    row_req = torch.arange(rb).repeat_interleave(T)
    tables = torch.empty(rb * T, P, dtype=torch.int32)
    lens = torch.empty(rb * T, dtype=torch.int32)
    _window_rows(
        bt16_flat,
        ctx_eff,
        positions,
        row_req,
        window,
        T,
        torch.empty(rb * T, dtype=torch.int64),
        tables,
        lens,
    )
    # Window starts 63, 64, 65 -> page-16s 3, 4, 4.
    start16 = [3, 4, 4] if window else [0, 0, 0]
    expect_lens = [98 - 16 * s for s in start16] + [8, 8, 8]
    assert lens.tolist() == expect_lens
    for row, s in enumerate(start16 + [0, 0, 0]):
        first = row_req[row].item() * P + s
        assert tables[row].tolist() == list(range(first, first + P))


@pytest.fixture
def host_staging(monkeypatch):
    """`AllocationCap`'s pinned staging and CUDA event, without a GPU."""
    if torch.cuda.is_available():
        return
    zeros = torch.zeros

    def unpinned(*args, pin_memory=False, **kwargs):
        return zeros(*args, **kwargs)

    class _Event:
        def synchronize(self):
            pass

        def record(self):
            pass

    monkeypatch.setattr(torch, "zeros", unpinned)
    monkeypatch.setattr(torch.cuda, "Event", _Event)


def _pro_draft(max_num_seqs, sliding_window):
    Draft = pro_dspark.MiniMaxM3ProDSpark
    draft = Draft.__new__(Draft)
    draft.max_num_seqs, draft.sliding_window = max_num_seqs, sliding_window
    draft._block_rows_p = None
    return draft


@needs_triton
def test_block_rows_read_the_page16_table_itself(host_staging):
    """A table entry is one page-16: a block row's table is its request's own
    from the page holding its window's first key, then zeros; its length what
    is left of the request's keys from there."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    i64 = {"dtype": torch.int64, "device": device}
    draft = _pro_draft(2, 32)
    pos = torch.full((2, 2), -5, **i64)
    slots = torch.full((4,), -5, **i64)
    ctx = torch.full((2,), -5, dtype=torch.int32, device=device)
    block_tables = torch.tensor(
        [[9, 2, 6, 8], [5, 0, 0, 0]], dtype=torch.int32, device=device
    )
    md = types.SimpleNamespace()
    anchors = torch.tensor([49, 2], **i64)
    draft.build_paged_block(
        md, block_tables, anchors, pos, slots, ctx, [4, 1], 2, 16, 3
    )
    # Rows at 50 and 51 attend keys 19.. and 20.., both on the table's second
    # page; their length (52 - 16) ends on the table's last page.
    assert md.dspark_bt16_rows.tolist() == [
        [2, 6, 8, 0],
        [2, 6, 8, 0],
        [5, 0, 0, 0],
        [5, 0, 0, 0],
    ]
    assert md.dspark_ctx_rows.tolist() == [36, 36, 5, 5]
    assert pos.tolist() == [[50, 51], [3, 4]]
    assert slots.tolist() == [130, 131, 83, 84]
    assert ctx.tolist() == [52, 5]


def _block_meta_inputs(rng, running_bs, scheduled_bs, T, E, W, device):
    """A drafting round's staged anchors, block tables and allocations: some
    anchors at the table's end, some at its start, allocations a page short
    of or past what the block needs, pad rows anchored at 0."""
    max_tokens = W * E
    anchors = torch.zeros(running_bs, dtype=torch.int64)
    allocated = []
    for i in range(scheduled_bs):
        kind = rng.random()
        if kind < 0.2:
            anchor = rng.randrange(max(0, max_tokens - 2 * T), max_tokens)
        elif kind < 0.3:
            anchor = rng.randrange(0, 3)
        else:
            anchor = rng.randrange(0, max_tokens - T - 1)
        anchors[i] = anchor
        need = (anchor + 1) // E + 1
        allocated.append(min(W, max(1, need + rng.choice([-1, 0, 0, 1, 2]))))
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    tables = torch.randint(
        0, 50000, (running_bs + 2, W), dtype=torch.int32, generator=g
    )
    return anchors.to(device), tables.to(device), allocated


def _build_block_meta(draft, anchors, tables, allocated, scheduled_bs, T, E):
    """`DSparkProposer._build_paged_block_metadata` for ``draft``, on a stub
    proposer: the attention metadata it publishes."""
    from atom.spec_decode.dspark_proposer import DSparkProposer

    running_bs, device = anchors.shape[0], anchors.device
    max_bs = tables.shape[0]
    np_rows = object()
    buffer = types.SimpleNamespace(
        gpu=tables,
        np=np_rows,
        _block_table=types.SimpleNamespace(cpu=np_rows, lengths=allocated),
    )
    i64 = {"dtype": torch.int64, "device": device}
    i32 = {"dtype": torch.int32, "device": device}
    stub = types.SimpleNamespace(
        draft_tokens_per_seq=T,
        runner=types.SimpleNamespace(
            block_size=E, forward_vars={"block_tables": buffer}
        ),
        model=draft,
        _blk_positions=torch.full((max_bs, T), -5, **i64),
        _blk_offsets=torch.arange(1, T + 1, **i64),
        _blk_slots=torch.full((max_bs * T,), -5, **i64),
        _blk_ctx_lens=torch.full((max_bs,), -5, **i32),
        _blk_cu_seqlens_q=torch.arange(0, (max_bs + 1) * T, step=T, **i32),
        _blk_dtype_q=torch.bfloat16,
        _bound_draft_k_cache=lambda fc: object(),
        _draft_is_dcp_sharded=False,
        _draft_entry_tokens=E,
        _paged_pad_ctx_len=min(1 + T, E),
        _block_page_idx=DSparkProposer._block_page_idx,
    )
    stub._publish_paged_block = lambda *a: DSparkProposer._publish_paged_block(stub, *a)
    md = types.SimpleNamespace()
    DSparkProposer._build_paged_block_metadata(
        stub,
        None,
        md,
        anchors,
        scheduled_bs=scheduled_bs,
        max_seqlen_k=E * tables.shape[1],
    )
    md.blk_positions = stub._blk_positions[:running_bs]
    return md


def _check_block_meta(rounds, running_bs, scheduled_bs, T, E, W, window):
    """``rounds`` random drafting rounds: the one-launch build
    (`dspark_block_meta`) publishes the torch reference's slots, context
    lengths and block positions, and every block row's table and length, bit
    for bit, on every entry the row's decode reads."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(f"{running_bs}/{scheduled_bs}/{T}/{E}/{W}/{window}")
    for _ in range(rounds):
        anchors, tables, allocated = _block_meta_inputs(
            rng, running_bs, scheduled_bs, T, E, W, device
        )
        max_bs = tables.shape[0]
        ref_draft = _TorchChainDraft(max_bs, window)
        ref = _build_block_meta(
            ref_draft, anchors, tables, allocated, scheduled_bs, T, E
        )
        got = _build_block_meta(
            _pro_draft(max_bs, window), anchors, tables, allocated, scheduled_bs, T, E
        )
        for name in ("slot_mapping", "context_lens", "blk_positions", "block_tables"):
            assert torch.equal(getattr(got, name).view(-1), getattr(ref, name).view(-1))
        assert torch.equal(got.cu_seqlens_q, ref.cu_seqlens_q)
        assert (got.max_seqlen_q, got.max_seqlen_k) == (
            ref.max_seqlen_q,
            ref.max_seqlen_k,
        )
        assert torch.equal(got.dspark_ctx_rows, ref.dspark_ctx_rows)
        rows = running_bs * T
        P = (E // 16) * W
        # `_window_rows` left each row's window start + its request * P here.
        req = torch.arange(running_bs, device=device).repeat_interleave(T)
        own = P - (ref_draft._start16[:rows] - req * P)
        n_read = (ref.dspark_ctx_rows.long() + 15) // 16
        assert bool((n_read <= own).all()), "a row would read past its own table"
        for r in range(rows):
            upto = int(own[r])
            assert torch.equal(
                got.dspark_bt16_rows[r, :upto], ref.dspark_bt16_rows[r, :upto]
            ), r


_BLOCK_BATCHES = [(1, 1), (4, 4), (5, 3), (8, 1)]


@needs_triton
@pytest.mark.parametrize("window", [0, 40, 2048])
# Pro's page-16 pool and M3.1's page-128 one; the last two have P > 1024
# page-16s a row, so their tables span several column-chunk programs.
@pytest.mark.parametrize("E,W", [(16, 96), (128, 12), (16, 2500), (128, 320)])
@pytest.mark.parametrize("T", [3, 7])
@pytest.mark.parametrize("running_bs,scheduled_bs", _BLOCK_BATCHES)
def test_build_paged_block_is_the_torch_chain(
    host_staging, running_bs, scheduled_bs, T, E, W, window
):
    """The one-launch build is the torch chain (`_check_block_meta`)."""
    _check_block_meta(2, running_bs, scheduled_bs, T, E, W, window)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("window", [0, 40, 2048])
# 8192 page-16s a row (max-model-len 131072) and 65536 (about 1M), in Pro's
# page-16 pool and M3.1's page-128 one.
@pytest.mark.parametrize("E,W", [(16, 8192), (128, 1024), (16, 65536), (128, 8192)])
@pytest.mark.parametrize("running_bs,scheduled_bs", _BLOCK_BATCHES)
def test_build_paged_block_is_the_torch_chain_at_serving_widths(
    running_bs, scheduled_bs, E, W, window
):
    """The same at the table widths a server runs, at k=7."""
    _check_block_meta(6, running_bs, scheduled_bs, 7, E, W, window)


def test_yarn_magnitude_is_the_configs_attention_factor():
    """aiter scales YaRN's cos/sin by ``0.1 ln(factor) + 1`` times its
    ``attn_factor``; the draft asks for the config's ``attention_factor``."""
    from aiter.rotary_embedding import _yarn_get_mscale

    cfg = _pro_text_config()
    scaling = pro_dspark.M3ProDSparkAttention._rope_scaling(None, cfg)
    assert "attention_factor" not in scaling
    assert scaling["rope_type"] == "yarn" and scaling["factor"] == 48.0
    assert math.isclose(_yarn_get_mscale(48.0) * scaling["attn_factor"], 1.0)
    # The config itself is left as it was.
    assert cfg.rope_scaling["attention_factor"] == 1.0


def test_unscaled_rope_stays_unscaled():
    cfg = _pro_text_config(rope_scaling=None)
    assert pro_dspark.M3ProDSparkAttention._rope_scaling(None, cfg) is None


def _paged_proposer(model, num_spec):
    """A `DSparkProposer` of the paged (separate-draft) flavor, with nothing
    built but what the block width reads: the draft model and the k resolved
    from a Pro draft config."""
    from atom.spec_decode.dspark_proposer import DSparkProposer

    cfg = _pro_text_config()
    SpeculativeConfig.hf_config_override(cfg)
    p = DSparkProposer.__new__(DSparkProposer)
    p.speculative_config = types.SimpleNamespace(
        draft_model_hf_config=cfg,
        num_speculative_tokens=num_spec,
        use_dspark_with_draft=lambda: True,
    )
    p.mtp_k = p._resolve_mtp_k()
    p.model = model
    return p


@pytest.mark.parametrize("k", range(1, 9))
def test_the_pro_draft_drafts_its_trained_block_below_it(k):
    p = _paged_proposer(pro_dspark.MiniMaxM3ProDSpark, k)
    assert p.mtp_k == k
    # The block is bidirectional, so a narrower one changes every drafted
    # token: below 7 the draft still runs all 7, above it the verify's k.
    assert p.draft_tokens_per_seq == max(k, 7)


@pytest.mark.parametrize("k", [3, 7])
def test_a_paged_draft_without_a_trained_block_drafts_k(k):
    """Kimi-K3's draft is width-agnostic: its block is whatever k says."""
    p = _paged_proposer(types.SimpleNamespace(), k)
    assert p.draft_tokens_per_seq == k


@pytest.mark.parametrize("k", [1, 4, 7])
def test_the_verify_gets_k_drafts_of_the_wider_block(k):
    """The block pass returns its whole width; the verify, and the verify
    scheduler's confidence, get only the first k columns."""
    model = types.SimpleNamespace(
        drafts_trained_block=True, block_attention_is_mla=False
    )
    p = _paged_proposer(model, k)
    scheduled_bs, running_bs, width = 3, 4, p.draft_tokens_per_seq
    ids = torch.arange(running_bs * width, dtype=torch.int32).view(running_bs, -1)
    conf = torch.rand(running_bs, width)
    seen = {}
    # The block buffers, sized as the server sizes them, and the block's
    # positions built from them; the pool is not bound, so nothing past that.
    p.config = types.SimpleNamespace(max_num_seqs=8)
    p.device = torch.device("cpu")
    p._init_draft_block_buffers()
    p._blk_dtype_q = torch.bfloat16
    p._bound_draft_k_cache = lambda fc: None
    p._publish_draft_shape = lambda fc, **kw: seen.update(kw)
    p.runner = types.SimpleNamespace(
        block_size=16,
        forward_vars={"batch_info": types.SimpleNamespace(gpu=torch.zeros(3))},
    )
    p._verify_scheduler = types.SimpleNamespace(
        compute_ell=lambda c: seen.setdefault("ell_shape", tuple(c.shape)),
        set_last_ell=lambda ell: None,
    )
    pass_ = types.SimpleNamespace(
        label=lambda s, r: "eager", run=lambda bs, **kw: (ids[:bs], conf[:bs])
    )
    fc = types.SimpleNamespace(
        context=types.SimpleNamespace(running_bs=running_bs, is_prefill=True)
    )
    md = types.SimpleNamespace(max_seqlen_k=32)
    staged = {"anchor_positions": torch.arange(running_bs, dtype=torch.int64) * 10}

    out = p._propose_with_draft(pass_, fc, md, staged, scheduled_bs)

    assert out.tolist() == ids[:scheduled_bs, :k].tolist()
    anchors = staged["anchor_positions"].view(-1, 1)
    assert (
        p._blk_positions[:running_bs].tolist()
        == (anchors + torch.arange(1, width + 1)).tolist()
    )
    assert seen["ell_shape"] == (scheduled_bs, k)
    # The pass itself is the whole block.
    assert seen["running_tokens"] == running_bs * width


# ---- the draft layers run the target's epilogues ----------------------------

# The stubs below model two TP ranks, so a missing or an extra all-reduce is
# off by 2x: a projection that skips its all-reduce returns this rank's partial
# sum, half the full one, and each all-reduce sums the two ranks' halves.
_TP = 2


class _Norm(torch.nn.Module):
    """A Gemma RMSNorm's two forms: ``norm(x)`` and the residual-add
    ``norm(x, residual) -> (normed, x + residual)``."""

    def __init__(self, n, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.weight = torch.nn.Parameter(torch.randn(n, generator=g) * 0.1)

    def ref(self, x):
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        return x * rms * (1.0 + self.weight)

    def forward(self, x, residual=None):
        if residual is None:
            return self.ref(x)
        residual = x + residual
        return self.ref(residual), residual


class _Linear(torch.nn.Module):
    """A native MXFP8 linear as the layer sees it on one of ``_TP`` ranks: it
    takes ``(x, x_scale)``; it returns the reduced product when
    ``reduce_results`` says so, else this rank's partial sum of it."""

    def __init__(self, n, seed, native=True):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(n, n, generator=g) / n**0.5
        self.native_a8_group_rows = 32 if native else None
        self.reduce_results = True

    def forward(self, x, x_scale=None):
        assert x_scale is None or x_scale == "mx", x_scale
        out = x @ self.w
        return out if self.reduce_results else out / _TP


class _Attn(torch.nn.Module):
    def __init__(self, n, seed, native=True):
        super().__init__()
        self.qkv_proj = _Linear(n, seed, native)
        self.o_proj = _Linear(n, seed + 1)

    def forward(self, positions, x):
        x, x_scale = x if isinstance(x, tuple) else (x, None)
        return self.o_proj(torch.tanh(self.qkv_proj(x, x_scale)))


class _Mlp(torch.nn.Module):
    def __init__(self, n, seed, native=True):
        super().__init__()
        self.gate_up_proj = _Linear(n, seed, native)
        self.down_proj = _Linear(n, seed + 1)

    def forward(self, x, x_scale=None, quant_down=False):
        return self.down_proj(torch.relu(self.gate_up_proj(x, x_scale)))


def _mx_epilogues(monkeypatch):
    """The fused norms the layer calls, as the math they compute on one of
    ``_TP`` ranks holding equal partial sums: the all-reduce, the residual add
    and the norm, with the MXFP8 pair standing for the normed rows (its bytes
    are tests/test_m3pro_mxfp8_fusion.py's). The target's default variants."""
    from atom.model_ops.minimax_m3 import pro_attn

    def norm_mx(x, residual, norm, write_bf16=True):
        assert residual is None and not write_bf16
        return None, norm.ref(x), "mx", None

    def ar_norm_mx(x, residual, norm, lamport_rows=0):
        residual = x * _TP + residual
        out = norm.ref(residual)
        return out, out, "mx", residual

    def ar_norm(x, residual, norm):
        return norm(x * _TP, residual)

    monkeypatch.delenv("ATOM_M3PRO_VARIANTS", raising=False)
    monkeypatch.setattr(pro_attn, "_variants", pro_attn.ProVariants())
    monkeypatch.setattr(pro_dspark, "gemma_rms_norm_mxfp8", norm_mx)
    monkeypatch.setattr(pro_dspark, "fused_allreduce_gemma_rms_norm_mxfp8", ar_norm_mx)
    monkeypatch.setattr(pro_dspark, "fused_allreduce_gemma_rms_norm", ar_norm)
    monkeypatch.setattr(pro_dspark, "_aiter_mx_norm", lambda: True)


def _stub_layer_init(native=True):
    def init(self, atom_config, config, quant_config, layer_num, prefix=""):
        torch.nn.Module.__init__(self)
        n = config.hidden_size
        self.self_attn = _Attn(n, 10 * layer_num, native)
        self.mlp = _Mlp(n, 10 * layer_num + 5)
        self.input_layernorm = _Norm(n, 10 * layer_num + 7)
        self.post_attention_layernorm = _Norm(n, 10 * layer_num + 8)

    return init


def _draft(monkeypatch, layers=3, n=64, native=True):
    from atom.models.minimax_m3 import dspark as m3_dspark

    monkeypatch.setattr(
        m3_dspark.M3DSparkDecoderLayer, "__init__", _stub_layer_init(native)
    )
    cfg = types.SimpleNamespace(hidden_size=n)
    Draft = pro_dspark.MiniMaxM3ProDSpark
    draft = Draft.__new__(Draft)
    torch.nn.Module.__init__(draft)
    draft.layers = torch.nn.ModuleList(
        [Draft.layer_cls(None, cfg, None, i) for i in range(layers)]
    )
    draft.final_norm = _Norm(n, 99)
    draft.embed_tokens = torch.nn.Embedding(32, n)
    with torch.no_grad():
        draft.embed_tokens.weight.copy_(
            torch.randn(32, n, generator=torch.Generator().manual_seed(1))
        )
    draft.noise_token_id = 31
    return draft


def test_fused_draft_layers_compute_the_unfused_block(monkeypatch):
    """Where the target fuses its norms, each draft layer hands its UN-reduced
    output and the residual on, the next layer's fused input norm (or the
    final norm) all-reduces and adds it, and the block comes out as the
    unfused layers'. A dropped or an extra all-reduce is off by 2x."""
    from atom.model_ops.minimax_m3 import pro_attn

    _mx_epilogues(monkeypatch)
    ids = torch.tensor([3, 7])
    positions = torch.arange(8)
    monkeypatch.setattr(
        pro_attn, "_variants", pro_attn.ProVariants(fuse_mxfp8_quant=False)
    )
    ref_draft = _draft(monkeypatch)
    assert not any(layer.fused_epilogues for layer in ref_draft.layers)
    ref = ref_draft.block_backbone(ids, positions, 4)
    monkeypatch.setattr(pro_attn, "_variants", pro_attn.ProVariants())
    draft = _draft(monkeypatch)
    for layer in draft.layers:
        assert layer.fused_epilogues
        assert not layer.self_attn.o_proj.reduce_results
        assert not layer.mlp.down_proj.reduce_results
    got = draft.block_backbone(ids, positions, 4)
    torch.testing.assert_close(got, ref, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("off", ["variant", "diag", "linear", "aiter"])
def test_draft_layers_keep_the_unfused_epilogues_where_the_target_does(
    monkeypatch, off
):
    """The fused epilogues need the target's ``fuse_mxfp8_quant`` (and not its
    ``diag`` variant), native MXFP8 linears and an aiter with the MXFP8 norms;
    without any one the layer stays unfused and its projections keep their
    all-reduce."""
    from atom.model_ops.minimax_m3 import pro_attn

    _mx_epilogues(monkeypatch)
    if off == "variant":
        monkeypatch.setattr(
            pro_attn, "_variants", pro_attn.ProVariants(fuse_mxfp8_quant=False)
        )
    if off == "diag":
        monkeypatch.setattr(pro_attn, "_variants", pro_attn.ProVariants(diag=True))
    if off == "aiter":
        monkeypatch.setattr(pro_dspark, "_aiter_mx_norm", lambda: False)
    draft = _draft(monkeypatch, native=off != "linear")
    for layer in draft.layers:
        assert not layer.fused_epilogues
        assert layer.self_attn.o_proj.reduce_results
        assert layer.mlp.down_proj.reduce_results


@functools.cache
def _mx_weight(n_in, n_out, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(n_in, n_out, generator=g) / n_in**0.5
    return w.to(torch.bfloat16).cuda().float()


class _MxGemm(torch.nn.Module):
    """A native MXFP8 linear on one GPU, as its GEMM sees it: it records the
    MXFP8 input it is handed, or ``quantize_fp8`` of a bf16 one (what the
    GEMM does with it), and returns that input, dequantized, times a fixed
    bf16 weight. TP 1, so ``reduce_results`` changes nothing."""

    native_a8_group_rows = 32

    def __init__(self, n_in, n_out, seed, name, log):
        super().__init__()
        self.w = _mx_weight(n_in, n_out, seed)
        self.name, self.log, self.reduce_results = name, log, True

    def forward(self, x, x_scale=None):
        from atom.model_ops.blockscale import quantize_fp8

        if x_scale is None:
            x, x_scale = quantize_fp8(x)
        x_scale = x_scale.contiguous().view(torch.uint8)
        self.log.append((self.name, x.contiguous().view(torch.uint8), x_scale))
        m, n = x.shape
        scale = torch.exp2(x_scale.float() - 127).view(m, n // 32, 1)
        x = (x.float().view(m, n // 32, 32) * scale).view(m, n)
        return (x @ self.w).to(torch.bfloat16)


class _MxAttn(torch.nn.Module):
    """The attention's two GEMMs around a stand-in for the per-row decode."""

    def __init__(self, n, seed, log, qkv=256):
        super().__init__()
        self.qkv_proj = _MxGemm(n, qkv, seed, "qkv_proj", log)
        self.o_proj = _MxGemm(qkv, n, seed + 1, "o_proj", log)

    def forward(self, positions, x):
        x, x_scale = x if isinstance(x, tuple) else (x, None)
        return self.o_proj(torch.tanh(self.qkv_proj(x, x_scale)))


def _gpu_draft(monkeypatch, fused, layers, n, inter, log):
    """A Pro draft of ``layers`` layers on one GPU: the real layer forwards,
    Gemma norms, SwiGLU-OAI MLP forward and norm / quant kernels; recording
    GEMMs (`_MxGemm`) and a stand-in for the attention core."""
    from atom.model_ops import layernorm
    from atom.model_ops.layernorm import GemmaRMSNorm
    from atom.model_ops.minimax_m3 import pro_attn
    from atom.models.minimax_m3 import dspark as m3_dspark
    from atom.models.minimax_m3.model import MiniMaxM3MLP

    def norm(seed):
        g = torch.Generator().manual_seed(seed)
        m = GemmaRMSNorm(n, eps=1e-6)
        m.weight.data = (torch.randn(n, generator=g) * 0.3).to(torch.bfloat16).cuda()
        return m

    def init(self, atom_config, config, quant_config, layer_num, prefix=""):
        torch.nn.Module.__init__(self)
        s = 10 * layer_num
        self.self_attn = _MxAttn(n, s, log)
        mlp = MiniMaxM3MLP.__new__(MiniMaxM3MLP)
        torch.nn.Module.__init__(mlp)
        mlp.gate_up_proj = _MxGemm(n, 2 * inter, s + 2, "gate_up_proj", log)
        mlp.down_proj = _MxGemm(inter, n, s + 3, "down_proj", log)
        mlp.swiglu_alpha, mlp.swiglu_beta, mlp.swiglu_limit = 1.702, 1.0, 7.0
        self.mlp = mlp
        self.input_layernorm = norm(s + 7)
        self.post_attention_layernorm = norm(s + 8)

    monkeypatch.setattr(layernorm, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        pro_attn, "_variants", pro_attn.ProVariants(fuse_mxfp8_quant=fused)
    )
    monkeypatch.setattr(m3_dspark.M3DSparkDecoderLayer, "__init__", init)
    Draft = pro_dspark.MiniMaxM3ProDSpark
    draft = Draft.__new__(Draft)
    torch.nn.Module.__init__(draft)
    cfg = types.SimpleNamespace(hidden_size=n)
    draft.layers = torch.nn.ModuleList(
        [Draft.layer_cls(None, cfg, None, i) for i in range(layers)]
    )
    draft.final_norm = norm(99)
    return draft


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("inter", [2560, 20480])  # the dense FFN at TP 8, TP 1
@pytest.mark.parametrize("rows", [7, 8, 28, 32])
def test_fused_draft_layers_feed_each_gemm_the_unfused_layers_bytes(
    monkeypatch, rows, inter
):
    """At TP 1, on the GPU, at Pro's hidden size: two draft layers and the
    final norm, with the target's epilogues and without. Every GEMM gets the
    same MXFP8 bytes (E4M3 values and E8M0 scales): the fused norms and the
    fused SwiGLU write what ``quantize_fp8`` of the unfused layer's bf16
    activations gives. With the same GEMM inputs, the block's output is
    bitwise the unfused one; at TP 8 only the all-reduce's summation order
    may differ."""
    n, layers = 7168, 2
    g = torch.Generator().manual_seed(rows + inter)
    x = (torch.randn(rows, n, generator=g) * 2).to(torch.bfloat16).cuda()
    positions = torch.arange(rows, device="cuda")
    outs, logs = {}, {}
    for fused in (False, True):
        log = logs[fused] = []
        draft = _gpu_draft(monkeypatch, fused, layers, n, inter, log)
        assert all(layer.fused_epilogues == fused for layer in draft.layers)
        hidden, residual = x, None
        for layer in draft.layers:
            hidden, residual = layer(positions, hidden, residual)
        outs[fused] = draft.block_final_norm(hidden, residual)
    names = ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"] * layers
    assert [e[0] for e in logs[False]] == names
    assert [e[0] for e in logs[True]] == names
    for i, ((name, q, s), (_, fq, fs)) in enumerate(zip(logs[False], logs[True])):
        layer = i // 4
        assert torch.equal(fq, q), f"layer {layer} {name}: E4M3 bytes differ"
        assert torch.equal(fs, s), f"layer {layer} {name}: E8M0 scales differ"
    assert torch.equal(outs[True], outs[False])
