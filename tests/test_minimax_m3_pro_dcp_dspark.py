# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's DSpark draft under decode context parallelism.

Its pool sits outside the DCP group (``MiniMaxM3ProDSpark.kv_pool_spec``): an entry is
``16 * dcp`` tokens, ``dcp`` page-16 blocks end to end with their per-token fp8 scale
planes, and the target's builder stages the draft's own context slots. At DCP 1 the spec
builds the generic pool; at DCP 8 the draft's write slots, page-16 tables and lengths are
DCP 1's element for element, so it writes and reads the same keys through the same
pages; an entry moves through offload under its block id whole. Without a GPU the
block's one-launch build runs in Triton's CPU interpreter.
"""

from __future__ import annotations

import functools
import os
import random
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import gpu_stack_import
from torch import nn

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

from atom.model_ops.attentions.draft_ctx_slots import DraftCtxSlotsMixin
from atom.model_ops.attentions.mha_kv_pool import MhaKvPool
from atom.spec_decode.draft_kv import DraftKvBuilder, DraftKvPoolSpec, draft_kv_builder
from atom.utils import CpuGpuBuffer

PRO_SPEC = DraftKvPoolSpec(flash_layout=False, own_scales=True)
FP8 = torch.float8_e4m3fnuz
# the Pro draft's attention, one KV head a rank, at a CPU-sized head
DRAFT = {"layers": 2, "num_kv_heads": 1, "head_dim": 64, "block_size": 16}


def _config(dcp):
    return SimpleNamespace(kv_cache_block_size=16, decode_context_parallel_size=dcp)


class _Layer(nn.Module):
    """A draft attention as `DraftKvBuilder` binds it: paged, not MLA, page 16."""

    base_attention, use_mla, page_tokens, check_cache = True, False, 16, None

    def __init__(self):
        super().__init__()
        self.layer_num = 0


def test_the_pro_spec_names_the_shuffle_pool_and_its_scale_planes():
    pro = gpu_stack_import("atom.models.minimax_m3.pro_dspark")
    assert pro.MiniMaxM3ProDSpark.kv_pool_spec == PRO_SPEC
    with pytest.raises(ValueError, match="FLASH"):
        DraftKvPoolSpec(own_scales=True)


@pytest.mark.parametrize("dcp", [1, 8])
def test_an_entry_is_dcp_blocks_and_gets_its_own_slots_only_under_dcp(dcp):
    assert PRO_SPEC.pool_kwargs(dcp) == {"blocks_per_entry": dcp, "flash_layout": False}
    assert PRO_SPEC.entry_tokens(_config(dcp)) == 16 * dcp
    assert PRO_SPEC.needs_own_ctx_slots(_config(dcp)) is (dcp > 1)


def _alloc(pool, entries):
    pool.allocate(entries, "cpu")
    return pool


def test_at_dcp1_the_spec_builds_the_generic_pool():
    """The draft at DCP 1 is unchanged: the pool its spec builds is the one the
    generic path (no spec) builds, view for view."""
    hf = SimpleNamespace(num_key_value_heads=1, head_dim=64)
    kw = {"world_size": 8, "block_size": 16, "layers": 2, "kv_dtype": FP8}
    a = _alloc(MhaKvPool.from_hf_config(hf, **kw, **PRO_SPEC.pool_kwargs(1)), 4)
    b = _alloc(MhaKvPool.from_hf_config(hf, **kw), 4)
    assert a.entry_bytes == b.entry_bytes and a.entry_tokens == b.entry_tokens == 16
    for x, y in zip(a.kv_views(1) + a.scale_views(1), b.kv_views(1) + b.scale_views(1)):
        assert (x.shape, x.stride(), x.dtype) == (y.shape, y.stride(), y.dtype)


def test_the_spec_binds_the_pools_own_scale_planes(monkeypatch):
    """`draft_kv_builder` builds the spec's pool (SHUFFLE, `dcp` blocks an entry, at
    the scheduler's block) and binds the pool's per-token scale planes, not a unit
    scale."""
    from atom.utils import selector

    seen = {}

    class _Backend:
        @staticmethod
        def make_kv_pool(hf, *, world_size, kv_dtype, **kw):
            seen.update(kw)
            return object()

    aiter = types.ModuleType("aiter")
    aiter.dtypes = SimpleNamespace(d_dtypes={"fp8": FP8})
    monkeypatch.setitem(sys.modules, "aiter", aiter)
    family = next(iter(selector.DRAFT_POOL_FAMILIES))
    monkeypatch.setattr(selector, "attn_family", lambda _hf: family)
    monkeypatch.setattr(selector, "get_attn_backend", lambda _family: _Backend)
    runner = SimpleNamespace(
        world_size=8,
        config=SimpleNamespace(
            **vars(_config(8)),
            kv_cache_dtype="fp8",
            speculative_config=SimpleNamespace(draft_kv_dtype="fp8"),
        ),
        drafter=SimpleNamespace(model=nn.Sequential()),
        attn_metadata_builder=SimpleNamespace(block_size=16),
    )
    model = SimpleNamespace(kv_pool_spec=PRO_SPEC)
    builder = draft_kv_builder(runner, SimpleNamespace(), model)
    assert builder._kv_scale is None
    builder._make_pool(layers=1, target_block_size=16)
    assert seen == {
        "layers": 1,
        "target_block_size": 16,
        "blocks_per_entry": 8,
        "flash_layout": False,
    }

    pool = _alloc(MhaKvPool(**DRAFT, kv_dtype=FP8, **PRO_SPEC.pool_kwargs(8)), 3)
    layer = _Layer()
    runner.config.max_model_len = 4096
    bound = DraftKvBuilder(runner, lambda **_: pool, kv_scale=builder._kv_scale)
    bound._kv_pool = pool
    bound._pool_rows = {"draft_kv": {layer: 0}}
    bound.build_kv_cache_tensor(layer)
    k_scale, _ = pool.scale_views(0)
    assert layer.k_scale.data_ptr() == k_scale.data_ptr()
    assert tuple(layer.k_scale.shape) == (3 * 8, 1, 16)
    assert tuple(layer.k_cache.shape)[0] == 3 * 8


def _fill(pool, seed):
    g = torch.Generator().manual_seed(seed)
    for _, t in pool.region_tensors():
        b = t.view(torch.uint8)
        b.copy_(torch.randint(0, 256, b.shape, dtype=torch.uint8, generator=g))


def test_an_entry_moves_under_its_block_id_whole():
    """What an offload tier copies for a block id (the pool's PAGE units) is the
    entry's `dcp` page-16 blocks of every layer, K, V and scales; restored under
    another block id, the draft reads the same keys from that entry's blocks."""
    src = _alloc(MhaKvPool(**DRAFT, kv_dtype=FP8, **PRO_SPEC.pool_kwargs(8)), 5)
    dst = _alloc(MhaKvPool(**DRAFT, kv_dtype=FP8, **PRO_SPEC.pool_kwargs(8)), 5)
    _fill(src, 0)
    _fill(dst, 1)
    pages = [
        (a.view, b.view)
        for a, b in zip(
            src.transfer_pages("draft", "draft", [0, 1], [16, 16]),
            dst.transfer_pages("draft", "draft", [0, 1], [16, 16]),
        )
    ]
    for a, b in pages:
        assert a.shape[0] == 5
        b[1].copy_(a[3])
    for layer in range(2):
        for x, y in zip(
            src.kv_views(layer) + src.scale_views(layer),
            dst.kv_views(layer) + dst.scale_views(layer),
        ):
            assert torch.equal(x[24:32].view(torch.uint8), y[8:16].view(torch.uint8))


# -- the block: DCP 8's addressing is DCP 1's -------------------------------------


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


def _block(tables, allocated, anchors, scheduled_bs, T, entry_tokens, max_model_len):
    """`DSparkProposer._build_paged_block_metadata` for the Pro draft on a stub
    proposer whose target blocks at 16: the metadata it publishes, and the draft."""
    pytest.importorskip("triton")
    pro = gpu_stack_import("atom.models.minimax_m3.pro_dspark")
    from atom.spec_decode.dspark_proposer import DSparkProposer

    max_bs, device = tables.shape[0], tables.device
    draft = pro.MiniMaxM3ProDSpark.__new__(pro.MiniMaxM3ProDSpark)
    draft.max_num_seqs, draft.sliding_window, draft._block_rows_p = max_bs, 2048, None
    draft._mk_packed = torch.empty(0)  # the Markov head's, not the block's
    rows = object()
    buffer = SimpleNamespace(
        gpu=tables, np=rows, _block_table=SimpleNamespace(cpu=rows, lengths=allocated)
    )
    i64 = {"dtype": torch.int64, "device": device}
    i32 = {"dtype": torch.int32, "device": device}
    stub = SimpleNamespace(
        draft_tokens_per_seq=T,
        config=SimpleNamespace(max_model_len=max_model_len),
        runner=SimpleNamespace(block_size=16, forward_vars={"block_tables": buffer}),
        model=draft,
        _blk_positions=torch.full((max_bs, T), -5, **i64),
        _blk_offsets=torch.arange(1, T + 1, **i64),
        _blk_slots=torch.full((max_bs * T,), -5, **i64),
        _blk_ctx_lens=torch.full((max_bs,), -5, **i32),
        _blk_cu_seqlens_q=torch.arange(0, (max_bs + 1) * T, step=T, **i32),
        _blk_dtype_q=torch.bfloat16,
        _bound_draft_k_cache=lambda fc: object(),
        _draft_is_dcp_sharded=False,
        _draft_entry_tokens=entry_tokens,
        _paged_pad_ctx_len=min(1 + T, entry_tokens),
        _block_page_idx=DSparkProposer._block_page_idx,
    )
    stub._publish_paged_block = lambda *a: DSparkProposer._publish_paged_block(stub, *a)
    md = SimpleNamespace()
    DSparkProposer._build_paged_block_metadata(
        stub, None, md, anchors, scheduled_bs=scheduled_bs, max_seqlen_k=max_model_len
    )
    md.blk_positions = stub._blk_positions[: anchors.shape[0]]
    return md, draft


@pytest.mark.parametrize("running_bs,scheduled_bs", [(1, 1), (4, 4), (5, 3)])
@pytest.mark.parametrize("T", [3, 7])
def test_the_dcp8_block_writes_and_reads_what_the_dcp1_block_does(
    host_staging, running_bs, scheduled_bs, T
):
    """DCP 8 hands the block the target's table (`max_model_len / 16` columns, of which
    a request holds `/ 128`) and 128-token entries; DCP 1 the page-16 table of the same
    pages. The block's write slots, positions, lengths and every page-16 table entry
    its rows read are the same numbers, and its per-row tables are as wide (the
    proposer cuts the table to what the entries can hold)."""
    rng = random.Random(f"{running_bs}/{scheduled_bs}/{T}")
    w8 = 6
    max_model_len = 128 * w8
    max_bs = running_bs + 2
    g = torch.Generator().manual_seed(rng.randrange(1 << 30))
    t128 = torch.randint(0, 4000, (max_bs, w8), dtype=torch.int32, generator=g)
    # the target's table under DCP 8: as wide as at DCP 1, its tail never allocated
    wide = torch.randint(0, 4000, (max_bs, 8 * w8), dtype=torch.int32, generator=g)
    wide[:, :w8] = t128
    t16 = t128.repeat_interleave(8, dim=1) * 8 + torch.arange(8, dtype=torch.int32).repeat(w8)
    anchors = torch.zeros(running_bs, dtype=torch.int64)
    alloc8 = []
    for i in range(scheduled_bs):
        anchors[i] = rng.choice([rng.randrange(0, 3), rng.randrange(0, max_model_len - T - 1)])
        need = (int(anchors[i]) + 1) // 128 + 1
        alloc8.append(min(w8, max(1, need + rng.choice([-1, 0, 1]))))
    dcp8, d8 = _block(wide, alloc8, anchors, scheduled_bs, T, 128, max_model_len)
    dcp1, d1 = _block(t16, [8 * a for a in alloc8], anchors, scheduled_bs, T, 16, max_model_len)
    assert d8._block_rows_p == d1._block_rows_p == 8 * w8
    for name in ("slot_mapping", "context_lens", "blk_positions", "dspark_ctx_rows"):
        assert torch.equal(getattr(dcp8, name), getattr(dcp1, name)), name
    reads = (dcp1.dspark_ctx_rows.long() + 15) // 16
    for r in range(running_bs * T):
        n = int(reads[r])
        assert torch.equal(dcp8.dspark_bt16_rows[r, :n], dcp1.dspark_bt16_rows[r, :n]), r


# -- the target's builder stages the draft's context slots ---------------------------

_TABLES = [[7, 9] + [0] * 14, [4] + [0] * 15]


class _Group:
    def __init__(self):
        self.counts = {}

    def set_count(self, buffer, count):
        self.counts[id(buffer)] = count


def _builder_stub(rows=64):
    def buf(*size, dtype=torch.int64):
        return CpuGpuBuffer(*size, dtype=dtype, device="cpu", pin_memory=False)

    var = {"draft_ctx_slots": buf(rows)}
    var["draft_ctx_slots"].np[:] = 7
    for prefix in ("", "mixed_"):
        var[f"{prefix}positions"] = buf(rows)
        var[f"{prefix}block_tables"] = buf(2, 16, dtype=torch.int32)
        var[f"{prefix}block_tables"].np[:] = _TABLES
    return SimpleNamespace(
        model_runner=SimpleNamespace(
            forward_vars=var,
            draft_kv_builder=SimpleNamespace(kv_pool=SimpleNamespace(entry_tokens=128)),
        ),
        token_axis_scratch=np.empty(rows, dtype=np.int64),
    )


def _batch(lens, dummy=False):
    return SimpleNamespace(
        is_dummy_run=dummy,
        block_tables=_TABLES,
        total_seqs_num_prefill=len(lens),
        num_scheduled_tokens=np.asarray(lens, dtype=np.int32),
    )


@pytest.mark.parametrize("dcp", [1, 8])
def test_the_pro_builder_declares_draft_slots_only_under_dcp(monkeypatch, dcp):
    from atom.model_ops.attentions import draft_ctx_slots

    aiter_attention = gpu_stack_import("atom.model_ops.attentions.aiter_attention")
    if not torch.cuda.is_available():
        monkeypatch.setattr(
            draft_ctx_slots,
            "CpuGpuBuffer",
            functools.partial(CpuGpuBuffer, pin_memory=False),
        )
    builder = SimpleNamespace(max_num_batched_tokens=16, device="cpu")
    runner = SimpleNamespace(
        config=_config(dcp),
        drafter=SimpleNamespace(model=SimpleNamespace(kv_pool_spec=PRO_SPEC)),
        forward_vars={},
    )
    aiter_attention.AiterAttentionMetadataBuilder._declare_draft_ctx_slots(
        builder, runner
    )
    assert ("draft_ctx_slots" in runner.forward_vars) is (dcp > 1)


def test_the_pro_builders_prefill_stages_every_token_at_128_a_block_id(monkeypatch):
    """No DCP filter and 128 tokens a block id; `-1` out to the padded width."""
    aiter_attention = gpu_stack_import("atom.model_ops.attentions.aiter_attention")
    monkeypatch.setattr(
        aiter_attention.CommonAttentionBuilder,
        "stage_prefill",
        lambda self, *a: ("md", "pos"),
    )
    stub = _builder_stub()
    stub.model_runner.forward_vars["positions"].np[:5] = [126, 127, 128, 5, 6]
    builder = object.__new__(aiter_attention.AiterAttentionMetadataBuilder)
    builder.__dict__.update(vars(stub))
    builder._pro_ring_slots = None
    group = _Group()
    assert builder.stage_prefill(_batch([3, 2]), 2, 8, group) == ("md", "pos")
    buf = stub.model_runner.forward_vars["draft_ctx_slots"]
    assert group.counts[id(buf)] == 8
    assert buf.np[:8].tolist() == [
        7 * 128 + 126,
        7 * 128 + 127,
        9 * 128,
        4 * 128 + 5,
        4 * 128 + 6,
        -1,
        -1,
        -1,
    ]


def test_a_mixed_steps_decode_half_stages_after_its_prefill_rows():
    """A mixed step's draft slots are the step's rows: the prefill half's, then the
    decode half's off its own (``mixed_``) positions and table, `-1` to the step's
    padded width, counted into the group the prefill half goes up in."""
    stub = _builder_stub()
    var = stub.model_runner.forward_vars
    var["positions"].np[:3] = [0, 1, 2]
    var["mixed_positions"].np[:4] = [200, 201, 40, 41]
    group = _Group()
    DraftCtxSlotsMixin._stage_draft_ctx_slots(stub, _batch([3]), np.array([3]), 3, group)
    DraftCtxSlotsMixin._stage_draft_ctx_slots(
        stub, _batch([2, 2]), np.array([2, 2]), 9, group, "mixed_", first=3
    )
    buf = var["draft_ctx_slots"]
    assert group.counts[id(buf)] == 9
    assert buf.np[:9].tolist() == [
        7 * 128,
        7 * 128 + 1,
        7 * 128 + 2,
        9 * 128 + 72,
        9 * 128 + 73,
        4 * 128 + 40,
        4 * 128 + 41,
        -1,
        -1,
    ]


# -- the config and the draft admit it -----------------------------------------------


def test_the_m31_draft_refuses_dcp_and_the_pro_draft_takes_it():
    dspark = gpu_stack_import("atom.models.minimax_m3.dspark")
    pro = gpu_stack_import("atom.models.minimax_m3.pro_dspark")

    class _Past(Exception):
        """Raised by the first read past the DCP check."""

    class _Config:
        decode_context_parallel_size = 8

        @property
        def hf_config(self):
            raise _Past

    with pytest.raises(NotImplementedError, match="does not support DCP"):
        dspark.MiniMaxM3DSpark(_Config())
    with pytest.raises(_Past):
        pro.MiniMaxM3ProDSpark(_Config())
