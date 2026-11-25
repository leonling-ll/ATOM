# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's mixed prefill+decode step metadata (`prepare_mixed`).

A mixed batch lists its prefill rows first and its decode rows after. Its
metadata is the prefill half, as today's prefill build gives it for the prefill
rows alone, carrying the decode half on ``mixed_decode``, as today's decode build
gives it for the decode rows alone, in buffers of its own. Both references are
built through `build` by the same builder, before the mixed step, so every
request already owns its ring slot.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.utils import envs

BLOCK = 16
K = 5  # DSpark draft tokens: a decode row verifies K + 1 query tokens
Q = K + 1
# Two prefill chunks: one resuming behind a cached prefix, one fresh.
PREFILL_CACHED = [32, 0]
PREFILL_NEW = [20, 13]
DECODE_LENS = [40, 17, 64, 23, 90]  # context, verify rows included


def test_mixed_batch_is_off_by_default(monkeypatch):
    monkeypatch.delenv("ATOM_MIXED_BATCH", raising=False)
    assert not envs.ATOM_MIXED_BATCH
    monkeypatch.setenv("ATOM_MIXED_BATCH", "1")
    assert envs.ATOM_MIXED_BATCH


gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the builder stages through the GPU"
)


def make_runner(monkeypatch, mixed: bool):
    pytest.importorskip("aiter")
    from atom.distributed import ulysses_sp
    from atom.model_engine.model_runner import ModelRunner
    from atom.model_ops.attentions import aiter_attention, backends
    from atom.model_ops.minimax_m3 import pro_attn

    monkeypatch.setenv("ATOM_MIXED_BATCH", "1" if mixed else "0")
    monkeypatch.setenv("ATOM_USE_UNIFIED_ATTN", "0")
    # the compressed index cache: every request owns a ring at `pro_ring_base`
    monkeypatch.setattr(
        pro_attn, "_variants", pro_attn.ProVariants(index_compress=True)
    )
    for module in (ulysses_sp, backends):
        monkeypatch.setattr(module, "attn_head_shard_size", lambda: 1)
    for module in (backends, aiter_attention):
        if hasattr(module, "get_tp_group"):
            monkeypatch.setattr(
                module, "get_tp_group", lambda: SimpleNamespace(world_size=1)
            )
        if hasattr(module, "get_dcp_world_size"):
            monkeypatch.setattr(module, "get_dcp_world_size", lambda: 1)
        if hasattr(module, "get_dcp_rank"):
            monkeypatch.setattr(module, "get_dcp_rank", lambda: 0)
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cuda", 0)
    runner.max_bs = 8
    runner.max_num_batched_tokens = 128
    runner.block_size = BLOCK
    runner.kv_cache_dtype = "bf16"
    runner.num_spec_tokens = K
    runner.has_mla_indexer = False
    runner.use_mrope = False
    runner.enforce_eager = True
    runner.arange_np = np.arange(128, dtype=np.int64)
    runner.config = SimpleNamespace(
        pipeline_parallel_size=1,
        max_model_len=256,
        max_num_batched_tokens=runner.max_num_batched_tokens,
        enable_tbo=False,
        enable_tbo_decode=False,
        prefill_context_parallel_size=1,
        kv_cache_dtype="bf16",
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=SimpleNamespace(num_speculative_tokens=K),
        hf_config=SimpleNamespace(
            model_type="minimax_m3",
            num_attention_heads=16,
            num_key_value_heads=1,
            sparse_attention_config={
                "use_sparse_attention": True,
                "sparse_enable_kv_share": True,
                "sparse_block_size": BLOCK,
            },
        ),
    )
    runner.tokenID_processor = SimpleNamespace(
        num_rejected=None, verify_ctx_lens=lambda ctx: ctx
    )
    runner.forward_vars = {
        name: CpuGpu(64, dtype=torch.int32, device=runner.device)
        for name in ("input_ids", "decode_src", "batch_info")
    }
    runner.forward_vars["positions"] = CpuGpu(
        128, dtype=torch.int64, device=runner.device, publication_group="positions"
    )
    builder = aiter_attention.AiterAttentionMetadataBuilder(model_runner=runner)
    assert builder._has_pro_sparse_attention and builder._pro_ring_slots is not None
    runner.attn_metadata_builder = builder
    runner._init_forward_vars_ring()
    runner._init_h2d_publication()
    return runner, builder


def CpuGpu(*args, **kwargs):
    from atom.utils import CpuGpuBuffer

    return CpuGpuBuffer(*args, **kwargs)


def _tables(lens, first):
    return [
        np.arange(first + 16 * i, first + 16 * i + -(-n // BLOCK), dtype=np.int32)
        for i, n in enumerate(lens)
    ]


def make_batch(prefill: bool, decode: bool):
    """Request ids 0, 1 are the prefill chunks, 2.. the decode sequences."""
    p_ctx = [c + n for c, n in zip(PREFILL_CACHED, PREFILL_NEW)]
    rows = []  # (req_id, context, scheduled, cached)
    if prefill:
        rows += [(i, p_ctx[i], PREFILL_NEW[i], PREFILL_CACHED[i]) for i in range(2)]
    if decode:
        rows += [(2 + i, n, Q, n - Q) for i, n in enumerate(DECODE_LENS)]
    req_ids = [r[0] for r in rows]
    ctx = np.array([r[1] for r in rows], dtype=np.int32)
    tables = [_tables([r[1]], 1 + 16 * r[0])[0] for r in rows]
    num_p = 2 if prefill else 0
    num_d = len(DECODE_LENS) if decode else 0
    sched = np.array([r[2] for r in rows], dtype=np.int32)
    return SimpleNamespace(
        req_ids=req_ids,
        context_lens=ctx,
        num_scheduled_tokens=sched,
        num_cached_tokens=[r[3] for r in rows],
        block_tables=tables,
        total_seqs_num=len(rows),
        total_seqs_num_prefill=num_p,
        total_seqs_num_decode=num_d,
        total_tokens_num=int(sched.sum()),
        total_tokens_num_prefill=int(sched[:num_p].sum()),
        total_tokens_num_decode=int(sched[num_p:].sum()),
        is_dummy_run=False,
        state_maintenance_ops=SimpleNamespace(
            relocations=[], checkpoint_stores=[], checkpoint_restores=[]
        ),
    )


def run_step(runner, builder, batch, running_bs, running_tokens, is_prefill):
    """One step's `build`, framed as `ModelRunner` frames it: the query layout
    published first, the staging marked enqueued after."""
    runner._advance_forward_vars()
    runner._gate_staging_reuse()
    builder.publish_query_metadata(
        np.asarray(batch.num_scheduled_tokens, np.int32), running_bs, running_tokens
    )
    md, positions = builder.build(
        batch, running_bs, running_tokens, Q, is_prefill=is_prefill
    )
    runner._mark_staging_h2d_enqueued()
    runner._record_forward_vars_event()
    torch.cuda.synchronize()
    return md, positions


def fields(md):
    """Every field the build set, on the host; the CSR's indices as far as its
    indptr reaches (the rest is the buffer's stale tail)."""
    out = {}
    for name, value in vars(md).items():
        if name.startswith("_") or name == "mixed_decode" or value is None:
            continue
        if isinstance(value, torch.Tensor):
            if name == "kv_indices":
                value = value[: int(md.kv_indptr[-1])]
            value = value.cpu().clone()
        out[name] = value
    return out


def assert_same(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(actual[name], value), name
        else:
            assert actual[name] == value, name


@gpu
@pytest.mark.parametrize("pad", [0, 1])
def test_each_half_is_its_rows_built_alone(monkeypatch, pad):
    """2 prefill chunks and 5 decode sequences at k=5. ``pad`` decode rows (and
    their tokens) of graph padding go to the decode half."""
    runner, builder = make_runner(monkeypatch, mixed=True)
    p_tokens = sum(PREFILL_NEW)
    d_bs = len(DECODE_LENS) + pad
    d_tokens = d_bs * Q

    md, pos = run_step(runner, builder, make_batch(True, False), 2, p_tokens, True)
    prefill, prefill_pos = fields(md), pos.cpu().clone()
    md, pos = run_step(runner, builder, make_batch(False, True), d_bs, d_tokens, False)
    decode, decode_pos = fields(md), pos.cpu().clone()

    mixed = make_batch(True, True)
    md, pos = run_step(runner, builder, mixed, 2 + d_bs, p_tokens + d_tokens, True)
    half = md.mixed_decode
    assert_same(fields(md), prefill)
    assert_same(fields(half), decode)
    # the step's positions: the prefill rows', then the decode rows'
    assert torch.equal(pos.cpu(), torch.cat([prefill_pos, decode_pos]))
    # the decode half lives in buffers of its own, apart from the prefill half's
    var = runner.forward_vars
    for name in ("slot_mapping", "context_lens", "block_tables", "cu_seqlens_q"):
        assert getattr(half, name).data_ptr() == var[f"mixed_{name}"].gpu.data_ptr()
        assert getattr(md, name).data_ptr() == var[name].gpu.data_ptr()
    assert half.pro_ring_base.data_ptr() == var["mixed_pro_ring_base"].gpu.data_ptr()
    # every request keeps the ring it was given
    assert (half.pro_ring_base[: len(DECODE_LENS)] >= 0).all()
    assert (md.pro_ring_base >= 0).all()


@gpu
def test_off_builds_no_mixed_step(monkeypatch):
    """Off: no decode-half buffers, and a prefill step builds as it always has,
    whatever rows the batch holds."""
    runner, builder = make_runner(monkeypatch, mixed=False)
    assert not any(name.startswith("mixed_") for name in runner.forward_vars)
    batch = make_batch(True, True)
    p_tokens = sum(PREFILL_NEW)
    with pytest.raises(NotImplementedError, match="ATOM_MIXED_BATCH"):
        builder.prepare_mixed(batch, 7, batch.total_tokens_num, Q)
    md, _ = run_step(runner, builder, batch, 7, batch.total_tokens_num, True)
    assert not hasattr(md, "mixed_decode")
    assert md.max_seqlen_q == max(PREFILL_NEW)
    assert int(md.cu_seqlens_q[2]) == p_tokens
