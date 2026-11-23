# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pro mono phase 1's plain-Python decisions: the TP8 widths the kernels bake,
the deployments it refuses, and the latched row limit.

A wrong width here makes a kernel read the right bytes at the wrong offset, and
a refusal that does not fire serves a deployment the kernels were never built
for, so each is pinned on the CPU rather than only inside a kernel."""

from types import SimpleNamespace

import pytest

from atom.models.minimax_m3.pro_mono import config as pro_config
from atom.models.minimax_m3.pro_mono.config import (
    EXPERTS,
    HIDDEN,
    INTER_FULL,
    MAX_ROWS,
    MAX_ROWS_LAYOUT,
    Q_HEADS,
    SUPPORTED_TP,
    TOPK,
    ProDims,
    config_refusal,
    gate_rows,
    latch_rows,
    rows_refusal,
)
from atom.mono.plan.execution import BLOCKS
from atom.mono.plan.shard import shard_refusal
from atom.mono.runtime import deployment


def test_tp8_widths():
    """The widths the design's stage table and byte budget are written for."""
    d = ProDims(8)
    assert (d.heads, d.o_k) == (16, 3072)
    assert (d.e_rank, d.inter_valid, d.inter_r, d.inter_s) == (48, 640, 768, 640)
    assert (d.own_cols, d.own_tasks, d.nt) == (896, 7, 56)
    # the routed planes FusedMoE leaves after its NVFP4 -> MXFP4 prep, per
    # expert: w13 fp4x2 [1536, 3584], scale [1536, 224]; w2 [7168, 384], [7168, 24]
    assert (d.w13_rows, d.w13_bytes) == (1536, 1536 * 3584)
    assert (d.w13_scale_cols, d.w13_scale_bytes) == (224, 1536 * 224)
    assert (d.w2_bytes, d.w2_scale_cols, d.w2_scale_bytes) == (
        7168 * 384,
        24,
        7168 * 24,
    )
    assert d.sh_gate_up == (1280, 7168) and d.sh_down == (7168, 640)
    # every rank's band of the router covers the experts once
    assert d.e_rank * d.tp == EXPERTS
    assert MAX_ROWS <= MAX_ROWS_LAYOUT


def test_supported_tp_is_what_dims_accepts():
    for tp in SUPPORTED_TP:
        assert shard_refusal(ProDims, tp) is None


@pytest.mark.parametrize("tp", [1, 2, 3, 4, 5, 16])
def test_every_other_tp_is_refused_by_name(tp):
    """TP4 divides every width and is still refused: the placement is TP8's."""
    why = shard_refusal(ProDims, tp)
    assert why is not None and f"TP {tp}" in why


def _hf_config(**over):
    fields = {
        "hidden_size": HIDDEN,
        "num_attention_heads": Q_HEADS,
        "head_dim": 256,
        "partial_rotary_factor": 0.25,
        "num_local_experts": EXPERTS,
        "num_experts_per_tok": TOPK,
        "intermediate_size": INTER_FULL,
        "n_shared_experts": 1,
        "scoring_func": "sigmoid",
        "use_routing_bias": True,
        "routed_scaling_factor": 2.0,
        "n_group": None,
        "swiglu_alpha": 1.702,
        "swiglu_limit": 7.0,
        "sparse_attention_config": {
            "value_head_dim": 192,
            "sparse_indexer_lookahead_num_layers": 2,
        },
    }
    fields.update(over)
    return SimpleNamespace(**fields)


def _atom_config(spec_k=7, **hf_over):
    return SimpleNamespace(
        hf_config=_hf_config(**hf_over),
        tensor_parallel_size=8,
        parallel_config=SimpleNamespace(data_parallel_size=1),
        pipeline_parallel_size=1,
        enable_expert_parallel=False,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        is_multinode_tp=False,
        enable_tbo=False,
        enable_tbo_decode=False,
        kv_transfer_config=None,
        kv_cache_block_size=16,
        compilation_config=SimpleNamespace(
            level=0,
            cudagraph_mode=SimpleNamespace(uses_tr_hybrid=lambda: False),
        ),
        speculative_config=(
            None
            if spec_k is None
            else SimpleNamespace(method="dspark", num_speculative_tokens=spec_k)
        ),
    )


@pytest.fixture
def servable(monkeypatch):
    """Everything outside the model config satisfied (gfx950, 256 CUs, not a
    plugin, mono on), so the refusal reports on the deployment under test."""
    monkeypatch.setattr(deployment, "_gfx", lambda: "gfx950")
    monkeypatch.setattr(deployment, "is_vllm", lambda: False)
    monkeypatch.setattr(deployment, "is_sglang", lambda: False)
    monkeypatch.setattr(deployment, "_cu_count", lambda: BLOCKS)
    monkeypatch.setattr(pro_config.envs, "ATOM_MONO_ENABLE", True)


def test_the_shipped_pro_deployment_is_served(servable):
    assert config_refusal(_atom_config()) is None


def test_plain_decode_and_decode_cp_are_served(servable):
    cfg = _atom_config(spec_k=None)
    cfg.decode_context_parallel_size = 8
    assert config_refusal(cfg) is None


@pytest.mark.parametrize(
    "change, why",
    [
        ({"tensor_parallel_size": 4}, "TP 4"),
        ({"kv_cache_block_size": 128}, "block size 128"),
        ({"enable_expert_parallel": True}, "expert parallel"),
    ],
)
def test_deployment_refusals(servable, change, why):
    cfg = _atom_config()
    for k, v in change.items():
        setattr(cfg, k, v)
    assert why in (config_refusal(cfg) or "")


@pytest.mark.parametrize(
    "over, why",
    [
        ({"hidden_size": 6144}, "hidden 6144"),
        ({"num_local_experts": 256}, "256 experts"),
        ({"num_experts_per_tok": 8}, "top-8"),
        ({"intermediate_size": 3072}, "intermediate 3072"),
        ({"n_shared_experts": 0}, "0 shared experts"),
        ({"scoring_func": "softmax"}, "scoring 'softmax'"),
        ({"use_routing_bias": False}, "no routing bias"),
        ({"routed_scaling_factor": 1.0}, "routed scaling 1.0"),
        ({"n_group": 8}, "grouped top-k"),
        ({"swiglu_limit": 10.0}, "SwiGLU limit"),
    ],
)
def test_checkpoint_refusals(servable, over, why):
    assert config_refusal(_atom_config(**over)) == why


def test_a_verify_wider_than_max_rows_is_refused(servable):
    assert config_refusal(_atom_config(spec_k=MAX_ROWS - 1)) is None
    assert "verify width 17" in config_refusal(_atom_config(spec_k=MAX_ROWS))


def test_another_drafter_is_refused(servable):
    cfg = _atom_config()
    cfg.speculative_config.method = "eagle3"
    assert config_refusal(cfg) == "not DSpark"


def test_the_switch_turns_it_off(servable, monkeypatch):
    monkeypatch.setattr(pro_config.envs, "ATOM_MONO_ENABLE", False)
    assert config_refusal(_atom_config()) == "switched off"


@pytest.mark.parametrize("rows", [0, 1, 8, MAX_ROWS])
def test_rows_in_range_latch(rows):
    assert rows_refusal(rows) is None
    got, why = latch_rows(rows, None)
    assert got == rows and (why is None) == (rows > 0)


@pytest.mark.parametrize("rows", [-1, MAX_ROWS + 1, 32, "8", 8.0, True])
def test_rows_out_of_range_turn_the_phase_off(rows):
    assert rows_refusal(rows) is not None
    assert latch_rows(rows, None)[0] == 0


def test_ranks_that_read_different_rows_turn_it_off_on_every_rank(monkeypatch):
    """The latch's collective (bad, max, -min): a rank that read 8 beside one
    that read 16 serves neither, as does a rank that refused its value."""
    import atom.distributed.group_agreement as ga

    def peers(*others):
        def max_each(values, group):
            rows = [tuple(values), *others]
            return tuple(max(col) for col in zip(*rows))

        return max_each

    monkeypatch.setattr(ga, "group_max_each", peers((0, 16, -16)))
    assert latch_rows(8, "tp")[0] == 0
    monkeypatch.setattr(ga, "group_max_each", peers((1, 0, 0)))
    assert latch_rows(8, "tp") == (0, "another TP rank refused mono_ffn_rows")
    monkeypatch.setattr(ga, "group_max_each", peers((0, 8, -8)))
    assert latch_rows(8, "tp") == (8, None)


def test_the_gate_takes_only_rows_inside_the_latched_limit():
    assert gate_rows(8, 8) and gate_rows(1, 16)
    assert not gate_rows(9, 8)
    assert not gate_rows(0, 8)
    assert not gate_rows(4, 0)  # off
    assert not gate_rows(17, MAX_ROWS + 1)  # never past MAX_ROWS


def test_variants_take_the_row_limit():
    """``ATOM_M3PRO_VARIANTS`` names the field (unknown keys raise, so an older
    build rejects it rather than ignoring it); off by default."""
    pro_attn = pytest.importorskip(
        "atom.model_ops.minimax_m3.pro_attn",
        reason="pro_attn needs aiter, including aiter.ops.triton.fused_kv_cache",
    )
    assert pro_attn.ProVariants().mono_ffn_rows == 0
    assert pro_attn._parse_variants({"mono_ffn_rows": 8}).mono_ffn_rows == 8
