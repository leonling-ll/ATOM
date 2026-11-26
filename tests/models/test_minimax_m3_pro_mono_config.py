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
    MAX_ROWS_A16W4,
    MAX_ROWS_LAYOUT,
    Q_HEADS,
    ROUTED_ACT_NAMES,
    ROUTED_ACTS,
    SUPPORTED_TP,
    TOPK,
    ProDims,
    RoutedPlanes,
    act_refusal,
    config_refusal,
    gate_rows,
    latch_rows,
    max_rows,
    routed_width_refusal,
    rows_refusal,
)
from atom.mono.plan.execution import BLOCKS
from atom.mono.plan.shard import shard_refusal
from atom.mono.runtime import deployment


def test_tp8_widths():
    """The widths the kernel's stages and byte budget are written for."""
    d = ProDims(8)
    assert (d.heads, d.o_k) == (16, 3072)
    assert (d.e_rank, d.inter_valid, d.inter_s) == (48, 640, 640)
    assert (d.own_cols, d.own_tasks, d.nt) == (896, 7, 56)
    assert d.sh_gate_up == (1280, 7168) and d.sh_down == (7168, 640)
    # every rank's band of the router covers the experts once
    assert d.e_rank * d.tp == EXPERTS
    assert MAX_ROWS <= MAX_ROWS_LAYOUT


@pytest.mark.parametrize(
    "inter_r, w13, w2, s2_cols",
    [
        # Pro's pad_align 128 (the base since f69c3744): 640 unpadded, w2's
        # 20 scale groups padded to 24 by aiter's shuffle_scale
        (640, (1280, 1280 * 3584), 7168 * 320, 24),
        # the default pad_align 256
        (768, (1536, 1536 * 3584), 7168 * 384, 24),
    ],
)
def test_routed_planes_at_the_loaded_width(inter_r, w13, w2, s2_cols):
    """The routed planes FusedMoE leaves after its NVFP4 -> MXFP4 prep, per
    expert, at the width it loaded: w13 fp4x2 [2 inter_r, 3584] with scale
    [2 inter_r, 224]; w2 [7168, inter_r / 2] with scale [7168, s2_cols]."""
    p = RoutedPlanes(inter_r)
    assert (p.w13_rows, p.w13_bytes) == w13
    assert (p.w13_scale_cols, p.w13_scale_bytes) == (224, w13[0] * 224)
    assert (p.w2_bytes, p.w2_scale_cols, p.w2_scale_bytes) == (
        w2,
        s2_cols,
        7168 * s2_cols,
    )


@pytest.mark.parametrize("inter_r", [0, 512, 639, 656, 1.5 * 640, True, None])
def test_a_routed_width_the_kernel_cannot_address_is_refused(inter_r):
    assert routed_width_refusal(inter_r) is not None
    with pytest.raises(ValueError):
        RoutedPlanes(inter_r)


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
        ({"swiglu_beta": 0.0}, "SwiGLU beta"),
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


def test_mega_moe_turns_it_off(servable):
    # TP MegaMoE all-reduces a MoE layer's output in-kernel, so the next input
    # norm skips the all-reduce a mono layer's partial needs
    cfg = _atom_config()
    cfg.moe_backend = "mega"
    assert config_refusal(cfg) == "--moe-backend mega"


@pytest.mark.parametrize("rows", [0, 1, 8, MAX_ROWS])
def test_rows_in_range_latch(rows):
    assert rows_refusal(rows) is None
    got, why = latch_rows(rows, None)
    assert got == rows and (why is None) == (rows > 0)


@pytest.mark.parametrize("rows", [-1, MAX_ROWS_A16W4 + 1, 64, "8", 8.0, True])
def test_rows_out_of_range_turn_the_phase_off(rows):
    assert rows_refusal(rows) is not None
    assert latch_rows(rows, None)[0] == 0


@pytest.mark.parametrize("rows", [MAX_ROWS + 1, 20, 24, 25, MAX_ROWS_A16W4])
def test_a16w4_serves_up_to_32_rows_w4a4_and_w4a8_16(rows):
    """A16W4's FFN build takes two token tiles (17 .. 32 rows); W4A4 and W4A8
    hold one, so a latch past 16 turns them off with the reason."""
    assert (MAX_ROWS, MAX_ROWS_A16W4) == (16, 32)
    assert max_rows("bf16") == MAX_ROWS_A16W4
    assert rows_refusal(rows, "bf16") is None
    assert latch_rows(rows, None, act="bf16") == (rows, None)
    assert gate_rows(rows, rows) and gate_rows(1, rows)
    for act in ("fp4", "fp8"):
        assert max_rows(act) == MAX_ROWS
        assert rows_refusal(MAX_ROWS, act) is None
        why = rows_refusal(rows, act)
        assert why == f"mono_ffn_rows {rows} outside 0 .. 16 ({ROUTED_ACT_NAMES[act]})"
        assert latch_rows(rows, None, act=act) == (0, why)


def test_ranks_that_read_different_rows_turn_it_off_on_every_rank(monkeypatch):
    """The latch's collective (bad, max, -min of the rows, max, -min of the
    routed format): a rank that read 8 beside one that read 16 serves
    neither, as does a rank that refused its value."""
    import atom.distributed.group_agreement as ga

    def peers(*others):
        def max_each(values, group):
            rows = [tuple(values), *others]
            return tuple(max(col) for col in zip(*rows))

        return max_each

    bf16 = ROUTED_ACTS.index("bf16")
    monkeypatch.setattr(ga, "group_max_each", peers((0, 16, -16, bf16, -bf16)))
    assert latch_rows(8, "tp")[0] == 0
    monkeypatch.setattr(ga, "group_max_each", peers((1, 0, 0, 0, 0)))
    assert latch_rows(8, "tp") == (
        0,
        "another TP rank refused mono_ffn_rows or mono_routed_act",
    )
    monkeypatch.setattr(ga, "group_max_each", peers((0, 8, -8, bf16, -bf16)))
    assert latch_rows(8, "tp") == (8, None)
    # one rank read "fp4", another the default "bf16": the formats differ,
    # neither serves
    monkeypatch.setattr(ga, "group_max_each", peers((0, 8, -8, 0, 0)))
    rows, why = latch_rows(8, "tp")
    assert rows == 0 and "'fp4' and 'bf16'" in why
    fp8 = ROUTED_ACTS.index("fp8")
    monkeypatch.setattr(ga, "group_max_each", peers((0, 8, -8, fp8, -fp8)))
    rows, why = latch_rows(8, "tp", act="fp4")
    assert rows == 0 and "'fp4' and 'fp8'" in why


@pytest.mark.parametrize("act", ["bf16", "fp4", "fp8"])
def test_the_routed_formats_latch(act):
    assert act_refusal(act) is None
    assert latch_rows(16, None, act=act) == (16, None)


def test_the_routed_formats_keep_their_agreement_index():
    """The latch agrees on ``ROUTED_ACTS.index``: a format appended, never
    reordered, keeps an older rank's index meaning the same format."""
    assert ROUTED_ACTS[:2] == ("fp4", "fp8") and ROUTED_ACTS[2] == "bf16"
    assert set(ROUTED_ACT_NAMES) == set(ROUTED_ACTS)
    assert ROUTED_ACT_NAMES["bf16"] == "A16W4"


@pytest.mark.parametrize("act", ["fp16", "BF16", "FP8", "", None, 8])
def test_an_unknown_routed_format_turns_the_phase_off(act):
    assert act_refusal(act) is not None
    rows, why = latch_rows(16, None, act=act)
    assert rows == 0 and "mono_routed_act" in why


def test_the_gate_takes_only_rows_inside_the_latched_limit():
    assert gate_rows(8, 8) and gate_rows(1, 16)
    assert not gate_rows(9, 8)
    assert not gate_rows(0, 8)
    assert not gate_rows(4, 0)  # off
    # never past the widest build's (A16W4's) rows
    assert not gate_rows(MAX_ROWS_A16W4 + 1, MAX_ROWS_A16W4 + 1)


def test_variants_take_the_row_limit():
    """``ATOM_M3PRO_VARIANTS`` names the field (unknown keys raise, so an older
    build rejects it rather than ignoring it); 16 by default, 0 turns it off."""
    pro_attn = pytest.importorskip(
        "atom.model_ops.minimax_m3.pro_attn",
        reason="pro_attn needs aiter, including aiter.ops.triton.fused_kv_cache",
    )
    assert pro_attn.ProVariants().mono_ffn_rows == 16
    assert pro_attn._parse_variants({"mono_ffn_rows": 0}).mono_ffn_rows == 0
    assert pro_attn._parse_variants({"mono_ffn_rows": 8}).mono_ffn_rows == 8
    # A16W4 (production's a16w4 numerics) unless the file names W4A8 / W4A4
    assert pro_attn.ProVariants().mono_routed_act == "bf16"
    for act in ("fp4", "fp8"):
        v = pro_attn._parse_variants({"mono_routed_act": act})
        assert v.mono_routed_act == act


@pytest.mark.parametrize("level", [1, 2, 3])
def test_a_compiled_level_is_refused(servable, level):
    """The gate is Python in the decoder layer's forward: only level 0 runs it
    as written."""
    cfg = _atom_config()
    cfg.compilation_config.level = level
    assert config_refusal(cfg) is not None
