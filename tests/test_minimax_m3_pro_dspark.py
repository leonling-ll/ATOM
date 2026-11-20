# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's inline DSpark draft: config routing, the weights it reads out
of the target checkpoint, its page-16 block tables and its RoPE magnitude."""

from __future__ import annotations

import math
import types

import pytest
import torch

pytest.importorskip("aiter", reason="the draft imports the ATOM model stack")

from aiter import QuantType, dtypes
from transformers import PretrainedConfig

from atom.config import SpeculativeConfig
from atom.model_loader.weight_names import CheckpointNameRewriter
from atom.models.dspark_draft import draft_quant_config
from atom.models.minimax_m3 import pro_dspark
from atom.models.minimax_m3.dspark import _page16_table


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


def test_page16_table_of_a_page16_pool_is_the_table():
    bt = torch.tensor([[3, 7, 0], [5, 1, 2]], dtype=torch.int32)
    out = torch.full_like(bt, -1)
    _page16_table(bt, out, torch.arange(1, dtype=torch.int32))
    assert torch.equal(out, bt)


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


def test_block_rows_read_the_page16_table_itself(host_staging):
    """A table entry is one page-16: a block row's table is its request's own
    from the page holding its window's first key, its length what is left of
    the request's keys from there."""
    Draft = pro_dspark.MiniMaxM3ProDSpark
    draft = Draft.__new__(Draft)
    draft.max_num_seqs, draft._bt16 = 2, None
    draft.sliding_window = 32
    pos = torch.tensor([[50, 51], [3, 4]])
    slots = torch.tensor([[130, 131], [83, 84]])
    ctx = torch.tensor([52, 5], dtype=torch.int32)
    block_tables = torch.tensor([[9, 2, 6, 8], [5, 0, 0, 0]], dtype=torch.int32)
    md = types.SimpleNamespace()
    draft.prepare_paged_block(md, block_tables, pos, slots, ctx, [4, 1], 2, 16)
    # Rows at 50 and 51 attend keys 19.. and 20.., both on the table's second
    # page; the window runs into request 1's first page, which the length
    # (52 - 16) never reaches.
    assert md.dspark_bt16_rows.tolist() == [
        [2, 6, 8, 5],
        [2, 6, 8, 5],
        [5, 0, 0, 0],
        [5, 0, 0, 0],
    ]
    assert md.dspark_ctx_rows.tolist() == [36, 36, 5, 5]
    assert slots.tolist() == [[130, 131], [83, 84]]
    assert ctx.tolist() == [52, 5]


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
