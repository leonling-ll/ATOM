# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's packed projections (q, k, index q and index k in ``qkv_proj``, and
the names that route the checkpoint into it) and the routed scale folded into the
routing weights."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="the model stack imports aiter")

from aiter import QuantType, dtypes

from atom.config import QuantizationConfig
from atom.model_loader.weight_names import CheckpointNameRewriter
from atom.model_ops import linear as linear_mod
from atom.model_ops import moe as moe_mod
from atom.models.minimax_m3 import model as m3_model
from atom.models.minimax_m3.pro import (
    MiniMaxM3ProForConditionalGenerationVL,
)
from atom.quant_spec import LayerQuantConfig

PRO = MiniMaxM3ProForConditionalGenerationVL


def _target(name: str) -> tuple[str, object] | None:
    """The fused parameter and shard a checkpoint tensor lands in: the rewrite, then
    the first packed rule whose key it contains (as the loader's dispatch takes it)."""
    rewritten = CheckpointNameRewriter(
        weights_mapper=PRO.weights_mapper, weights_mapping=PRO.weights_mapping
    ).rewrite(name)
    for key, (fused, shard) in PRO.packed_modules_mapping.items():
        if key in rewritten:
            return rewritten.replace(key, fused), shard
    return rewritten, None


@pytest.mark.parametrize(
    "name, want",
    [
        (
            "language_model.model.layers.3.self_attn.q_proj.weight",
            ("language_model.model.layers.3.self_attn.qkv_proj.weight", "q"),
        ),
        (
            "language_model.model.layers.67.self_attn.k_proj.weight_scale_inv",
            ("language_model.model.layers.67.self_attn.qkv_proj.weight_scale", "k"),
        ),
        (
            "language_model.model.layers.0.self_attn.index_q_proj.weight",
            ("language_model.model.layers.0.self_attn.qkv_proj.weight", "index_q"),
        ),
        (
            "language_model.model.layers.9.self_attn.index_k_proj.weight_scale_inv",
            (
                "language_model.model.layers.9.self_attn.qkv_proj.weight_scale",
                "index_k",
            ),
        ),
        # the vision tower's attention keeps its names (its tower is unfused)
        (
            "vision_tower.vision_model.encoder.layers.2.self_attn.q_proj.weight",
            (
                "vision_tower.vision_model.encoder.layers.2.self_attn.q_proj.weight",
                None,
            ),
        ),
        (
            "vision_tower.vision_model.encoder.layers.2.self_attn.k_proj.bias",
            ("vision_tower.vision_model.encoder.layers.2.self_attn.k_proj.bias", None),
        ),
        (
            "language_model.model.layers.4.self_attn.o_proj.weight",
            ("language_model.model.layers.4.self_attn.o_proj.weight", None),
        ),
    ],
)
def test_checkpoint_names_route_into_the_packed_projections(name, want):
    assert _target(name) == want


def _mxfp8_config() -> QuantizationConfig:
    qc = QuantizationConfig()
    qc.global_spec = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=dtypes.fp8,
        weight_block_size=(1, 32),
        is_dynamic=True,
    )
    return qc


def _rows(n: int, k: int, first: int) -> torch.Tensor:
    """``n`` rows of FP8 bytes, row ``i`` all ``first + i`` (a byte pattern, not a value)."""
    rows = torch.arange(first, first + n, dtype=torch.uint8)[:, None].expand(n, k)
    return rows.contiguous().view(torch.float8_e4m3fn)


@pytest.mark.parametrize("rank", [0, 3])
def test_qk_proj_takes_the_q_shard_and_the_whole_k(monkeypatch, rank):
    tp = 4
    monkeypatch.setattr(
        linear_mod,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=tp, rank_in_group=rank),
    )
    hidden, head, heads = 64, 32, 8
    layer = linear_mod.QKVParallelLinear(
        hidden,
        head,
        heads,
        1,
        quant_config=_mxfp8_config(),
        prefix="model.layers.0.self_attn.qk_proj",
        v_head_size=0,
    )
    local_q = heads // tp * head
    assert layer.native_a8_group_rows is not None
    assert layer.weight.shape == (local_q + head, hidden)
    assert layer.weight_scale.shape[0] == local_q + head

    q = _rows(heads * head, hidden, 0)
    k = _rows(head, hidden, 200)
    layer.weight_loader(layer.weight, q, "q")
    layer.weight_loader(layer.weight, k, "k")
    q_scale = torch.arange(heads * head, dtype=torch.uint8)[:, None].repeat(1, 2)
    k_scale = torch.full((head, 2), 250, dtype=torch.uint8)
    layer.weight_loader(layer.weight_scale, q_scale.view(layer.weight_scale.dtype), "q")
    layer.weight_loader(layer.weight_scale, k_scale.view(layer.weight_scale.dtype), "k")

    w = layer.weight.data.view(torch.uint8)
    s = layer.weight_scale.data.view(torch.uint8)
    lo = rank * local_q
    assert torch.equal(
        w[:local_q, 0], torch.arange(lo, lo + local_q, dtype=torch.uint8)
    )
    assert torch.equal(w[local_q:], k.view(torch.uint8))
    assert torch.equal(s[:local_q], q_scale[lo : lo + local_q])
    assert torch.equal(s[local_q:], k_scale)


def test_index_qk_proj_stacks_index_q_over_index_k(monkeypatch):
    monkeypatch.setattr(
        linear_mod,
        "get_tp_group",
        lambda: SimpleNamespace(world_size=8, rank_in_group=5),
    )
    hidden, nq, nk = 64, 64, 32
    layer = linear_mod.MergedReplicatedLinear(
        hidden,
        [nq, nk],
        quant_config=_mxfp8_config(),
        prefix="model.layers.0.self_attn.index_qk_proj",
    )
    iq, ik = _rows(nq, hidden, 0), _rows(nk, hidden, 100)
    layer.weight_loader(layer.weight, iq, 0)
    layer.weight_loader(layer.weight, ik, 1)
    w = layer.weight.data.view(torch.uint8)
    assert torch.equal(w, torch.cat([iq, ik]).view(torch.uint8))


def test_select_experts_folds_the_scale_only_when_asked(monkeypatch):
    seen = []

    def fake_gating(w, ids, logits, bias, renorm, scale, **_):
        seen.append(scale)

    monkeypatch.setattr(moe_mod, "topk_gating", fake_gating)
    monkeypatch.setattr(moe_mod, "pass_scheduled_tokens_gpu", lambda n: None)
    logits = torch.zeros(3, 8)
    for fold in (False, True):
        moe_mod.FusedMoE.select_experts(
            hidden_states=logits,
            router_logits=logits,
            top_k=2,
            use_grouped_topk=False,
            renormalize=True,
            scoring_func="sigmoid",
            routed_scaling_factor=2.0,
            fold_routed_scaling_factor=fold,
        )
    assert seen == [1.0, 2.0]


def test_select_logical_passes_the_layers_fold(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        moe_mod.FusedMoE,
        "select_experts",
        staticmethod(lambda **kw: seen.update(kw)),
    )
    layer = SimpleNamespace(
        use_grouped_topk=False,
        top_k=4,
        renormalize=True,
        topk_group=None,
        num_expert_group=None,
        custom_routing_function=None,
        scoring_func="sigmoid",
        e_score_correction_bias=None,
        shared_expert_scoring_func=None,
        routed_scaling_factor=2.0,
        fold_routed_scaling_factor=True,
    )
    moe_mod._select_logical(layer, None, None, 384, 0)
    assert seen["fold_routed_scaling_factor"] is True


class _Experts(nn.Module):
    def __init__(self, fold: bool):
        super().__init__()
        self.fold_routed_scaling_factor = fold

    def forward(self, hidden_states, router_logits):
        return hidden_states + 1.0


@pytest.mark.parametrize("fold, want", [(True, 2.0), (False, 4.0)])
def test_moe_multiplies_only_when_the_routing_did_not(fold, want):
    moe = m3_model.MiniMaxM3MoE.__new__(m3_model.MiniMaxM3MoE)
    nn.Module.__init__(moe)
    moe.gate = nn.Identity()
    moe.experts = _Experts(fold)
    moe.routed_scaling_factor = 2.0
    moe.fuse_shared_experts = False
    moe.shared_experts = None
    out = moe(torch.ones(2, 4))
    assert torch.equal(out, torch.full((2, 4), want))
