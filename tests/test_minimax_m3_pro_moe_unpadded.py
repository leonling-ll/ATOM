# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3.1-Pro's routed experts at TP8 (intermediate 640 per rank) in the
MXFP4 layout, padded to 768 (the 256 default of ``Mxfp4MoEMethod``) and
unpadded (``pad_align=128``), through aiter's FlyDSL ``fused_moe``, the kernel
the Pro server runs.

The padded rows and columns are zero weights, so both layouts compute the same
function. The outputs are not bit-identical: the unpadded shape picks a stage-2
kernel with a 128-wide K tile instead of 256, and stage 2 accumulates the top-k
rows with atomics, so two runs of one layout differ by as much. The test holds
the unpadded output to the padded one within that rounding. Run on one gfx950
GPU.
"""

from types import SimpleNamespace

import pytest
import torch
from conftest import gpu_stack_import
from torch import nn

aiter = pytest.importorskip("aiter")
pytest.importorskip("flydsl")
needs_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName,
    reason="needs gfx950",
)

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from aiter.ops.flydsl.moe_common import GateMode

from atom.model_ops.moe import Mxfp4MoEMethod
from atom.quantization.quark.utils import quant_mxfp4_online_even

HIDDEN, INTER, EXPERTS, TOP_K = 7168, 640, 16, 4
SWIGLU_LIMIT = 7.0
# Two padded runs differ by up to 7e-3 (one token); a misplaced block is ~1.
REL_L2_TOL = 2e-2


def _mxfp4_layer(pad_align, w1, w3, w2):
    """An MXFP4 expert layer as ``Mxfp4MoEMethod`` sizes and lays it out, holding
    the quantized ``w1`` / ``w3`` / ``w2`` (each (packed, scale))."""
    method = object.__new__(Mxfp4MoEMethod)
    method.pad_align = pad_align
    method.moe = SimpleNamespace(hidden_dim=HIDDEN)
    method.static_input_scales = False
    method.is_gfx1250 = False
    method.is_guinterleave = False
    method.use_triton = method.use_triton_ep = method.use_triton_decode = False
    layer = nn.Module()
    layer.hidden_size = HIDDEN
    layer.intermediate_size_per_partition = INTER
    layer.has_bias = False
    layer.activation = ActivationType.Swiglu
    layer.num_fused_shared_experts = 0
    with torch.device("cuda"):
        method.create_weights(layer, EXPERTS, HIDDEN, INTER, aiter.dtypes.fp4x2)
    half = method.intermediate_size
    for name, rows in (("w13_weight", 0), ("w13_weight_scale", 1)):
        dst = getattr(layer, name).data.view(torch.uint8)
        for start, (q, s) in ((0, w1), (half, w3)):
            src = (q, s)[rows].view(torch.uint8)
            dst[:, start : start + INTER, : src.shape[-1]] = src
    q, s = w2
    layer.w2_weight.data.view(torch.uint8)[:, :, : q.shape[-1]] = q.view(torch.uint8)
    layer.w2_weight_scale.data.view(torch.uint8)[:, :, : s.shape[-1]] = s.view(
        torch.uint8
    )
    method._process_weight_layout_after_loading(layer)
    return method, layer


def _quant(w):
    q, s = quant_mxfp4_online_even(w.reshape(-1, w.shape[-1]))
    return q.view(w.shape[0], w.shape[1], -1), s.view(w.shape[0], w.shape[1], -1)


def _run(method, layer, x, topk_weights, topk_ids):
    return fused_moe(
        x,
        layer.w13_weight,
        layer.w2_weight,
        topk_weights,
        topk_ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=layer.w13_weight_scale,
        w2_scale=layer.w2_weight_scale,
        gate_mode=GateMode.SEPARATED.value,
        swiglu_limit=SWIGLU_LIMIT,
    )


def _hf_config(kv_share):
    sparse = {"use_sparse_attention": True, "sparse_enable_kv_share": kv_share}
    return SimpleNamespace(intermediate_size=5120, sparse_attention_config=sparse)


@needs_gfx950
@pytest.mark.parametrize("tokens", [1, 8, 64, 512, 4096])
def test_unpadded_experts_match_padded(tokens):
    torch.manual_seed(0)
    dev = "cuda"
    w1 = _quant(torch.randn(EXPERTS, INTER, HIDDEN, device=dev) * 0.02)
    w3 = _quant(torch.randn(EXPERTS, INTER, HIDDEN, device=dev) * 0.02)
    w2 = _quant(torch.randn(EXPERTS, HIDDEN, INTER, device=dev) * 0.02)
    padded = _mxfp4_layer(256, w1, w3, w2)
    unpadded = _mxfp4_layer(128, w1, w3, w2)
    assert padded[0].intermediate_size == 768
    assert unpadded[0].intermediate_size == INTER

    x = torch.randn(tokens, HIDDEN, device=dev, dtype=torch.bfloat16)
    logits = torch.randn(tokens, EXPERTS, device=dev)
    topk_weights, topk_ids = torch.sigmoid(logits).topk(TOP_K, dim=-1)
    topk_weights = (topk_weights / topk_weights.sum(-1, keepdim=True)).float()
    topk_ids = topk_ids.to(torch.int32)

    ref = _run(*padded, x, topk_weights, topk_ids)
    rerun = _run(*padded, x, topk_weights, topk_ids)
    out = _run(*unpadded, x, topk_weights, topk_ids)
    assert ref.abs().sum() > 0
    noise, diff = _rel_l2(rerun, ref), _rel_l2(out, ref)
    assert (
        diff < REL_L2_TOL
    ), f"unpadded vs padded {diff:.2e} (padded rerun {noise:.2e})"


def _rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


class _Built(Exception):
    """Raised by the stand-in FusedMoE once it has the constructor's kwargs."""


@pytest.mark.parametrize("kv_share, expected", [(True, 128), (False, None)])
def test_moe_block_passes_pad_align(monkeypatch, kv_share, expected):
    """MiniMaxM3MoE at TP8 hands FusedMoE pad_align 128 for Pro, None for M3."""
    model = gpu_stack_import("atom.models.minimax_m3.model")
    seen = {}

    def fused_moe(**kwargs):
        seen.update(kwargs)
        raise _Built

    monkeypatch.setattr(model, "get_tensor_model_parallel_world_size", lambda: 8)
    monkeypatch.setattr(model, "ReplicatedLinear", lambda *a, **k: nn.Identity())
    monkeypatch.setattr(model, "FusedMoE", fused_moe)
    config = _hf_config(kv_share)
    config.__dict__.update(hidden_size=64, num_local_experts=16, num_experts_per_tok=4)
    with pytest.raises(_Built):
        model.MiniMaxM3MoE(config, layer_id=0, prefix="model.layers.4.mlp")
    assert seen["intermediate_size"] == 5120
    assert seen["pad_align"] == expected
