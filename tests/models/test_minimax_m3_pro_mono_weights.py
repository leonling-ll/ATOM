# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pro mono phase 1's weight binder on fake layers shaped like the loaded model.

What is pinned: every binding is production's own storage (no copy -- the
owner ruling is that an optimization never changes weights), the router band is
this rank's 48 experts, and each layout the kernel cannot read is refused by
name rather than misread. The routed planes are 3 GB at full size, so they
live on the meta device: the binder reads only their sizes and attributes.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom.models.minimax_m3.pro_mono.config import EXPERTS, HIDDEN, ProDims
from atom.models.minimax_m3.pro_mono.weights import bind_layer
from atom.mono.runtime.consensus import MonoUnsupported

DIMS = ProDims(8)


def _param(shape, dtype, device="cpu", shuffled=False):
    p = nn.Parameter(torch.empty(shape, dtype=dtype, device=device), False)
    if shuffled:
        p.is_shuffled = True
    return p


def _mxfp8_linear(rows, cols):
    return SimpleNamespace(
        native_a8_group_rows=1,
        runs_bf16=False,
        is_output_padded=False,
        weight=_param((rows, cols), torch.float8_e4m3fn, shuffled=True),
        weight_scale=_param((rows, cols // 32), torch.uint8),
    )


def _norm():
    return SimpleNamespace(
        weight=_param((HIDDEN,), torch.bfloat16), variance_epsilon=1e-6
    )


def _experts():
    meta = "meta"
    return SimpleNamespace(
        top_k=4,
        num_fused_shared_experts=0,
        quant_method=SimpleNamespace(
            is_guinterleave=False, fp8_activations=False, intermediate_pad=0
        ),
        w13_swizzle_layout=None,
        w13_weight=_param((EXPERTS, 1536, 3584), torch.uint8, meta, True),
        w13_weight_scale=_param((EXPERTS * 1536, 224), torch.uint8, meta),
        w2_weight=_param((EXPERTS, HIDDEN, 384), torch.uint8, meta, True),
        w2_weight_scale=_param((EXPERTS * HIDDEN, 24), torch.uint8, meta),
    )


def _layer(layer_id=10):
    gate = SimpleNamespace(weight=_param((EXPERTS, HIDDEN), torch.bfloat16), bias=None)
    gate.weight.data.copy_(
        torch.arange(EXPERTS, dtype=torch.bfloat16)[:, None].expand(EXPERTS, HIDDEN)
    )
    return SimpleNamespace(
        layer_num=layer_id,
        is_moe_layer=True,
        self_attn=SimpleNamespace(o_proj=_mxfp8_linear(HIDDEN, 3072)),
        post_attention_layernorm=_norm(),
        block_sparse_moe=SimpleNamespace(
            gate=gate,
            e_score_correction_bias=_param((EXPERTS,), torch.float32),
            experts=_experts(),
            shared_experts=SimpleNamespace(
                gate_up_proj=_mxfp8_linear(1280, HIDDEN),
                down_proj=_mxfp8_linear(HIDDEN, 640),
            ),
        ),
    )


@pytest.fixture(scope="module")
def layer():
    return _layer()


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_binds_production_storage_in_place(layer, rank):
    ahead = _norm()
    b = bind_layer(layer, DIMS, rank, ahead)
    attn, moe = layer.self_attn, layer.block_sparse_moe
    sh = moe.shared_experts
    for got, param in (
        (b.w_o, attn.o_proj.weight),
        (b.s_o, attn.o_proj.weight_scale),
        (b.g_post, layer.post_attention_layernorm.weight),
        (b.g_ahead, ahead.weight),
        (b.bias, moe.e_score_correction_bias),
        (b.w_sgu, sh.gate_up_proj.weight),
        (b.s_sgu, sh.gate_up_proj.weight_scale),
        (b.w_sd, sh.down_proj.weight),
        (b.s_sd, sh.down_proj.weight_scale),
    ):
        assert got.data_ptr() == param.data_ptr() and got.shape == param.shape
    for got, param in (
        (b.w13, moe.experts.w13_weight),
        (b.s13, moe.experts.w13_weight_scale),
        (b.w2, moe.experts.w2_weight),
        (b.s2, moe.experts.w2_weight_scale),
    ):
        assert got.is_meta and got.shape == param.shape
    # the router band: 48 rows of the replicated gate, a view (no copy)
    w = moe.gate.weight
    assert b.w_gate.shape == (48, HIDDEN) and b.w_gate.is_contiguous()
    assert b.w_gate.data_ptr() == w.data_ptr() + 48 * rank * HIDDEN * 2
    # rows of expert ids e (bf16 rounds ids past 256, monotonically)
    ids = torch.arange(EXPERTS, dtype=torch.bfloat16)[48 * rank : 48 * rank + 48]
    assert torch.equal(b.w_gate[:, 0], ids)
    assert (b.eps, b.eps_ahead) == (1e-6, 1e-6)


def test_no_lookahead_reader(layer):
    b = bind_layer(layer, DIMS, 0, None)
    assert b.g_ahead is None and b.eps_ahead is None


def _breaks():
    """(what to change on a fresh layer, the refusal it must raise)."""

    def o(layer):
        return layer.self_attn.o_proj

    def ex(layer):
        return layer.block_sparse_moe.experts

    def unshuffle(p):
        del p.is_shuffled

    return [
        (lambda m: setattr(m, "is_moe_layer", False), "not a MoE layer"),
        (lambda m: setattr(o(m), "native_a8_group_rows", None), "native 1x32"),
        (lambda m: setattr(o(m), "runs_bf16", True), "dequantized to bf16"),
        (lambda m: unshuffle(o(m).weight), "o_proj: weight not preshuffled"),
        (
            lambda m: setattr(
                o(m), "weight_scale", _param((HIDDEN, 3072 // 128), torch.uint8)
            ),
            "o_proj: scale",
        ),
        (
            lambda m: setattr(
                m.block_sparse_moe.gate,
                "weight",
                _param((EXPERTS, HIDDEN), torch.float32),
            ),
            "router",
        ),
        (
            lambda m: setattr(m.block_sparse_moe, "e_score_correction_bias", None),
            "correction bias",
        ),
        (lambda m: setattr(ex(m).quant_method, "is_guinterleave", True), "interleaved"),
        (
            lambda m: setattr(ex(m).quant_method, "intermediate_pad", 128),
            "intermediate_pad 128",
        ),
        (lambda m: setattr(ex(m), "w13_swizzle_layout", object()), "Triton MXFP4"),
        (
            lambda m: setattr(ex(m), "num_fused_shared_experts", 1),
            "fused into the routed",
        ),
        (lambda m: unshuffle(ex(m).w2_weight), "not preshuffled"),
        (
            # an unpadded routed intermediate (640, not 768)
            lambda m: setattr(
                ex(m),
                "w2_weight",
                _param((EXPERTS, HIDDEN, 320), torch.uint8, "meta", True),
            ),
            "w2:",
        ),
        (
            lambda m: setattr(m.block_sparse_moe, "shared_experts", None),
            "no shared expert",
        ),
        (
            lambda m: setattr(
                m.block_sparse_moe.shared_experts,
                "down_proj",
                _mxfp8_linear(HIDDEN, 768),
            ),
            "shared down_proj: weight",
        ),
    ]


@pytest.mark.parametrize("brk, why", _breaks())
def test_refuses_what_the_kernel_cannot_read(brk, why):
    layer = _layer()
    brk(layer)
    with pytest.raises(MonoUnsupported, match=why):
        bind_layer(layer, DIMS, 0)
