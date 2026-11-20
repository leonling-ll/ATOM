# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's vision tower: Flash's stack with unfused q/k/v (Pro's packed mapping
keeps those names for its text attention), the model routing and the image processor."""

from __future__ import annotations

import types

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("aiter", reason="the vision stack imports the ATOM model stack")

from atom.models import registry
from atom.models.minimax_m3 import image_processing, vl
from atom.models.minimax_m3.model import (
    MiniMaxM3SparseForConditionalGenerationTextOnly,
)
from atom.models.minimax_m3.pro import (
    MiniMaxM3ProForConditionalGenerationTextOnly,
    MiniMaxM3ProForConditionalGenerationVL,
)
from atom.multimodal import registry as mm_registry

PRO_ARCH = "MiniMaxM3ProForConditionalGeneration"


def _tiny_config():
    vision = types.SimpleNamespace(
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=128,
        patch_size=2,
        num_channels=3,
        layer_norm_eps=1e-5,
        rope_theta=10000.0,
        hidden_act="gelu",
        img_token_compression_config={
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
        },
    )
    return types.SimpleNamespace(
        vision_config=vision,
        text_config=types.SimpleNamespace(hidden_size=32),
        projector_hidden_size=48,
        projector_hidden_act="gelu",
        multimodal_projector_bias=True,
    )


@pytest.fixture
def cpu_attention(monkeypatch):
    def sdpa_varlen(q, k, v, cu_seqlens, max_seqlen):
        out = []
        for s, e in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
            qs, ks, vs = (t[s:e].transpose(0, 1) for t in (q, k, v))
            out.append(F.scaled_dot_product_attention(qs, ks, vs).transpose(0, 1))
        return torch.cat(out).flatten(start_dim=-2)

    monkeypatch.setattr(vl, "_varlen_attention", sdpa_varlen)


def test_the_unfused_tower_is_named_as_the_checkpoint():
    tower, _, _ = vl.build_vision_modules(_tiny_config(), fused_qkv=False)
    names = {name for name, _ in tower.named_parameters()}
    for proj in ("q_proj", "k_proj", "v_proj"):
        assert f"vision_model.encoder.layers.0.self_attn.{proj}.weight" in names
        assert f"vision_model.encoder.layers.1.self_attn.{proj}.bias" in names
    assert not any("qkv_proj" in name for name in names)


def test_the_unfused_tower_computes_the_fused_one(cpu_attention):
    torch.manual_seed(0)
    cfg = _tiny_config()
    fused = vl.build_vision_modules(cfg, fused_qkv=True)
    unfused = vl.build_vision_modules(cfg, fused_qkv=False)
    for a, b in zip(fused, unfused):
        sd = {}
        for name, value in a.state_dict().items():
            if ".qkv_proj." not in name:
                sd[name] = value
                continue
            for shard, part in zip("qkv", value.chunk(3, dim=0)):
                sd[name.replace(".qkv_proj.", f".{shard}_proj.")] = part
        b.load_state_dict(sd, strict=True)
    grid = [[1, 4, 6]]
    pixels = torch.randn(24, 3 * 2 * 2 * 2)
    (tf, pf, mf), (tu, pu, mu) = fused, unfused
    with torch.no_grad():
        want = mf(pf(tf(pixels, grid)))
        got = mu(pu(tu(pixels, grid)))
    assert got.shape == (6, 32)
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_pro_vl_runs_pros_text_model_and_keeps_its_checkpoint_names():
    cls = MiniMaxM3ProForConditionalGenerationVL
    mro = cls.__mro__
    # Flash's VL __init__ builds the tower after super().__init__, which is Pro's text model.
    assert mro.index(vl.MiniMaxM3SparseForConditionalGenerationVL) < mro.index(
        MiniMaxM3ProForConditionalGenerationTextOnly
    )
    assert mro.index(MiniMaxM3ProForConditionalGenerationTextOnly) < mro.index(
        MiniMaxM3SparseForConditionalGenerationTextOnly
    )
    assert not any(
        k in cls.packed_modules_mapping for k in (".q_proj", ".k_proj", ".v_proj")
    )
    assert cls.fused_vision_qkv is False
    assert "language_model.model.dspark." in cls.skip_weight_prefixes
    assert vl.MiniMaxM3SparseForConditionalGenerationVL.fused_vision_qkv is True


def test_pro_routes_to_the_vl_model_and_its_image_inputs():
    assert registry.support_model_arch_dict[PRO_ARCH] == (
        "atom.models.minimax_m3.pro.MiniMaxM3ProForConditionalGenerationVL"
    )
    config = types.SimpleNamespace(
        hf_config=types.SimpleNamespace(architectures=[PRO_ARCH])
    )
    builder = mm_registry.get_multimodal_input_builder(config)
    assert builder is image_processing.build_minimax_m3_inputs
    assert mm_registry._MULTIMODAL_ARCH_TO_PROCESSOR[PRO_ARCH].endswith(
        "image_processing.minimax_m3_pro_processor"
    )


@pytest.mark.parametrize(
    ("preprocessor", "size"),
    [
        (
            {"size": {"height": 2016, "width": 2016}, "max_pixels": 4064256},
            {"shortest_edge": 56 * 56, "longest_edge": 4064256},
        ),
        (
            {"size": {"height": 1344, "width": 1344}, "min_pixels": 784},
            {"shortest_edge": 784, "longest_edge": 1344 * 1344},
        ),
        ({"size": [672, 672]}, None),
    ],
)
def test_the_pro_processor_reads_a_size_box_as_a_pixel_range(
    monkeypatch, preprocessor, size
):
    from transformers import AutoProcessor
    from transformers.image_processing_base import ImageProcessingMixin

    seen = {}

    def fake_from_pretrained(path, **kwargs):
        seen.update(kwargs, path=path)
        return types.SimpleNamespace(chat_template=None)

    monkeypatch.setattr(
        ImageProcessingMixin,
        "get_image_processor_dict",
        classmethod(lambda cls, path, **kw: (dict(preprocessor), {})),
    )
    monkeypatch.setattr(AutoProcessor, "from_pretrained", fake_from_pretrained)
    tokenizer = types.SimpleNamespace(chat_template="]~!b[{{ x }}")
    processor = image_processing.minimax_m3_pro_processor(
        types.SimpleNamespace(model="/models/m3-pro"), tokenizer, None
    )
    assert seen["path"] == "/models/m3-pro"
    assert seen.get("size") == size
    assert seen["trust_remote_code"] is True
    assert processor.chat_template == tokenizer.chat_template
