# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pro mono phase 1's weight binder on fake layers shaped like the loaded model.

What is pinned: every binding is production's own storage (no copy -- the
owner ruling is that an optimization never changes weights), the router band is
this rank's 48 experts, and each layout the kernel cannot read is refused by
name rather than misread. The routed planes are 3 GB at full size, so they
live on the meta device: the binder reads only their sizes and attributes.

The routed planes are also built by production itself (aiter needed, CPU
only): ``FusedMoE._online_quant`` converting an NVFP4 source into the real
``Mxfp4MoEMethod``'s planes, then its gfx950 weight prep. The binder must take
what that leaves, and the kernel's addressing (``mono.device.mx``) and the
shared unshuffles must read back what the conversion wrote, pad included.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from atom.models.minimax_m3.pro_mono import weights as weights_mod
from atom.models.minimax_m3.pro_mono.config import EXPERTS, HIDDEN, ProDims
from atom.models.minimax_m3.pro_mono.weights import bind_layer
from atom.mono.runtime.consensus import MonoUnsupported

DIMS = ProDims(8)
# ``MiniMaxM3MoE``'s pad_align for Pro (f69c3744): a rank's 640 stays unpadded
PRO_PAD_ALIGN = 128


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


def _experts(inter_r=640):
    """The routed experts as the loader leaves them at ``inter_r`` (640: Pro's
    pad_align 128; 768: the default 256): w2's scale groups padded to 8."""
    meta = "meta"
    s2_cols = -(-inter_r // 32 // 8) * 8
    return SimpleNamespace(
        top_k=4,
        num_fused_shared_experts=0,
        quant_method=SimpleNamespace(
            is_guinterleave=False,
            fp8_activations=False,
            hidden_pad=0,
            intermediate_pad=0,
            intermediate_size=inter_r,
        ),
        w13_swizzle_layout=None,
        w13_weight=_param((EXPERTS, 2 * inter_r, 3584), torch.uint8, meta, True),
        w13_weight_scale=_param((EXPERTS * 2 * inter_r, 224), torch.uint8, meta),
        w2_weight=_param((EXPERTS, HIDDEN, inter_r // 2), torch.uint8, meta, True),
        w2_weight_scale=_param((EXPERTS * HIDDEN, s2_cols), torch.uint8, meta),
    )


def _layer(layer_id=10, inter_r=640):
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
            experts=_experts(inter_r),
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
        assert got.is_meta and got.shape == param.shape and got.dtype == torch.uint8
    # the router band: 48 rows of the replicated gate, a view (no copy)
    w = moe.gate.weight
    assert b.w_gate.shape == (48, HIDDEN) and b.w_gate.is_contiguous()
    assert b.w_gate.data_ptr() == w.data_ptr() + 48 * rank * HIDDEN * 2
    # rows of expert ids e (bf16 rounds ids past 256, monotonically)
    ids = torch.arange(EXPERTS, dtype=torch.bfloat16)[48 * rank : 48 * rank + 48]
    assert torch.equal(b.w_gate[:, 0], ids)
    assert (b.eps, b.eps_ahead) == (1e-6, 1e-6)


@pytest.mark.parametrize("inter_r", [640, 768])
def test_binds_the_width_the_loader_padded_to(inter_r):
    """The planes' width is read off the loaded method
    (``intermediate_size``), not assumed: Pro's 640 and the default pad's 768
    bind, each with its own plane sizes."""
    b = bind_layer(_layer(inter_r=inter_r), DIMS, 2)
    assert b.planes.inter_r == inter_r
    assert b.w13.shape == (EXPERTS, 2 * inter_r, HIDDEN // 2)
    assert b.w2.shape == (EXPERTS, HIDDEN, inter_r // 2)
    assert b.s2.shape == (EXPERTS * HIDDEN, b.planes.w2_scale_cols)


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
        (lambda m: setattr(ex(m), "source_is_nvfp4", True), "still the checkpoint"),
        (lambda m: setattr(ex(m).quant_method, "hidden_pad", 256), "hidden_pad 256"),
        (
            # an online "mxfp8" target: FP8 experts of the same rows
            lambda m: setattr(
                ex(m),
                "w13_weight",
                _param((EXPERTS, 1280, 7168), torch.float8_e4m3fn, "meta", True),
            ),
            "w13: torch.float8_e4m3fn",
        ),
        (
            # a scale plane kept 3-D (not the flattened shuffle_scale plane)
            lambda m: setattr(
                ex(m),
                "w2_weight_scale",
                _param((EXPERTS, HIDDEN, 24), torch.uint8, "meta"),
            ),
            "w2 scale:",
        ),
        (
            lambda m: setattr(ex(m), "num_fused_shared_experts", 1),
            "fused into the routed",
        ),
        (lambda m: unshuffle(ex(m).w2_weight), "not preshuffled"),
        (
            # w2 at the 256 pad's 768 while the loader says 640
            lambda m: setattr(
                ex(m),
                "w2_weight",
                _param((EXPERTS, HIDDEN, 384), torch.uint8, "meta", True),
            ),
            "w2:",
        ),
        (
            # w2's 20 scale groups not padded to shuffle_scale's 24
            lambda m: setattr(
                ex(m),
                "w2_weight_scale",
                _param((EXPERTS * HIDDEN, 20), torch.uint8, "meta"),
            ),
            "w2 scale:",
        ),
        (
            # w13 at the 768 rows while the loader says 640 (the refusal on
            # the base before this port, the other way round)
            lambda m: setattr(
                ex(m),
                "w13_weight",
                _param((EXPERTS, 1536, 3584), torch.uint8, "meta", True),
            ),
            r"w13: torch.uint8 \(384, 1536, 3584\), expected \(384, 1280, 3584\)",
        ),
        (
            lambda m: delattr(ex(m).quant_method, "intermediate_size"),
            "routed intermediate None",
        ),
        (
            lambda m: setattr(ex(m).quant_method, "intermediate_size", 600),
            "routed intermediate 600",
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


# ------------------------------------------- the planes as production makes them

ROUTED_ENV = (
    "ATOM_USE_TRITON_MOE",
    "ATOM_USE_TRITON_GEMM",
    "ATOM_USE_TRITON_MOE_DECODE",
    "ATOM_USE_TRITON_MOE_A4W4",
    "ATOM_MOE_GU_ITLV",
)


def _production(monkeypatch):
    """atom's MoE module with the recipe's gfx950 setting (no MoE env set)."""
    pytest.importorskip("aiter")
    moe_mod = importlib.import_module("atom.model_ops.moe")
    shuffle_mod = importlib.import_module("aiter.ops.shuffle")
    for env in ROUTED_ENV:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(moe_mod, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(shuffle_mod, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(
        moe_mod,
        "get_current_atom_config",
        lambda: SimpleNamespace(eplb_enable=False, moe_backend="standard"),
    )
    return moe_mod


def _specs():
    from aiter import QuantType, dtypes

    from atom.quant_spec import NVFP4_DTYPE, LayerQuantConfig

    source = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=NVFP4_DTYPE,
        is_dynamic=True,
        quant_method="compressed-tensors",
    )
    target = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=dtypes.fp4x2,
        is_dynamic=True,
        quant_method="quark",
    )
    return source, target


class _Experts(nn.Module):
    """The FusedMoE state ``_online_quant``, ``Mxfp4MoEMethod`` and the binder
    read, at Pro's TP8 widths (intermediate 640 a rank) over ``count``
    experts, holding the checkpoint's NVFP4 planes (``Nvfp4MoEMethod``'s
    shapes) until ``_online_quant`` runs."""

    def __init__(self, moe_mod, count, source, target, seed=0):
        super().__init__()
        from atom.quant_spec import NVFP4_DTYPE

        inter = DIMS.inter_valid
        g = torch.Generator().manual_seed(seed)
        u8 = lambda *s: torch.randint(0, 256, s, dtype=torch.uint8, generator=g)
        # positive E4M3 block scales, 0.25 .. 4
        e4 = lambda *s: (torch.rand(*s, generator=g) * 3.75 + 0.25).to(
            torch.float8_e4m3fn
        )
        p = lambda t: nn.Parameter(t, requires_grad=False)
        self.w13_weight = p(u8(count, 2 * inter, HIDDEN // 2))
        self.w13_weight_scale = p(e4(count, 2 * inter, HIDDEN // 16))
        self.w13_weight_scale_2 = p(torch.rand(count, 2, generator=g) + 0.5)
        self.w2_weight = p(u8(count, HIDDEN, inter // 2))
        self.w2_weight_scale = p(e4(count, HIDDEN, inter // 16))
        self.w2_weight_scale_2 = p(torch.rand(count, generator=g) + 0.5)
        self.register_parameter("w13_bias", None)
        self.register_parameter("w2_bias", None)
        self.top_k = 4
        self.num_fused_shared_experts = 0
        self.hidden_size = HIDDEN
        self.intermediate_size_per_partition = inter
        self.has_bias = False
        self.online_quant = True
        self.quant_config = SimpleNamespace(
            get_layer_quant_config=lambda *_a, use_online_quant=False, **_k: (
                target if use_online_quant else source
            )
        )
        self.layer_name = "model.layers.10.block_sparse_moe.experts"
        self.layer_quant_config = source
        self.source_is_nvfp4 = True
        self.params_dtype = NVFP4_DTYPE
        # the source method, with the pad MiniMaxM3MoE pins for the MX target
        self.quant_method = SimpleNamespace(
            hidden_pad=0, intermediate_pad=0, need_normalize_e4m3fn_to_e4m3fnuz=False
        )
        self.moe_config = SimpleNamespace(
            a_quant_dtype=None,
            mxfp4_fp8_activations=False,
            use_ep=False,
            in_dtype=torch.bfloat16,
            hidden_dim=HIDDEN,
        )
        self.moe_quant_params = {
            "num_experts": count,
            "hidden_size": HIDDEN,
            "intermediate_size_per_partition": inter,
            "params_dtype": NVFP4_DTYPE,
            "weight_loader": lambda *_a, **_k: None,
        }
        self.pad_align_override = None
        self._stream_online_quant = False
        self.local_num_experts = self.num_local_base_experts = count
        self.expert_layout = SimpleNamespace(routed_physical_per_rank=count)
        self.use_ep = False
        self.tp_size, self.tp_rank = 8, 3
        self._copy_quant_storage = moe_mod.FusedMoE._copy_quant_storage


def _quantize(weight, **_kwargs):
    """MXFP4 of a dequantized weight, in ``quant_weight_online``'s dtypes (the
    aiter HIP quantizer it calls needs a GPU; only the bytes' places matter)."""
    from aiter import dtypes

    from atom.mono.weights.formats import quantize_mxfp4

    q, s = quantize_mxfp4(weight)
    return q.view(dtypes.fp4x2), s.view(dtypes.fp8_e8m0)


def _expected(src, e, pad):
    """Expert e's logical planes as production must hold them at the loaded
    width ``pad``: (w13 bytes [2 pad, 3584], w13 codes [2 pad, 224], w2 bytes
    [7168, pad / 2], w2 codes [7168, pad / 32]), any pad zero bytes and code
    0."""
    from atom.quantization.quark.utils import dequantize_nvfp4

    inter = DIMS.inter_valid

    def mx(w, s, s2):
        q, c = _quantize(dequantize_nvfp4(w.contiguous(), s.contiguous(), s2))
        return q.view(torch.uint8), c.view(torch.uint8)

    w13 = torch.zeros(2 * pad, HIDDEN // 2, dtype=torch.uint8)
    c13 = torch.zeros(2 * pad, HIDDEN // 32, dtype=torch.uint8)
    for half in range(2):
        rows = slice(half * inter, (half + 1) * inter)
        q, c = mx(src["w13"][e, rows], src["s13"][e, rows], src["s13_2"][e, half])
        w13[half * pad : half * pad + inter] = q
        c13[half * pad : half * pad + inter] = c
    w2 = torch.zeros(HIDDEN, pad // 2, dtype=torch.uint8)
    c2 = torch.zeros(HIDDEN, pad // 32, dtype=torch.uint8)
    q, c = mx(src["w2"][e], src["s2"][e], src["s2_2"][e])
    w2[:, : inter // 2], c2[:, : inter // 32] = q, c
    return w13, c13, w2, c2


@pytest.fixture(scope="module", params=[PRO_PAD_ALIGN, None], ids=["pad128", "pad256"])
def produced(request):
    """Two experts converted and laid out by production at Pro's pad_align
    (128: the 640 unpadded) and at the method's default (256: 768): (the
    experts module, its NVFP4 source)."""
    with pytest.MonkeyPatch.context() as mp:
        moe_mod = _production(mp)
        mp.setattr(moe_mod, "quant_weight_online", _quantize)
        source, target = _specs()
        experts = _Experts(moe_mod, 2, source, target)
        # what FusedMoE(pad_align=...) sets (``MiniMaxM3MoE``)
        experts.pad_align_override = request.param
        src = {
            "w13": experts.w13_weight.data.clone(),
            "s13": experts.w13_weight_scale.data.clone(),
            "s13_2": experts.w13_weight_scale_2.data.clone(),
            "w2": experts.w2_weight.data.clone(),
            "s2": experts.w2_weight_scale.data.clone(),
            "s2_2": experts.w2_weight_scale_2.data.clone(),
        }
        moe_mod.FusedMoE._online_quant(experts)
        assert type(experts.quant_method) is moe_mod.Mxfp4MoEMethod
        experts.quant_method.process_weights_after_loading(experts)
        yield experts, src


def _bind_routed(experts, monkeypatch, count):
    monkeypatch.setattr(weights_mod, "EXPERTS", count)
    return weights_mod._routed(SimpleNamespace(experts=experts), DIMS, "experts")


def test_binds_the_planes_production_converts(produced, monkeypatch):
    """The four planes FusedMoE's NVFP4 -> MXFP4 conversion and gfx950 prep
    leave, bound in place at the width the loader padded to, read back
    (shared unshuffles) as the conversion wrote them: gate / up at rows 0 /
    inter_r, any pad (w2's K 640 .. 768 at the 256 pad) zero bytes and zero
    codes."""
    from atom.mono.weights.layer import (
        _unshuffle_linear_scale,
        _unshuffle_linear_weight,
    )

    experts, src = produced
    method = experts.quant_method
    pad = method.intermediate_size
    want_pad = 640 if experts.pad_align_override == PRO_PAD_ALIGN else 768
    assert (pad, method.intermediate_pad) == (want_pad, 0)
    w13, s13, w2, s2, planes = _bind_routed(experts, monkeypatch, 2)
    assert planes.inter_r == pad
    for got, param in (
        (w13, experts.w13_weight),
        (s13, experts.w13_weight_scale),
        (w2, experts.w2_weight),
        (s2, experts.w2_weight_scale),
    ):
        assert got.data_ptr() == param.data_ptr() and got.dtype == torch.uint8
    w13, w2 = w13.view(torch.uint8), w2.view(torch.uint8)
    w13.is_shuffled = w2.is_shuffled = True
    got13 = _unshuffle_linear_weight(w13)
    got2 = _unshuffle_linear_weight(w2)
    gotc13 = _unshuffle_linear_scale(
        s13, experts=2, rows=planes.w13_rows, groups=HIDDEN // 32
    )
    gotc2 = _unshuffle_linear_scale(s2, experts=2, rows=HIDDEN, groups=pad // 32)
    for e in range(2):
        want13, wantc13, want2, wantc2 = _expected(src, e, pad)
        assert torch.equal(got13[e], want13)
        assert torch.equal(gotc13[e], wantc13)
        assert torch.equal(got2[e], want2)
        assert torch.equal(gotc2[e], wantc2)


def test_the_kernels_addressing_reads_the_production_planes(produced, monkeypatch):
    """``mono.device.mx``'s ``fp4_tile_load`` dword arithmetic (lane l: row
    l % 16 of a 16-row group, 32 k from 32 (l // 16) of a 128-k step) and
    ``scale_index`` on the bound planes give the conversion's bytes, for gate,
    up and down rows and the pad."""
    pytest.importorskip("flydsl")
    from atom.mono.device.mx import scale_index

    experts, src = produced
    w13, s13, w2, s2, routed = _bind_routed(experts, monkeypatch, 2)
    pad = routed.inter_r
    planes = (
        # (plane, scale, rows, K, expected bytes, expected codes)
        (w13, s13, 2 * pad, HIDDEN, 0, 1, routed.w13_scale_cols),
        (w2, s2, HIDDEN, pad, 2, 3, routed.w2_scale_cols),
    )
    e = 1
    want = _expected(src, e, pad)
    for w, s, rows, k, wi, ci, cols in planes:
        flat = w.reshape(-1)
        base = e * rows * k // 2
        for rg in (0, 39, 40, 41, 47, 48, 79, 80, 87, rows // 16 - 1):
            if rg >= rows // 16:
                continue
            for st in (0, k // 128 - 1):
                for lane in (0, 5, 16, 63):
                    dw = rg * (2 * k) + st * 256 + lane * 4
                    got = flat[base + dw * 4 : base + dw * 4 + 16]
                    col = (st * 128 + (lane // 16) * 32) // 2
                    exp = want[wi][16 * rg + lane % 16, col : col + 16]
                    assert torch.equal(got, exp), (rows, rg, st, lane)
        groups = k // 32
        for r in (0, 17, 639, 640, 767, 768, 1407, 1408, rows - 1):
            if r >= rows:
                continue
            for c in (0, 5, groups - 4, groups - 1):
                si = scale_index(e * rows + r, c, cols)
                assert s.reshape(-1)[si] == want[ci][r, c], (rows, r, c)


def test_binds_the_full_layer_production_lays_out(monkeypatch):
    """At Pro's full 384 experts (on the meta device: 3 GB otherwise), the
    real ``Mxfp4MoEMethod`` sizes and lays out exactly the planes
    ``bind_layer`` takes."""
    moe_mod = _production(monkeypatch)
    from aiter import dtypes

    _, target = _specs()
    layer = _layer()
    experts = _Experts(moe_mod, 1, *_specs())
    method = moe_mod.Mxfp4MoEMethod(target, experts.moe_config)
    method.pad_align = PRO_PAD_ALIGN  # FusedMoE's pad_align_override for Pro
    with torch.device("meta"):
        method.create_weights(
            experts,
            **{
                **experts.moe_quant_params,
                "num_experts": EXPERTS,
                "params_dtype": dtypes.fp4x2,
            },
        )
    method.intermediate_pad = 0  # as MiniMaxM3MoE pins it
    experts.quant_method = method
    experts.source_is_nvfp4 = False
    method.process_weights_after_loading(experts)
    layer.block_sparse_moe.experts = experts
    b = bind_layer(layer, DIMS, 5)
    # 640 a rank, no pad; w2's 20 scale groups shuffled as 24
    assert b.planes.inter_r == 640 and b.planes.w2_scale_cols == 24
    assert b.w13.shape == (EXPERTS, 1280, HIDDEN // 2)
    assert b.s13.shape == (EXPERTS * 1280, HIDDEN // 32)
    assert b.w2.shape == (EXPERTS, HIDDEN, 320)
    assert b.s2.shape == (EXPERTS * HIDDEN, 24)
