# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro ``ProVariants.moe_decode`` "day0": the routed experts of a MoE call of
at most 64 rows on aiter's ``minimax_m3_moe_experts_decode``.

``FusedMoE.forward_impl`` hands a call to the layer's ``pro_decode_experts`` when that
says it runs it, and keeps its own path otherwise. ``ProDecodeExperts`` takes a call only
on a TP layer with the plain sigmoid routing folded into the weights and separated gate/up
MXFP4 weights on the FlyDSL path, routes every row with ``topk_gating`` (no row count: the
sort takes no -1 ids) and hands the layer's weights over as they lie. The variant is off by
default, and an aiter without the op leaves the layer on ``fused_moe``.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiter_stub import _ExternalModuleLoader, atom_without_aiter


class _FlydslFinder(importlib.abc.MetaPathFinder):
    """FlyDSL stand-ins where it is not installed (the MoE module's import chain)."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "flydsl" or fullname.startswith("flydsl."):
            return importlib.util.spec_from_loader(
                fullname, _ExternalModuleLoader(), is_package=True
            )
        return None


@pytest.fixture
def load():
    finder = None
    if importlib.util.find_spec("flydsl") is None:
        finder = _FlydslFinder()
        sys.meta_path.insert(0, finder)
    try:
        with atom_without_aiter() as load:
            yield load
    finally:
        if finder is not None:
            sys.meta_path.remove(finder)
            for name in [n for n in sys.modules if n.split(".")[0] == "flydsl"]:
                sys.modules.pop(name, None)


# --- FusedMoE.forward_impl ----------------------------------------------------------


class _Decode:
    def __init__(self, runs):
        self._runs = runs
        self.calls = []

    def runs(self, layer, hidden_states):
        return self._runs

    def forward(self, layer, hidden_states, router_logits):
        self.calls.append((layer, hidden_states, router_logits))
        return hidden_states * 3


class _FellThrough(Exception):
    pass


def _fused_moe(load, monkeypatch, decode):
    moe = load("atom.model_ops.moe")

    def dp_group():
        raise _FellThrough

    monkeypatch.setattr(moe, "get_dp_group", dp_group)
    layer = object.__new__(moe.FusedMoE)
    torch.nn.Module.__init__(layer)
    layer.quant_method = object()
    layer.sp_mega_moe = None
    layer.pro_decode_experts = decode
    return layer


def test_forward_impl_hands_a_call_the_decode_experts_run(load, monkeypatch):
    decode = _Decode(runs=True)
    layer = _fused_moe(load, monkeypatch, decode)
    x, logits = torch.ones(4, 8), torch.zeros(4, 3)
    out = layer.forward_impl(x, logits)
    assert torch.equal(out, x * 3)
    assert len(decode.calls) == 1 and decode.calls[0][2] is logits


@pytest.mark.parametrize("why", ["declined", "shared_add", "absent"])
def test_forward_impl_keeps_its_path_otherwise(load, monkeypatch, why):
    decode = None if why == "absent" else _Decode(runs=why != "declined")
    layer = _fused_moe(load, monkeypatch, decode)
    shared = torch.zeros(4, 8) if why == "shared_add" else None
    with pytest.raises(_FellThrough):
        layer.forward_impl(torch.ones(4, 8), torch.zeros(4, 3), shared_add=shared)
    assert decode is None or not decode.calls


# --- ProDecodeExperts ---------------------------------------------------------------


def _pro_layer(load, **over):
    """A Pro routed-expert layer as ``ProDecodeExperts`` reads it."""
    from aiter import ActivationType

    moe = load("atom.model_ops.moe")
    method = moe.Mxfp4MoEMethod.__new__(moe.Mxfp4MoEMethod)
    method.fused_experts = None
    method.is_guinterleave = False
    method.use_triton = False
    method.fp8_activations = False
    fields = {
        "quant_method": method,
        "moe_parallel_config": SimpleNamespace(
            use_ep=False, dp_size=1, dp_logical_ratio=1
        ),
        "reduce_results": False,
        "top_k": 4,
        "num_fused_shared_experts": 0,
        "scoring_func": "sigmoid",
        "use_grouped_topk": False,
        "custom_routing_function": None,
        "renormalize": True,
        "fold_routed_scaling_factor": True,
        "routed_scaling_factor": 2.0,
        "activation": ActivationType.Swiglu,
        "swiglu_limit": 7.0,
        "w13_bias": None,
        "w2_bias": None,
        "e_score_correction_bias": torch.zeros(384),
        "w13_weight": torch.zeros(384, 1280, 3584, dtype=torch.uint8),
        "w13_weight_scale": torch.zeros(384 * 1280, 224, dtype=torch.uint8),
        "w2_weight": torch.zeros(384, 7168, 320, dtype=torch.uint8),
        "w2_weight_scale": torch.zeros(384 * 7168, 24, dtype=torch.uint8),
    }
    fields.update(over)
    for name, value in over.items():
        if name in ("is_guinterleave", "use_triton", "fp8_activations"):
            setattr(method, name, value)
    return SimpleNamespace(**fields)


def _decode_experts(load, supported=True):
    mod = load("atom.model_ops.fused_moe.pro_experts_decode")
    calls = {}

    def topk_gating(w, ids, logits, bias, renorm, scale, **kw):
        calls["gating"] = (logits, bias, renorm, scale, kw)
        ids.copy_(torch.arange(4, dtype=torch.int32).repeat(ids.shape[0], 1))
        w.fill_(0.5)

    def experts(h, w, ids, w13, s13, w2, s2):
        calls["experts"] = (h, w.clone(), ids.clone(), w13, s13, w2, s2)
        return h + 1

    return (
        mod.ProDecodeExperts(experts, lambda h, w13, w2: supported, topk_gating),
        calls,
    )


def test_runs_a_pro_tp_layer_the_op_supports(load):
    decode, _ = _decode_experts(load)
    assert decode.runs(_pro_layer(load), torch.zeros(8, 7168, dtype=torch.bfloat16))
    declined, _ = _decode_experts(load, supported=False)
    assert not declined.runs(
        _pro_layer(load), torch.zeros(65, 7168, dtype=torch.bfloat16)
    )


@pytest.mark.parametrize(
    "over",
    [
        {"is_guinterleave": True},
        {"use_triton": True},
        {"fp8_activations": True},
        {"fold_routed_scaling_factor": False},
        {"num_fused_shared_experts": 1},
        {"reduce_results": True},
        {"use_grouped_topk": True},
        {"scoring_func": "softmax"},
        {"swiglu_limit": 10.0},
        {"w13_bias": torch.zeros(384, 1280)},
        {
            "moe_parallel_config": SimpleNamespace(
                use_ep=True, dp_size=1, dp_logical_ratio=1
            )
        },
        {
            "moe_parallel_config": SimpleNamespace(
                use_ep=False, dp_size=2, dp_logical_ratio=1
            )
        },
    ],
    ids=lambda o: next(iter(o)),
)
def test_other_layers_stay_on_fused_moe(load, over):
    decode, _ = _decode_experts(load)
    layer = _pro_layer(load, **over)
    assert not decode.runs(layer, torch.zeros(8, 7168, dtype=torch.bfloat16))


def test_forward_routes_every_row_and_hands_the_weights_over(load):
    decode, calls = _decode_experts(load)
    layer = _pro_layer(load)
    x = torch.randn(6, 7168).to(torch.bfloat16)
    logits = torch.randn(6, 384).to(torch.bfloat16)
    out = decode.forward(layer, x, logits)
    assert torch.equal(out, x + 1)
    got_logits, bias, renorm, scale, kw = calls["gating"]
    assert got_logits is logits and bias is layer.e_score_correction_bias
    assert renorm is True and scale == 2.0
    assert kw == {"score_func": "sigmoid"}  # no num_valid_rows: padding rows routed too
    h, w, ids, w13, s13, w2, s2 = calls["experts"]
    assert h is x and ids.shape == (6, 4) and ids.dtype == torch.int32
    assert w.dtype == torch.float32 and torch.all(w == 0.5)
    assert (w13, s13, w2, s2) == (
        layer.w13_weight,
        layer.w13_weight_scale,
        layer.w2_weight,
        layer.w2_weight_scale,
    )


# --- the variant ---------------------------------------------------------------------


def _from_variants(load, monkeypatch, value, flydsl=None):
    mod = load("atom.model_ops.fused_moe.pro_experts_decode")
    pro_attn = load("atom.model_ops.minimax_m3.pro_attn")
    # the variants as a fresh process has them, put back after the test
    monkeypatch.setattr(pro_attn, "_variants", pro_attn.ProVariants())
    monkeypatch.setattr(pro_attn, "_variants_src", None)
    if value is None:
        monkeypatch.delenv("ATOM_M3PRO_VARIANTS", raising=False)
    else:
        monkeypatch.setenv("ATOM_M3PRO_VARIANTS", f'{{"moe_decode": "{value}"}}')
    if flydsl is not None:
        monkeypatch.setitem(sys.modules, "aiter.ops.flydsl", flydsl)
    return mod.ProDecodeExperts.from_variants()


def _flydsl(with_op):
    m = types.ModuleType("aiter.ops.flydsl")
    if with_op:
        m.minimax_m3_moe_experts_decode = lambda *a: None
        m.minimax_m3_moe_experts_decode_supported = lambda *a: True
    return m


def test_off_by_default(load, monkeypatch):
    assert (
        load("atom.model_ops.minimax_m3.pro_attn").ProVariants().moe_decode == "aiter"
    )
    assert _from_variants(load, monkeypatch, None, _flydsl(True)) is None
    assert _from_variants(load, monkeypatch, "aiter", _flydsl(True)) is None


def test_day0_builds_where_aiter_has_the_op(load, monkeypatch):
    mod = load("atom.model_ops.fused_moe.pro_experts_decode")
    got = _from_variants(load, monkeypatch, "day0", _flydsl(True))
    assert isinstance(got, mod.ProDecodeExperts)
    assert _from_variants(load, monkeypatch, "day0", _flydsl(False)) is None


def test_unknown_choice_is_refused(load, monkeypatch):
    with pytest.raises(ValueError, match="moe_decode"):
        _from_variants(load, monkeypatch, "fast", _flydsl(True))
