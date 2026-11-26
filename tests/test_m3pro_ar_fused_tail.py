# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro ``ProVariants.ar_fused_rows`` and ``moe_tail_ar``.

``fused_allreduce_gemma_rms_norm_mxfp8`` sends calls of at most ``fused_rows`` rows to
aiter's one-launch Lamport all-reduce + norm + MXFP8, the addend with them; where that op
declines, or above the rows, the addend is added first and the call keeps its route. A
MoE layer under ``moe_tail_ar`` hands its shared-expert output to the next layer's input
all-reduce instead of adding it, which gives the same values; the last layer adds its own.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiter_stub import _ExternalModuleLoader, atom_without_aiter

TP = 8


class _FlydslFinder(importlib.abc.MetaPathFinder):
    """FlyDSL stand-ins (any attribute a MagicMock) where it is not installed: the
    MiniMax-M3 model module's import chain reaches its mono kernels."""

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


def _rows(m, n=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(m, n, generator=g).to(torch.bfloat16)


def _ar_ref(x, residual):
    """What the fake all-reduces compute: TP equal partial sums, then the residual."""
    return (x.float() * TP + residual.float()).to(torch.bfloat16)


@pytest.fixture
def tp_ar(load, monkeypatch):
    """layernorm at TP 8 over fake aiter ops that record what they were given."""
    layernorm = load("atom.model_ops.layernorm")
    calls = []

    def lamport(x, residual, w, eps, gemma_norm, mxfp8, addend):
        calls.append(("lamport", x, addend))
        if x.shape[0] in tp_ar.declines:
            return None
        h = _ar_ref(x if addend is None else x + addend, residual)
        return h, h, "q", "s"

    def mxfp8_quant(x, residual, w, eps, gemma_norm):
        calls.append(("fused", x, None))
        h = _ar_ref(x, residual)
        return h, h, "q", "s"

    ops = SimpleNamespace(
        tensor_model_parallel_fused_allreduce_rmsnorm_lamport=lamport,
        tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant=mxfp8_quant,
    )
    import aiter.dist

    monkeypatch.setattr(aiter.dist, "communication_op", ops, raising=False)
    monkeypatch.setitem(sys.modules, "aiter.dist.communication_op", ops)
    monkeypatch.setattr(layernorm, "get_tensor_model_parallel_world_size", lambda: TP)
    monkeypatch.setattr(layernorm, "_lamport_ar_on", lambda: False)
    tp_ar.declines = set()
    tp_ar.calls = calls
    tp_ar.layernorm = layernorm
    return tp_ar


_NORM = SimpleNamespace(
    weight=torch.zeros(64),
    variance_epsilon=1e-6,
    use_fused_quant=False,
    fused_allreduce=False,
)


@pytest.mark.parametrize("fused_rows", [0, 16])
def test_tp1_adds_the_addend_before_the_norm(load, monkeypatch, fused_rows):
    layernorm = load("atom.model_ops.layernorm")
    seen = []

    def norm_mx(x, residual, norm, write_bf16=True, q_out=None, q_scale_out=None):
        seen.append(x)
        return x, "q", "s", residual

    monkeypatch.setattr(layernorm, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(layernorm, "gemma_rms_norm_mxfp8", norm_mx)
    x, res, add = _rows(4, seed=1), _rows(4, seed=2), _rows(4, seed=3)
    layernorm.fused_allreduce_gemma_rms_norm_mxfp8(
        x, res, _NORM, 0, fused_rows, addend=add
    )
    assert len(seen) == 1 and torch.equal(seen[0], x + add)


@pytest.mark.parametrize("addend", [False, True])
def test_rows_up_to_fused_rows_take_the_lamport_op_with_the_addend(tp_ar, addend):
    x, res = _rows(16, seed=1), _rows(16, seed=2)
    add = _rows(16, seed=3) if addend else None
    _, q, s, res_out = tp_ar.layernorm.fused_allreduce_gemma_rms_norm_mxfp8(
        x, res, _NORM, 0, 16, addend=add
    )
    (kind, got_x, got_add), *rest = tp_ar.calls
    assert kind == "lamport" and not rest
    assert torch.equal(got_x, x) and got_add is add
    assert (q, s) == ("q", "s")
    assert torch.equal(res_out, _ar_ref(x if add is None else x + add, res))


@pytest.mark.parametrize("rows, declines", [(17, False), (16, True)])
def test_other_calls_add_first_and_keep_their_route(tp_ar, rows, declines):
    if declines:
        tp_ar.declines.add(rows)
    x, res, add = _rows(rows, seed=1), _rows(rows, seed=2), _rows(rows, seed=3)
    *_, res_out = tp_ar.layernorm.fused_allreduce_gemma_rms_norm_mxfp8(
        x, res, _NORM, 0, 16, addend=add
    )
    kinds = [c[0] for c in tp_ar.calls]
    assert kinds == (["lamport", "fused"] if declines else ["fused"])
    # the route below the Lamport op gets the bf16 sum, as the MoE's own add makes it
    assert torch.equal(tp_ar.calls[-1][1], x + add)
    assert torch.equal(res_out, _ar_ref(x + add, res))


def test_fused_rows_write_the_callers_mxfp8_rows(tp_ar, monkeypatch):
    # the decode front's ring slot (q_out / q_scale_out) holds the one-launch op's bytes,
    # so the front GEMM reads them where the input norm was asked to write
    x, res = _rows(4, seed=1), _rows(4, seed=2)
    q = torch.randint(0, 256, (4, 64), dtype=torch.uint8)
    q_scale = torch.randint(0, 256, (4, 2), dtype=torch.uint8)

    def one_launch(hidden_states, residual, norm, addend):
        return hidden_states, q, q_scale, residual

    monkeypatch.setattr(tp_ar.layernorm, "_lamport_ar_on", lambda: True)
    monkeypatch.setattr(tp_ar.layernorm, "_lamport_fused_ar_rms_norm_mxfp8", one_launch)
    q_out = torch.zeros(4, 64, dtype=torch.uint8)
    q_scale_out = torch.zeros(4, 2, dtype=torch.uint8)
    _, got_q, got_s, _ = tp_ar.layernorm.fused_allreduce_gemma_rms_norm_mxfp8(
        x, res, _NORM, 16, 16, q_out=q_out, q_scale_out=q_scale_out
    )
    assert got_q is q_out and got_s is q_scale_out
    assert torch.equal(q_out, q) and torch.equal(q_scale_out, q_scale)


def test_fused_rows_off_never_calls_the_lamport_op(tp_ar):
    x, res = _rows(4, seed=1), _rows(4, seed=2)
    tp_ar.layernorm.fused_allreduce_gemma_rms_norm_mxfp8(x, res, _NORM, 0)
    assert [c[0] for c in tp_ar.calls] == ["fused"]


@pytest.mark.parametrize("rows", [4, 17])
def test_reduced_rows_take_the_tp1_norm_into_the_callers_rows(
    tp_ar, monkeypatch, rows
):
    # TP MegaMoE all-reduced the MoE output in-kernel: only the add + norm run
    seen = []

    def norm_mx(x, residual, norm, write_bf16=True, q_out=None, q_scale_out=None):
        seen.append((x, q_out, q_scale_out))
        return x, q_out, q_scale_out, residual

    monkeypatch.setattr(tp_ar.layernorm, "_lamport_ar_on", lambda: True)
    monkeypatch.setattr(tp_ar.layernorm, "gemma_rms_norm_mxfp8", norm_mx)
    layernorm = tp_ar.layernorm
    assert layernorm.fused_allreduce_gemma_rms_norm_mxfp8_writes_out(
        rows, _NORM, 16, reduced=True
    )
    x, res = _rows(rows, seed=1), _rows(rows, seed=2)
    q_out = torch.zeros(rows, 64, dtype=torch.uint8)
    q_scale_out = torch.zeros(rows, 2, dtype=torch.uint8)
    layernorm.fused_allreduce_gemma_rms_norm_mxfp8(
        x, res, _NORM, 16, 16, q_out=q_out, q_scale_out=q_scale_out, reduced=True
    )
    assert tp_ar.calls == []
    assert len(seen) == 1
    got_x, got_q, got_s = seen[0]
    assert got_x is x and got_q is q_out and got_s is q_scale_out


class _Shared(nn.Module):
    def forward(self, x, x_scale=None, quant_down=False):
        return x * 0.37 + 1.0


class _Experts(nn.Module):
    fold_routed_scaling_factor = True

    def forward(self, hidden_states, router_logits):
        return hidden_states * 1.7


def _moe(load):
    m3_model = load("atom.models.minimax_m3.model")
    moe = m3_model.MiniMaxM3MoE.__new__(m3_model.MiniMaxM3MoE)
    nn.Module.__init__(moe)
    moe.gate = nn.Identity()
    moe.experts = _Experts()
    moe.routed_scaling_factor = 2.0
    moe.fuse_shared_experts = False
    moe.shared_experts = _Shared()
    return moe


def test_moe_hands_over_routed_and_shared_which_sum_to_its_output(load):
    moe = _moe(load)
    x = _rows(5, seed=4)
    routed, shared = moe(x, defer_shared=True)
    assert torch.equal(routed + shared, moe(x))


# --- the decoder layers -------------------------------------------------------------


class _Attn(nn.Module):
    # bf16 projections: no one front GEMM over the ring (front stays None)
    qkv_proj = SimpleNamespace(runs_bf16=True)

    def forward(
        self,
        positions,
        hidden_states,
        hidden_states_scale,
        index_src,
        diag_srcs,
        front=None,
    ):
        assert front is None
        return hidden_states * 0.5


def _layers(load, monkeypatch, num_layers=3, end_layer=None):
    pro = load("atom.models.minimax_m3.pro")
    pro_attn = load("atom.model_ops.minimax_m3.pro_attn")
    seen = []

    def ar_norm_mx(
        x,
        residual,
        norm,
        lamport_rows=0,
        fused_rows=0,
        addend=None,
        q_out=None,
        q_scale_out=None,
        reduced=False,
    ):
        seen.append((norm.layer, addend))
        h = _ar_ref(x if addend is None else x + addend, residual)
        return h, h, torch.ones(h.shape[0], 2), h

    def norm_mx(x, residual, norm, write_bf16=True, q_out=None, q_scale_out=None):
        return None, x, "s", None

    monkeypatch.setattr(pro, "fused_allreduce_gemma_rms_norm_mxfp8", ar_norm_mx)
    monkeypatch.setattr(pro, "gemma_rms_norm_mxfp8", norm_mx)
    monkeypatch.setattr(pro, "_aiter_mx_norm", lambda: True)
    monkeypatch.setattr(pro, "_native_a8", lambda *linears: True)
    cfg = SimpleNamespace(sparse_attention_config={}, num_hidden_layers=num_layers)
    state = pro.ProLookaheadState(cfg)
    if end_layer is not None:
        state.end_layer = end_layer
    layers = []
    for i in range(num_layers):
        layer = pro.MiniMaxM3ProDecoderLayer.__new__(pro.MiniMaxM3ProDecoderLayer)
        nn.Module.__init__(layer)
        layer._pro_state = state
        layer.layer_num = i
        layer.self_attn = _Attn()
        layer.input_layernorm = SimpleNamespace(layer=(i, "in"))
        layer.post_attention_layernorm = SimpleNamespace(layer=(i, "post"))
        layer.is_moe_layer = True
        layer.block_sparse_moe = _moe(load)
        layer.block_sparse_moe.shared_experts.gate_up_proj = nn.Identity()
        layers.append(layer)

    def run(**variants):
        monkeypatch.setattr(pro_attn, "_variants", pro_attn.ProVariants(**variants))
        seen.clear()
        h, residual = _rows(6, seed=9), None
        for layer in layers[: state.end_layer]:
            h, residual = layer(None, h, residual)
        return h, residual, list(seen)

    run.layers, run.state = layers, state
    return run


def test_the_next_layers_input_all_reduce_adds_the_shared_output(load, monkeypatch):
    run = _layers(load, monkeypatch)
    want_h, want_res, plain = run(ar_fused_rows=256)
    got_h, got_res, deferred = run(ar_fused_rows=256, moe_tail_ar=True)
    assert all(add is None for _, add in plain)
    handed = {norm: add for norm, add in deferred if add is not None}
    # layers 1 and 2 get layer 0's and 1's shared output; the last layer adds its own
    assert set(handed) == {(1, "in"), (2, "in")}
    assert torch.equal(got_res, want_res) and torch.equal(got_h, want_h)


def test_a_pipeline_stages_last_layer_adds_its_own(load, monkeypatch):
    """Layers 0-1 of 3 on this stage: layer 1's shared output never reaches layer 2's
    all-reduce here, so it is added on this stage."""
    run = _layers(load, monkeypatch, end_layer=2)
    want_h, want_res, _ = run(ar_fused_rows=256)
    got_h, got_res, deferred = run(ar_fused_rows=256, moe_tail_ar=True)
    assert {norm for norm, add in deferred if add is not None} == {(1, "in")}
    assert torch.equal(got_res, want_res) and torch.equal(got_h, want_h)


def test_mono_check_modes_reference_segment_sums_and_leaves_no_tail(load, monkeypatch):
    """``_segment``, production's segment that mono's check mode runs as the
    kernel's reference, returns the summed FFN partial under ``moe_tail_ar`` and
    leaves no tail: the forward goes on from the kernel's summed output, so a tail
    left here would be added again by the next layer's input all-reduce."""
    pro_attn = load("atom.model_ops.minimax_m3.pro_attn")
    run = _layers(load, monkeypatch)
    layer, state = run.layers[0], run.state
    o, residual = _rows(6, seed=3), _rows(6, seed=5)

    def segment(**variants):
        v = pro_attn.ProVariants(**variants)
        monkeypatch.setattr(pro_attn, "_variants", v)
        state.moe_tail = None
        ffn, *_ = layer._segment(o, residual.clone(), v, True, True)
        return ffn, state.take_moe_tail()

    want, _ = segment()
    got, tail = segment(moe_tail_ar=True)
    assert tail is None
    assert torch.equal(got, want)
