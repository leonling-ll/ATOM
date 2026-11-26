# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro ``ProVariants.moe_glue_fold`` on the ``moe_route_glue`` path.

A MoE layer whose post-attention all-reduce runs as the route op hands its shared-expert
output to the next layer's input all-reduce (the last layer of a stage adds its own), and
has the route op write the lookahead norm's MXFP8 rows where ``put`` would have run that
norm: the same rows (the reader's ring rows up to ``FRONT_ROWS``, new tensors above), the
same layer outputs.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aiter_stub import atom_without_aiter
from test_m3pro_ar_fused_tail import _ar_ref, _FlydslFinder, _rows

H = 64


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


def _mx(x, norm):
    """A stand-in MXFP8 norm: E4M3 bytes and E8M0 codes that depend on the row and the
    norm."""
    q = (x.float() * norm.gain).to(torch.float8_e4m3fn)
    s = (x.float().view(x.shape[0], -1, 32).abs().amax(-1) * 7).to(torch.uint8)
    return q, s


def _into(t, dst):
    if dst is None:
        return t
    dst.view(torch.uint8).copy_(t.view(torch.uint8))
    return dst


class _Attn(nn.Module):
    qkv_proj = SimpleNamespace(runs_bf16=True)

    def __init__(self, seen):
        super().__init__()
        self.seen = seen

    def forward(
        self, positions, hidden_states, hidden_states_scale, index_src, diag_srcs, front
    ):
        if index_src is not None:
            q, s = index_src
            self.seen.append((q.view(torch.uint8).clone(), s.view(torch.uint8).clone()))
        # layer 0's input norm is MXFP8 (the stand-in's bytes)
        return (hidden_states.float() * 0.5).to(torch.bfloat16)


class _Experts(nn.Module):
    fold_routed_scaling_factor = True

    def forward_routed(self, x, w, ids, out):
        return out.copy_(x * 1.7)


class _Shared(nn.Module):
    gate_up_proj = nn.Identity()

    def forward(self, x, x_scale=None, quant_down=False):
        return x * 0.37 + 1.0


def _layers(load, monkeypatch, num_layers=4):
    pro = load("atom.models.minimax_m3.pro")
    pro_attn = load("atom.model_ops.minimax_m3.pro_attn")
    m3_model = load("atom.models.minimax_m3.model")
    glue = pro.moe_route_glue
    log = SimpleNamespace(ar=[], route=[], norm=[], index=[])

    def ar_norm_mx(x, residual, norm, lamport_rows=0, fused_rows=0, addend=None, **_):
        log.ar.append((norm.layer, addend))
        h = _ar_ref(x if addend is None else x + addend, residual)
        return h, h, torch.ones(h.shape[0], 2), h

    def norm_mx(x, residual, norm, write_bf16=True, q_out=None, q_scale_out=None):
        log.norm.append(norm.layer)
        q, s = _mx(x, norm)
        return None, _into(q, q_out), _into(s, q_scale_out), None

    def route(x, residual, norm, moe, lookahead=None):
        log.route.append((norm.layer, lookahead))
        res = _ar_ref(x, residual)
        normed = (res.float() * 0.5).to(torch.bfloat16)
        r = {
            "residual_out": res,
            "normed": normed,
            "x_q": normed,
            "x_scale": None,
            "topk_weights": None,
            "topk_ids": None,
            "moe_buf": torch.zeros_like(res),
        }
        if lookahead is not None:
            n, q, s = lookahead
            la_q, la_s = _mx(res, n)
            r["la_q"], r["la_scale"] = _into(la_q, q), _into(la_s, s).view(torch.uint8)
        return r

    monkeypatch.setattr(pro, "fused_allreduce_gemma_rms_norm_mxfp8", ar_norm_mx)
    monkeypatch.setattr(pro, "gemma_rms_norm_mxfp8", norm_mx)
    monkeypatch.setattr(pro, "_aiter_mx_norm", lambda: True)
    monkeypatch.setattr(pro, "_native_a8", lambda *linears: True)
    monkeypatch.setattr(glue, "route", route)
    monkeypatch.setattr(glue, "applies", lambda moe, rows: True)
    monkeypatch.setattr(glue, "presort", lambda r, moe: None)
    monkeypatch.setattr(glue, "takes_lookahead", lambda: True)
    cfg = SimpleNamespace(
        sparse_attention_config={
            "sparse_indexer_lookahead_num_layers": 2,
            "sparse_indexer_lookahead_start_layer": 2,
        },
        num_hidden_layers=num_layers,
    )
    state = pro.ProLookaheadState(cfg)
    layers = []
    for i in range(num_layers):
        layer = pro.MiniMaxM3ProDecoderLayer.__new__(pro.MiniMaxM3ProDecoderLayer)
        nn.Module.__init__(layer)
        layer._pro_state = state
        layer.layer_num = i
        layer.self_attn = _Attn(log.index)
        layer.input_layernorm = SimpleNamespace(layer=(i, "in"), gain=1.0 + i / 8)
        layer.post_attention_layernorm = SimpleNamespace(layer=(i, "post"))
        layer.is_moe_layer = True
        moe = m3_model.MiniMaxM3MoE.__new__(m3_model.MiniMaxM3MoE)
        nn.Module.__init__(moe)
        moe.experts = _Experts()
        moe.routed_scaling_factor = 2.0
        moe.shared_experts = _Shared()
        layer.block_sparse_moe = moe
        state.norms[i] = layer.input_layernorm
        state.index_linears[i] = (layer.self_attn.qkv_proj,)
        layers.append(layer)

    def run(rows, **variants):
        monkeypatch.setattr(
            pro_attn,
            "_variants",
            pro_attn.ProVariants(moe_route_glue=True, ar_fused_rows=256, **variants),
        )
        for v in vars(log).values():
            v.clear()
        h, residual = _rows(rows, n=H, seed=9), None
        for layer in layers:
            h, residual = layer(None, h, residual)
        return h, residual

    return run, log, glue


@pytest.mark.parametrize("rows", [6, 20])  # the reader's ring rows; new tensors
def test_the_route_op_writes_the_lookahead_rows_and_the_next_ar_adds_the_shared(
    load, monkeypatch, rows
):
    run, log, _ = _layers(load, monkeypatch)
    want_h, want_res = run(rows)
    want_index = list(log.index)
    # today: the lookahead norm runs for readers 2 and 3, every AR gets no addend
    assert {(2, "in"), (3, "in")} <= set(log.norm)
    assert all(add is None for _, add in log.ar)

    got_h, got_res = run(rows, moe_glue_fold=True)
    assert torch.equal(got_h, want_h) and torch.equal(got_res, want_res)
    assert len(log.index) == len(want_index) == 2
    for (q, s), (wq, ws) in zip(log.index, want_index):
        assert torch.equal(q, wq) and torch.equal(s, ws)
    assert not {(2, "in"), (3, "in")} & set(log.norm)  # no lookahead norm launch
    asked = {norm: la is not None for norm, la in log.route}
    # readers 2 and 3 exist; layers 2 and 3 have none
    assert asked == {
        (0, "post"): True,
        (1, "post"): True,
        (2, "post"): False,
        (3, "post"): False,
    }
    # layers 1-3 get the shared output of the layer before; the last adds its own
    assert {norm for norm, add in log.ar if add is not None} == {
        (1, "in"),
        (2, "in"),
        (3, "in"),
    }


@pytest.mark.parametrize("rows, folds", [(6, False), (16, False), (20, True)])
def test_ar_fused_min_rows_keeps_todays_launches_at_or_below_it(
    load, monkeypatch, rows, folds
):
    """ar_fused_min_rows 16 (mono's widths): steps of at most 16 rows keep the lookahead
    norm and the shared add; wider ones fold both, the same bytes either way."""
    run, log, _ = _layers(load, monkeypatch)
    want_h, want_res = run(rows)
    got_h, got_res = run(rows, moe_glue_fold=True, ar_fused_min_rows=16)
    assert torch.equal(got_h, want_h) and torch.equal(got_res, want_res)
    norm_ran = {(2, "in"), (3, "in")} <= set(log.norm)
    added = sum(add is not None for _, add in log.ar)
    assert (norm_ran, added) == ((False, 3) if folds else (True, 0))


@pytest.mark.parametrize("rows, folds", [(20, False), (24, False), (30, True)])
def test_the_bound_follows_monos_latched_rows(load, monkeypatch, rows, folds):
    """mono latched at 24 rows (a launch's mono_ffn_rows) raises ar_fused_min_rows 16 to 24:
    the widths mono serves keep today's launches whatever the knob says."""
    run, log, _ = _layers(load, monkeypatch)
    want_h, want_res = run(rows)
    pro = load("atom.models.minimax_m3.pro")
    mono = SimpleNamespace(rows=24, take=lambda layer, rows: None)
    monkeypatch.setattr(pro.MiniMaxM3ProDecoderLayer, "_mono", mono)
    got_h, got_res = run(rows, moe_glue_fold=True, ar_fused_min_rows=16)
    assert torch.equal(got_h, want_h) and torch.equal(got_res, want_res)
    norm_ran = {(2, "in"), (3, "in")} <= set(log.norm)
    added = sum(add is not None for _, add in log.ar)
    assert (norm_ran, added) == ((False, 3) if folds else (True, 0))


def test_an_aiter_without_the_lookahead_keeps_the_norm_and_still_defers(
    load, monkeypatch
):
    run, log, glue = _layers(load, monkeypatch)
    want_h, want_res = run(6)
    want_index = list(log.index)
    monkeypatch.setattr(glue, "takes_lookahead", lambda: False)
    got_h, got_res = run(6, moe_glue_fold=True)
    assert torch.equal(got_h, want_h) and torch.equal(got_res, want_res)
    for (q, s), (wq, ws) in zip(log.index, want_index, strict=True):
        assert torch.equal(q, wq) and torch.equal(s, ws)
    assert {(2, "in"), (3, "in")} <= set(log.norm)
    assert all(la is None for _, la in log.route)
    assert sum(add is not None for _, add in log.ar) == 3


def test_forward_glue_hands_over_routed_and_shared_which_sum_to_its_output(load):
    m3_model = load("atom.models.minimax_m3.model")
    glue = load("atom.models.minimax_m3.moe_route_glue")
    moe = m3_model.MiniMaxM3MoE.__new__(m3_model.MiniMaxM3MoE)
    nn.Module.__init__(moe)
    moe.experts = _Experts()
    moe.routed_scaling_factor = 2.0
    moe.shared_experts = _Shared()
    x = _rows(5, n=H, seed=4)
    r = {
        "normed": x,
        "x_q": x,
        "x_scale": None,
        "topk_weights": None,
        "topk_ids": None,
    }
    orig = glue.presort
    glue.presort = lambda r, moe: None
    try:
        routed, shared = moe.forward_glue(
            {**r, "moe_buf": torch.zeros_like(x)}, defer_shared=True
        )
        summed = moe.forward_glue({**r, "moe_buf": torch.zeros_like(x)})
    finally:
        glue.presort = orig
    assert torch.equal(routed + shared, summed)


def test_route_hands_the_op_the_lookahead_norm_and_its_rows(load, monkeypatch):
    glue = load("atom.models.minimax_m3.moe_route_glue")
    seen = {}
    op = SimpleNamespace(
        fused_all_reduce_rmsnorm_moe_route=lambda *a, **kw: seen.update(kw) or "r"
    )
    name = "aiter.ops.triton.comms.fused.all_reduce_rmsnorm_moe_route"
    monkeypatch.setitem(sys.modules, name, op)
    ex = SimpleNamespace(
        global_num_experts=8,
        top_k=2,
        renormalize=True,
        routed_scaling_factor=2.0,
        fold_routed_scaling_factor=True,
    )
    moe = SimpleNamespace(
        experts=ex, gate=SimpleNamespace(weight=None), e_score_correction_bias=None
    )
    monkeypatch.setitem(glue._COMMS, (H, 8, 2), object())
    norm = SimpleNamespace(weight=torch.zeros(H), variance_epsilon=1e-6)
    la_norm = SimpleNamespace(weight=torch.ones(H), variance_epsilon=1e-5)
    x = torch.zeros(3, H)
    glue.route(x, x, norm, moe)
    assert "lookahead_weight" not in seen
    q = torch.empty(3, H, dtype=torch.float8_e4m3fn)
    s = torch.empty(3, H // 32, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    glue.route(x, x, norm, moe, (la_norm, q, s))
    assert seen["lookahead_weight"] is la_norm.weight
    assert seen["lookahead_eps"] == 1e-5
    out = seen["out"]
    assert out["la_q"] is q and out["la_scale"].data_ptr() == s.data_ptr()
    assert out["la_scale"].dtype == torch.uint8
    glue.route(x, x, norm, moe, (la_norm, None, None))
    assert seen["out"] is None
