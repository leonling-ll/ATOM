# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's moe_route_glue (one aiter launch for a MoE layer's all-reduce, norm,
router, top-k and sort) and the presorted expert path it feeds: what the model hands aiter,
and aiter's generic-pipeline presort match (loaded from the aiter tree in AITER_D3_TREE).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="the model stack imports aiter")

from aiter import QuantType

from atom.model_ops import moe as moe_mod
from atom.models.minimax_m3 import model as m3_model
from atom.models.minimax_m3 import moe_route_glue as glue

E, H, K = 384, 64, 4


def _route_out(M=3):
    return {
        "normed": torch.randn(M, H, dtype=torch.bfloat16),
        "x_q": torch.zeros(M, H, dtype=torch.uint8),
        "x_scale": torch.zeros(M, H // 32, dtype=torch.uint8),
        "residual_out": torch.randn(M, H, dtype=torch.bfloat16),
        "topk_ids": torch.zeros(M, K, dtype=torch.int32),
        "topk_weights": torch.ones(M, K),
        "sorted_ids": torch.zeros(64, dtype=torch.int32),
        "sorted_weights": torch.zeros(64),
        "sorted_expert_ids": torch.zeros(4, dtype=torch.int32),
        "num_valid_ids": torch.tensor([16, M], dtype=torch.int32),
        "moe_buf": torch.zeros(M, H, dtype=torch.bfloat16),
    }


class _Experts(nn.Module):
    def __init__(self, fold=True, takes=True):
        super().__init__()
        self.fold_routed_scaling_factor = fold
        self.routed_scaling_factor = 2.0
        self.global_num_experts, self.top_k = E, K
        self.scoring_func, self.use_grouped_topk = "sigmoid", False
        self.custom_routing_function = None
        self.renormalize = True
        self._takes = takes
        self.calls = []

    def takes_presorted(self, rows):
        return self._takes

    def forward_routed(self, x, w, ids, out):
        self.calls.append((x, w, ids, out))
        return out.add_(1.0)


def _moe(fold=True, takes=True):
    moe = m3_model.MiniMaxM3MoE.__new__(m3_model.MiniMaxM3MoE)
    nn.Module.__init__(moe)
    moe.experts = _Experts(fold, takes)
    moe.gate = SimpleNamespace(weight=torch.zeros(E, H, dtype=torch.bfloat16))
    moe.e_score_correction_bias = torch.zeros(E)
    moe.routed_scaling_factor = 2.0
    moe.fuse_shared_experts = False
    moe.shared_experts = lambda x, s, quant_down=False: torch.full(
        (x.shape[0], H), 5.0, dtype=torch.bfloat16
    )
    return moe


def test_forward_glue_registers_the_sort_and_adds_the_shared_expert(monkeypatch):
    seen = {}
    import aiter.fused_moe as afm

    monkeypatch.setattr(
        afm, "register_presorted_moe", lambda aux: seen.update(aux or {}), raising=False
    )
    moe, r = _moe(), _route_out()
    out = moe.forward_glue(r, quant_down=True)
    assert torch.equal(
        out, torch.full((3, H), 6.0, dtype=torch.bfloat16)
    )  # routed 0 + 1, shared 5
    ((x, w, ids, buf),) = moe.experts.calls
    assert (
        x is r["normed"]
        and w is r["topk_weights"]
        and ids is r["topk_ids"]
        and buf is r["moe_buf"]
    )
    assert seen["topk_ids_ptr"] == r["topk_ids"].data_ptr()
    assert seen["topk_weights_ptr"] == r["topk_weights"].data_ptr()
    assert seen["moe_buf_ptr"] == r["moe_buf"].data_ptr()
    assert seen["covers_all_rows"] and seen["moe_buf_zeroed"]
    assert (seen["M"], seen["block_m"], seen["num_experts"], seen["topk"]) == (
        3,
        16,
        E,
        K,
    )
    assert seen["sorted_token_ids"] is r["sorted_ids"]


def test_forward_glue_multiplies_only_when_the_routing_did_not(monkeypatch):
    import aiter.fused_moe as afm

    monkeypatch.setattr(afm, "register_presorted_moe", lambda aux: None, raising=False)
    out = _moe(fold=False).forward_glue(_route_out())
    assert torch.equal(out, torch.full((3, H), 7.0, dtype=torch.bfloat16))  # 2 x 1 + 5


@pytest.fixture
def comm(monkeypatch):
    monkeypatch.setitem(glue._COMMS, (H, E, K), object())


@pytest.mark.parametrize(
    "rows, fix, ok",
    [
        (64, {}, True),
        (512, {}, True),
        (513, {}, False),
        (64, {"takes": False}, False),
        (64, {"scoring": "softmax"}, False),
        (64, {"grouped": True}, False),
        (64, {"fused_shared": True}, False),
        (64, {"no_shared": True}, False),
    ],
)
def test_applies(comm, rows, fix, ok):
    moe = _moe(takes=not fix.get("takes") is False)
    if "scoring" in fix:
        moe.experts.scoring_func = fix["scoring"]
    if fix.get("grouped"):
        moe.experts.use_grouped_topk = True
    if fix.get("fused_shared"):
        moe.fuse_shared_experts = True
    if fix.get("no_shared"):
        moe.shared_experts = None
    assert glue.applies(moe, rows) is ok


def test_applies_needs_the_comm():
    assert not glue.applies(_moe(), 8)


def test_route_hands_the_op_pros_semantics(monkeypatch, comm):
    seen = {}
    mod = types.ModuleType("aiter.ops.triton.comms.fused.all_reduce_rmsnorm_moe_route")
    mod.fused_all_reduce_rmsnorm_moe_route = (
        lambda *a, **kw: seen.update(args=a, **kw) or "r"
    )
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    moe = _moe()
    norm = SimpleNamespace(weight=torch.zeros(H), variance_epsilon=1e-6)
    x, res = torch.zeros(3, H), torch.zeros(3, H)
    assert glue.route(x, res, norm, moe) == "r"
    a = seen["args"]
    assert (
        a[0] is x
        and a[1] is res
        and a[2] is norm.weight
        and a[3] == 1e-6
        and a[4] is moe.gate.weight
    )
    assert seen["correction_bias"] is moe.e_score_correction_bias
    assert (
        seen["scoring"] == "sigmoid"
        and seen["renormalize"]
        and seen["gemma"]
        and seen["quant"]
    )
    assert (
        seen["routed_scaling_factor"] == 2.0
        and seen["block_size"] == 16
        and seen["zero_moe_buf"]
    )
    moe.experts.fold_routed_scaling_factor = False
    glue.route(x, res, norm, moe)
    assert seen["routed_scaling_factor"] == 1.0  # the multiply stays in forward_glue


def test_forward_routed_passes_the_output_to_the_experts():
    seen = {}
    layer = SimpleNamespace(
        _run_experts=lambda *a, **kw: seen.update(args=a, **kw) or "out"
    )
    x, w, i, buf = (torch.zeros(1) for _ in range(4))
    assert moe_mod.FusedMoE.forward_routed(layer, x, w, i, buf) == "out"
    assert seen["args"] == (x, w, i) and seen["routed_output"] is buf


def _mxfp4_method(**over):
    m = moe_mod.Mxfp4MoEMethod.__new__(moe_mod.Mxfp4MoEMethod)
    for k, v in {
        "fused_experts": None,
        "use_triton": False,
        "use_triton_ep": False,
        "use_triton_decode": False,
        "is_gfx1250": False,
        "use_flydsl_herd": False,
        "is_guinterleave": False,
        "fp8_activations": False,
        "quant_type": QuantType.per_1x32,
    }.items():
        setattr(m, k, v)
    for k, v in over.items():
        setattr(m, k, v)
    return m


_LAYER = SimpleNamespace(
    **{
        n: None
        for n in (
            "w13_weight",
            "w2_weight",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_bias",
            "w2_bias",
            "w13_input_scale",
            "w2_input_scale",
            "expert_mask",
        )
    },
    swiglu_limit=7.0,
)


def test_apply_hands_fused_moe_the_presorted_output(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        moe_mod, "fused_moe", lambda *a, **kw: seen.update(kw) or kw.get("output")
    )
    buf = torch.zeros(2, 4)
    out = _mxfp4_method().apply(
        _LAYER,
        torch.zeros(2, 4),
        torch.ones(2, 1),
        torch.zeros(2, 1, dtype=torch.int32),
        routed_output=buf,
    )
    assert out is buf and seen["output"] is buf


def test_apply_refuses_a_presorted_output_on_a_sorting_path():
    with pytest.raises(RuntimeError, match="presorted output"):
        _mxfp4_method(fused_experts=object()).apply(
            _LAYER,
            torch.zeros(2, 4),
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.int32),
            routed_output=torch.zeros(2, 4),
        )


# ── aiter: the generic pipeline's presort match (opt/aiter-d3's presorted_moe.py) ──

_TREE = os.environ.get("AITER_D3_TREE", "/home/yuhzhu/work/opt_aiter_d3")


@pytest.fixture(scope="module")
def pm():
    path = os.path.join(_TREE, "aiter", "presorted_moe.py")
    if not os.path.isfile(path):
        pytest.skip(f"no aiter tree at {_TREE}")
    spec = importlib.util.spec_from_file_location("aiter_d3_presorted_moe", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "match_presorted_moe_generic"):
        pytest.skip("the aiter tree has no generic presort")
    return mod


def _aux(M=3, topk=K, experts=E, **over):
    ids = torch.zeros(M, topk, dtype=torch.int32)
    w = torch.ones(M, topk)
    rows = (M * topk + experts * 16 - topk + 15) // 16 * 16
    out = torch.zeros(M, H, dtype=torch.bfloat16)
    aux = {
        "sorted_token_ids": torch.zeros(rows, dtype=torch.int32),
        "sorted_weights": torch.zeros(rows),
        "sorted_expert_ids": torch.zeros(rows // 16, dtype=torch.int32),
        "num_valid_ids": torch.zeros(2, dtype=torch.int32),
        "topk_ids_ptr": ids.data_ptr(),
        "topk_weights_ptr": w.data_ptr(),
        "M": M,
        "block_m": 16,
        "num_experts": experts,
        "topk": topk,
        "covers_all_rows": True,
        "moe_buf_zeroed": True,
        "moe_buf_ptr": out.data_ptr(),
    }
    aux.update(over)
    return aux, ids, w, out


def _match(pm, aux, ids, w, out, **kw):
    args = {
        "block_m": 16,
        "num_experts": E,
        "topk": K,
        "expert_mask": None,
        "output": out,
    }
    args.update(kw)
    return pm.match_presorted_moe_generic(aux, ids, w, **args)


def test_generic_presort_matches_its_own_call(pm):
    aux, ids, w, out = _aux()
    assert _match(pm, aux, ids, w, out) is aux


@pytest.mark.parametrize(
    "break_it",
    [
        lambda a, ids, w, out: ({**a, "covers_all_rows": False}, ids, w, out, {}),
        lambda a, ids, w, out: ({**a, "moe_buf_zeroed": False}, ids, w, out, {}),
        lambda a, ids, w, out: (
            a,
            ids,
            w,
            torch.zeros_like(out),
            {},
        ),  # another output buffer
        lambda a, ids, w, out: (a, ids.clone(), w, out, {}),  # other topk_ids
        lambda a, ids, w, out: (a, ids, w, out, {"block_m": 32}),
        lambda a, ids, w, out: (a, ids, w, out, {"expert_mask": torch.ones(E)}),
        lambda a, ids, w, out: (a, ids, w, None, {}),
        lambda a, ids, w, out: (
            {**a, "sorted_expert_ids": a["sorted_expert_ids"][:-1]},
            ids,
            w,
            out,
            {},
        ),
        lambda a, ids, w, out: (
            {**a, "sorted_weights": a["sorted_weights"].to(torch.float16)},
            ids,
            w,
            out,
            {},
        ),
    ],
)
def test_generic_presort_refuses_what_it_cannot_stand_for(pm, break_it):
    a, ids, w, out = _aux()
    a2, ids2, w2, out2, kw = break_it(a, ids, w, out)
    assert _match(pm, a2, ids2, w2, out2, **kw) is None
