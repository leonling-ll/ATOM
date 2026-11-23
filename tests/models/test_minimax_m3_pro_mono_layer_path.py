# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The Pro decoder layer's two paths, with the kernels replaced by recorders.

* mono off (no route, a route that declines, a dense layer): the layer makes
  production's calls in production's order -- attention through o_proj,
  AR #1 + norm, the lookahead put, the MoE -- and returns their outputs;
* a MoE layer the route takes: the attention stops before o_proj and the
  runner's segment returns the layer's (FFN partial, residual);
* the launch (``ProMonoRunner.launch``, the compiled kernel faked): every
  ABI argument by name, the layer's ``moe_w`` table in ``MOE_WEIGHTS`` order;
* the segment (``ProMonoRunner.segment``, its launch faked): the lookahead in
  ``resid_in[L + 2]`` as MXFP8, production's MoE on the kernel's (x, x8) under
  ``upto = "norm"`` and none when the build writes the FFN; check mode runs
  production on a copy of the residual first and logs every output.

Needs aiter (``pro`` imports the model stack) and FlyDSL (the runner)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="pro imports the ATOM model stack")
pro = pytest.importorskip("atom.models.minimax_m3.pro")
runner_mod = pytest.importorskip("atom.models.minimax_m3.pro_mono.runner")

from atom.model_ops.minimax_m3 import pro_attn
from atom.model_ops.minimax_m3.pro_attn import ProVariants

N, H, OK = 3, 8, 6  # rows, hidden, o_proj's K (small stand-ins)
NATIVE = SimpleNamespace(native_a8_group_rows=1)
PLAIN = SimpleNamespace(native_a8_group_rows=None)


def _t(seed, cols=H):
    return torch.randn(N, cols, generator=torch.Generator().manual_seed(seed))


class Recorder:
    def __init__(self):
        self.calls = []


@pytest.fixture
def rec(monkeypatch):
    r = Recorder()
    monkeypatch.setattr(pro_attn, "_variants", ProVariants(ar_lamport_rows=16))
    monkeypatch.setattr(pro, "_aiter_mx_norm", lambda: True)

    def ar_norm(hidden, residual, norm):
        r.calls.append("ar_in")
        return hidden + residual, residual + 1

    def ar_norm_mx(hidden, residual, norm, lamport_rows):
        r.calls.append(("ar_mx", lamport_rows))
        h_mid = hidden + residual
        return h_mid * 2, h_mid * 3, h_mid * 4, h_mid

    monkeypatch.setattr(pro, "fused_allreduce_gemma_rms_norm", ar_norm)
    monkeypatch.setattr(pro, "fused_allreduce_gemma_rms_norm_mxfp8", ar_norm_mx)
    return r


def _layer(rec, moe=True, route=None):
    layer = pro.MiniMaxM3ProDecoderLayer.__new__(pro.MiniMaxM3ProDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_num = 4
    layer.is_moe_layer = moe
    layer.input_layernorm = object()
    layer.post_attention_layernorm = object()
    state = SimpleNamespace(
        source_for=lambda *a: None,
        put=lambda lid, x, res: rec.calls.append(("put", lid)),
        resid_in={},
    )
    layer._pro_state = state
    layer._mono = route

    class Attn:
        qk_proj = PLAIN
        index_qk_proj = PLAIN

        def __call__(self, **kw):
            rec.calls.append("attn")
            return self.o_proj(self.forward_core(**kw))

        def forward_core(self, **kw):
            rec.calls.append("core")
            return kw["hidden_states"][:, :OK] * 5

        def o_proj(self, core):
            rec.calls.append("o_proj")
            return torch.cat([core, core[:, : H - OK]], 1) * 7

    layer.self_attn = Attn()

    def ffn(x, *args, shared_input=None, quant_down=False):
        rec.calls.append(("ffn", shared_input is not None, quant_down))
        return x * 11

    ffn.shared_experts = SimpleNamespace(gate_up_proj=NATIVE)
    ffn.gate_up_proj = NATIVE
    if moe:
        layer.block_sparse_moe = ffn
    else:
        layer.mlp = ffn
    return layer


def _production(hidden, residual):
    """What the layer computes with the recorders, production path."""
    h, res = hidden + residual, residual + 1
    core = h[:, :OK] * 5
    o = torch.cat([core, core[:, : H - OK]], 1) * 7
    h_mid = o + res
    return h_mid * 2 * 11, h_mid


PRODUCTION_CALLS = [
    "ar_in", "attn", "core", "o_proj", ("ar_mx", 16), ("put", 4), ("ffn", True, True),
]  # fmt: skip


class Route:
    def __init__(self, runner):
        self.runner, self.asked = runner, []

    def take(self, layer_id, rows):
        self.asked.append((layer_id, rows))
        return self.runner


@pytest.mark.parametrize("route", ["none", "declines"])
def test_mono_off_is_the_production_path(rec, route):
    r = None if route == "none" else Route(None)
    layer = _layer(rec, route=r)
    hidden, residual = _t(1), _t(2)
    out, res = layer(None, hidden, residual)
    want_out, want_res = _production(hidden, residual)
    assert rec.calls == PRODUCTION_CALLS
    assert torch.equal(out, want_out) and torch.equal(res, want_res)
    if r is not None:
        assert r.asked == [(4, N)]


def test_a_dense_layer_never_asks_the_route(rec):
    r = Route(object())
    layer = _layer(rec, moe=False, route=r)
    layer(None, _t(1), _t(2))
    assert r.asked == [] and "core" in rec.calls and "o_proj" in rec.calls


def test_a_taken_layer_stops_before_o_proj_and_returns_the_segment(rec):
    seg_out = (_t(8), _t(9))

    class Runner:
        def segment(self, layer, core, residual, reference):
            rec.calls.append("segment")
            self.args = (layer, core, residual, reference)
            return seg_out

    runner = Runner()
    layer = _layer(rec, route=Route(runner))
    hidden, residual = _t(1), _t(2)
    out = layer(None, hidden, residual)
    assert rec.calls == ["ar_in", "core", "segment"]
    assert out == seg_out
    got_layer, core, res_in, reference = runner.args
    assert got_layer is layer
    assert torch.equal(core, (hidden + residual)[:, :OK] * 5)
    assert torch.equal(res_in, residual + 1)
    # the check's reference: production's o_proj and AR #1 + norm of the core
    _, _, _, h_mid = reference(res_in.clone())
    assert rec.calls[-2:] == ["o_proj", ("ar_mx", 16)]
    assert torch.equal(h_mid, _production(hidden, residual)[1])


# ------------------------------------------------------------- the segment


def _runner(rec, outputs, check=False, reader=6):
    r = runner_mod.ProMonoRunner.__new__(runner_mod.ProMonoRunner)
    r.check, r.debug, r.rank = check, False, 0
    r.layers = {4: runner_mod.MonoLayer(w=None, reader=reader, moe_w=None)}
    r.state = SimpleNamespace(resid_in={})

    def put(lid, x, h_mid):
        rec.calls.append(("put", lid))
        r.state.resid_in[lid + 2] = (h_mid * 13, h_mid.to(torch.uint8))

    r.state.put = put

    def launch(lid, core, residual):
        rec.calls.append("launch")
        return outputs

    r.launch = launch
    return r


def _outputs(ffn=False):
    out = {
        "h_mid": _t(20), "x": _t(21), "x8": _t(22), "x8s": _t(23).to(torch.uint8),
        "la8": _t(24), "la8s": torch.full((N, 1), 127, dtype=torch.uint8),
    }  # fmt: skip
    if ffn:
        out["ffn_out"] = _t(25)
    return out


def test_segment_hands_on_the_lookahead_and_runs_the_moe_on_the_kernels_x(rec):
    out = _outputs()
    r = _runner(rec, out)
    layer = _layer(rec)
    ffn, res = r.segment(layer, _t(3, OK), _t(4), None)
    assert rec.calls == ["launch", ("ffn", True, True)]
    assert torch.equal(ffn, out["x"] * 11) and res is out["h_mid"]
    la, la_s = r.state.resid_in[6]
    assert la is out["la8"] and la_s.dtype == torch.float8_e8m0fnu
    assert torch.equal(la_s.view(torch.uint8), out["la8s"])


def test_a_layer_without_a_reader_puts_no_lookahead(rec):
    r = _runner(rec, _outputs(), reader=None)
    r.segment(_layer(rec), _t(3, OK), _t(4), None)
    assert r.state.resid_in == {}


def test_a_build_that_writes_the_ffn_skips_the_production_moe(rec):
    out = _outputs(ffn=True)
    r = _runner(rec, out)
    ffn, _ = r.segment(_layer(rec), _t(3, OK), _t(4), None)
    assert rec.calls == ["launch"] and ffn is out["ffn_out"]


def test_check_mode_runs_production_on_a_copy_then_goes_on_from_mono(
    rec, monkeypatch, caplog
):
    monkeypatch.setattr(runner_mod, "capturing", lambda: False)
    out = _outputs()
    r = _runner(rec, out, check=True)
    residual = _t(4)
    seen = []

    def reference(res):
        rec.calls.append("reference")
        seen.append(res)
        return res * 2, res * 3, res * 4, res

    with caplog.at_level("INFO", logger="atom"):
        ffn, h_mid = r.segment(_layer(rec), _t(3, OK), residual, reference)
    assert rec.calls == [
        "reference", ("put", 4), ("ffn", True, True), "launch", ("ffn", True, True),
    ]  # fmt: skip
    assert seen[0] is not residual and torch.equal(seen[0], residual)
    # mono's outputs go on; production's lookahead was taken back out
    assert h_mid is out["h_mid"] and torch.equal(ffn, out["x"] * 11)
    assert r.state.resid_in[6][0] is out["la8"]
    line = caplog.records[-1].getMessage()
    for name in ("h_mid", "x8s", "la8 ", "la8s", "ffn"):
        assert name in line, line
    assert "x8s differ" in line and "la8s differ" in line


def test_check_mode_off_under_capture(rec, monkeypatch):
    monkeypatch.setattr(runner_mod, "capturing", lambda: True)
    r = _runner(rec, _outputs(), check=True)
    r.segment(_layer(rec), _t(3, OK), _t(4), lambda res: pytest.fail("ran"))
    assert rec.calls == ["launch", ("ffn", True, True)]


# -------------------------------------------------------------- the launch


def _bound():
    """A ``ProMoeLayer`` stand-in: a distinct small tensor per operand."""
    from atom.models.minimax_m3.pro_mono.weights import ProMoeLayer

    names = [f.name for f in ProMoeLayer.__dataclass_fields__.values()]
    vals = {n: torch.zeros(4 + i, dtype=torch.uint8) for i, n in enumerate(names)}
    vals |= {"layer_id": 4, "eps": 1e-6, "eps_ahead": 1e-6}
    return SimpleNamespace(**vals)


def test_moe_table_holds_the_operands_in_the_kernels_order():
    from atom.models.minimax_m3.pro_mono.layout import MOE_WEIGHTS

    w = _bound()
    table = runner_mod.moe_table(w, torch.device("cpu"))
    assert table.dtype == torch.int64
    assert table.tolist() == [getattr(w, n).data_ptr() for n in MOE_WEIGHTS]
    assert len(set(table.tolist())) == len(MOE_WEIGHTS)


@pytest.mark.parametrize("reader", [6, None])
def test_launch_passes_every_abi_argument(monkeypatch, reader):
    from atom.models.minimax_m3.pro_mono.layout import DIMS, HIDDEN

    w = _bound()
    r = runner_mod.ProMonoRunner.__new__(runner_mod.ProMonoRunner)
    r.debug, r.rank, r.first, r.last = False, 3, 4, 4
    table = runner_mod.moe_table(w, torch.device("cpu"))
    r.layers = {4: runner_mod.MonoLayer(w=w, reader=reader, moe_w=table)}
    r.scratch = torch.zeros(16, dtype=torch.uint8)
    r.plain = torch.zeros(32, dtype=torch.uint8)
    r.peers = SimpleNamespace(addresses=torch.zeros(8, dtype=torch.int64))
    steps = []
    r.mailboxes = SimpleNamespace(begin_step=lambda: steps.append(1))
    kind = r.layers[4].kind
    r.builds = {2: {kind: "launcher"}}
    got = []
    monkeypatch.setattr(runner_mod, "_run_compiled", lambda *a: got.append(a))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "stream")
    core = torch.zeros(2, DIMS.o_k, dtype=torch.bfloat16)
    res = torch.zeros(2, HIDDEN, dtype=torch.bfloat16)
    out = r.launch(4, core, res)
    (call,) = got
    assert call[0] == "launcher" and call[-1] == "stream" and steps == [1]
    args = dict(zip(runner_mod.ABI.names, call[1:-1]))
    assert args["moe_w"] == table.data_ptr()
    assert args["plain"] == r.plain.data_ptr()
    assert args["scratch"] == r.scratch.data_ptr() and args["rank"] == 3
    # no forward context: every row real
    assert args["nvalid"] == 0
    assert args["ffn_out"] == out["ffn_out"].data_ptr()
    assert out["ffn_out"].shape == (2, HIDDEN)
    for name in ("la8", "la8s"):
        if reader is None:
            assert args[name] == 0 and name not in out
        else:
            assert args[name] == out[name].data_ptr()


def test_the_real_row_count_is_the_forward_contexts(monkeypatch):
    """``nvalid``: production's own row count tensor when it covers the pass's
    rows, else 0 (every row routed)."""
    count = torch.tensor([5], dtype=torch.int32)
    monkeypatch.setattr(
        runner_mod,
        "pass_scheduled_tokens_gpu",
        lambda rows: count if rows == 8 else None,
    )
    assert runner_mod._real_rows_ptr(8) == count.data_ptr()
    assert runner_mod._real_rows_ptr(4) == 0
    wide = torch.tensor([5], dtype=torch.int64)
    monkeypatch.setattr(runner_mod, "pass_scheduled_tokens_gpu", lambda rows: wide)
    with pytest.raises(RuntimeError):
        runner_mod._real_rows_ptr(8)
