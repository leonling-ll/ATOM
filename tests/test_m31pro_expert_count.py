# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The expert count (``eplb.ExpertCountProfile``): the distinct routed experts each MoE
layer reads in a forward, recorded at ``FusedMoE._run_experts`` from the ids ``apply``
receives, padded graph rows left out; nothing recorded or allocated while it is off."""

import json
from types import SimpleNamespace

import pytest
import torch
from import_guard import skip_if_dependency_missing
from torch import nn

try:
    import atom.config  # noqa: F401  (before atom.model_ops, as test_eplb_module_a)
    from atom.model_ops import eplb, moe
except ImportError as _e:  # aiter / triton absent on a CPU-only runner
    skip_if_dependency_missing(_e, "requires the aiter stack")

ROUTED = 8  # routed experts; id 8 is a fused shared expert's column


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.expert_layout = SimpleNamespace(num_routed=ROUTED)
        self.expert_count_slot = None


def _batch(rows, seqs, prefill=0, dummy=False):
    return SimpleNamespace(
        total_tokens_num=rows,
        total_seqs_num=seqs,
        total_tokens_num_prefill=prefill,
        is_dummy_run=dummy,
    )


@pytest.fixture
def no_count(monkeypatch):
    monkeypatch.setattr(eplb, "_EXPERT_COUNT", None)
    monkeypatch.setattr(eplb, "_MANAGER", None)
    cfg = SimpleNamespace(eplb_enable=False)
    monkeypatch.setattr(eplb, "get_current_atom_config", lambda: cfg)


def test_distinct_experts_per_layer(tmp_path, no_count):
    layers = [_Layer(), _Layer()]
    path = tmp_path / "x_experts.jsonl"
    prof = eplb.enable_expert_count(layers, str(path))
    assert [m.expert_count_slot for m in layers] == [0, 1]
    assert eplb.get_expert_count() is prof
    prof.on_forward_start()
    # layer 0: 3 real rows (experts 1 2 3 / 3 4 5 / 1 1 7, the shared column 8 not
    # counted) and 2 padded rows of a graph bucket, routed anywhere
    ids0 = torch.tensor([[1, 2, 8], [3, 4, 8], [7, 1, 8], [0, 6, 8], [5, 5, 8]])
    prof.record_rows(0, ids0[:, :2].clone(), torch.tensor([3], dtype=torch.int32))
    prof.record_rows(0, ids0[:3, 2:].clone(), None)
    # layer 1: an eager forward (every row real) with -1 ids and a repeated expert
    prof.record_rows(1, torch.tensor([[2, -1], [2, 2], [-1, 0]]), None)
    prof.commit(_batch(rows=3, seqs=1))
    prof.commit(_batch(rows=3, seqs=1, dummy=True))
    prof.on_forward_start()
    prof.commit(_batch(rows=100, seqs=1, prefill=100))
    first, second = (json.loads(x) for x in path.read_text().splitlines())
    assert first["distinct"] == [5, 2] and first["ids"] == 6
    assert (first["phase"], first["rows"], first["seqs"]) == ("decode", 3, 1)
    assert second["phase"] == "prefill" and second["distinct"] == [0, 0]


def test_run_experts_records_the_real_rows_apply_receives(monkeypatch, no_count):
    ctx = SimpleNamespace(pcp_split=False, is_dummy_run=False)
    monkeypatch.setattr(
        moe, "get_forward_context", lambda: SimpleNamespace(context=ctx)
    )
    real = torch.tensor([2], dtype=torch.int32)
    monkeypatch.setattr(moe, "pass_scheduled_tokens_gpu", lambda rows: real)
    seen = {}

    def apply(**kw):
        seen.update(kw)
        return kw["x"]

    layer = _Layer()
    layer.moe_parallel_config = SimpleNamespace(pcp_merge=False)
    # placement moves every id up by one: the count reads the placed ids
    layer.quant_method = SimpleNamespace(
        place_experts=lambda lyr, w, i, g: (w, i + 1), apply=apply
    )
    layer.global_num_experts, layer.expert_map, layer.activation = ROUTED, None, None
    layer.apply_router_weight_on_input, layer.prefix = False, "m"
    x, w = torch.zeros(4, 2), torch.ones(4, 2)
    ids = torch.tensor([[0, 1], [0, 2], [5, 6], [4, 3]])
    run = moe.FusedMoE._run_experts

    def no_count_call(*a, **k):
        raise AssertionError("the count ran while off")

    monkeypatch.setattr(moe, "get_expert_count", no_count_call)
    run(layer, x, w, ids)  # off: the slot is None
    assert seen["scheduled_tokens_gpu"] is real
    monkeypatch.setattr(moe, "get_expert_count", eplb.get_expert_count)
    prof = eplb.enable_expert_count([layer], "/dev/null")
    prof.on_forward_start()
    run(layer, x, w, ids)
    assert prof._cur_pass_count[0].tolist() == [0, 2, 1, 1, 0, 0, 0, 0]


def test_off_allocates_nothing_and_passes_through(no_count):
    calls = []

    @eplb.with_eplb_forward_monitor
    def forward(self, batch):
        calls.append(batch)
        return "out"

    assert forward(None, _batch(8, 1)) == "out" and len(calls) == 1
    assert eplb.get_expert_count() is None
    assert _Layer().expert_count_slot is None


def test_forward_monitor_commits_each_forward(tmp_path, no_count):
    layer = _Layer()
    path = tmp_path / "e.jsonl"
    prof = eplb.enable_expert_count([layer], str(path))

    @eplb.with_eplb_forward_monitor
    def forward(self, batch):
        prof.record_rows(0, torch.tensor([[batch.total_seqs_num, 7]]), None)
        return batch.total_tokens_num

    assert forward(None, _batch(1, 1)) == 1 and forward(None, _batch(1, 3)) == 1
    recs = [json.loads(x)["distinct"] for x in path.read_text().splitlines()]
    assert recs == [[2], [2]]  # zeroed between forwards: {1, 7}, then {3, 7}
    assert prof._cur_pass_count[0, 1].item() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_count_replays_in_a_graph(tmp_path, no_count):
    """Captured once, the count follows the real-row count of each replay."""
    layer = _Layer().cuda()
    prof = eplb.enable_expert_count([layer], str(tmp_path / "g.jsonl"))
    ids = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]], device="cuda")
    real = torch.tensor([4], dtype=torch.int32, device="cuda")
    prof.record_rows(0, ids, real)  # warm up outside the capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        prof.record_rows(0, ids, real)
    for n, want in ((1, 2), (3, 6), (4, 8)):
        prof.on_forward_start()
        real.fill_(n)
        g.replay()
        torch.cuda.synchronize()
        assert int((prof._cur_pass_count[0] > 0).sum()) == want
