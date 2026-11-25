# SPDX-License-Identifier: MIT
"""MiniMax-M3 MoE ``two_streams`` and Pro's ``ProVariants.moe_two_stream_rows``.

On two streams a MoE layer forks a side stream from the current one, runs the shared
expert there, runs the router and routed experts on the current stream and joins the
side stream back before it reads the shared output. The outputs are those of
the one-stream path, summed or handed over (``moe_tail_ar``). The sequential path is the
reference throughout. On a GPU the same holds inside a recorded CUDA graph.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
# ``load``: that file's fixture, reached here as ``m3``
from test_m3pro_ar_fused_tail import _layers, _moe, _rows, load  # noqa: F401


@pytest.fixture
def m3(request):
    return request.getfixturevalue("load")


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, other):
        self.log.append(("wait", self.name, other.name))


@pytest.fixture
def streams(m3, monkeypatch):
    """Fake CUDA streams: ``log`` records the waits and which stream each expert ran on."""
    m3_model = m3("atom.models.minimax_m3.model")
    log = []
    main, side = _Stream("main", log), _Stream("side", log)
    current = [main]

    @contextmanager
    def on(stream):
        current.append(stream)
        try:
            yield
        finally:
            current.pop()

    monkeypatch.setattr(m3_model, "_moe_side_stream", lambda device: side)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a: current[-1])
    monkeypatch.setattr(torch.cuda, "stream", on)
    streams.log, streams.current = log, current
    return streams


def _logged(moe, streams):
    """Log the stream the routed experts and the shared expert run on."""
    experts, shared = moe.experts.forward, moe.shared_experts.forward

    def routed_on(*a, **k):
        streams.log.append(("routed", streams.current[-1].name))
        return experts(*a, **k)

    def shared_on(*a, **k):
        streams.log.append(("shared", streams.current[-1].name))
        return shared(*a, **k)

    moe.experts.forward, moe.shared_experts.forward = routed_on, shared_on
    return moe


@pytest.mark.parametrize("defer", [False, True])
def test_two_streams_fork_run_shared_on_the_side_and_join(m3, streams, defer):
    moe = _logged(_moe(m3), streams)
    x = _rows(5, seed=4)
    want = moe(x, defer_shared=defer)
    assert streams.log == [("routed", "main"), ("shared", "main")]
    streams.log.clear()
    got = moe(x, defer_shared=defer, two_streams=True)
    assert streams.log == [
        ("wait", "side", "main"),
        ("shared", "side"),
        ("routed", "main"),
        ("wait", "main", "side"),
    ]
    for g, w in zip(got, want) if defer else [(got, want)]:
        assert torch.equal(g, w)


def test_a_layer_without_a_shared_expert_stays_on_one_stream(m3, streams):
    moe = _moe(m3)
    moe.shared_experts = None
    x = _rows(5, seed=4)
    assert torch.equal(moe(x, two_streams=True), moe(x))
    assert streams.log == []


@pytest.mark.parametrize("rows, forks", [(0, False), (5, False), (6, True), (64, True)])
@pytest.mark.parametrize("tail", [False, True])
def test_layers_fork_up_to_the_variants_rows(
    m3, monkeypatch, streams, rows, forks, tail
):
    """Three MoE layers at 6 rows; ``moe_tail_ar`` hands the shared output to the next
    layer's all-reduce as before."""
    run = _layers(m3, monkeypatch)
    want_h, want_res, want_adds = run(ar_fused_rows=256, moe_tail_ar=tail)
    streams.log.clear()
    got_h, got_res, got_adds = run(
        ar_fused_rows=256, moe_tail_ar=tail, moe_two_stream_rows=rows
    )
    assert sum(e == ("wait", "side", "main") for e in streams.log) == (
        3 if forks else 0
    )
    assert torch.equal(got_h, want_h) and torch.equal(got_res, want_res)
    assert [n for n, a in got_adds if a is not None] == [
        n for n, a in want_adds if a is not None
    ]


# --- GPU: a recorded CUDA graph --------------------------------------------------------


class _GpuExperts(nn.Module):
    """A routed-expert stand-in long enough for the shared expert to run beside it."""

    fold_routed_scaling_factor = True

    def __init__(self, h, n):
        super().__init__()
        self.w = [
            torch.randn(h, h, device="cuda", dtype=torch.bfloat16) / h**0.5
            for _ in range(n)
        ]

    def forward(self, hidden_states, router_logits):
        y = hidden_states * router_logits[:, :1].sigmoid()
        for w in self.w:
            y = torch.tanh(y @ w)
        return y


class _GpuShared(nn.Module):
    def __init__(self, h):
        super().__init__()
        self.up = torch.randn(h, 2 * h, device="cuda", dtype=torch.bfloat16) / h**0.5
        self.down = torch.randn(h, h, device="cuda", dtype=torch.bfloat16) / h**0.5

    def forward(self, x, x_scale=None, quant_down=False):
        g, u = (x @ self.up).chunk(2, -1)
        return (torch.nn.functional.silu(g) * u) @ self.down


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("defer", [False, True])
def test_graph_replay_on_two_streams_equals_one_stream(m3, defer):
    torch.manual_seed(0)
    h = 1024
    moe = _moe(m3)
    moe.gate = nn.Linear(h, 8, bias=False, device="cuda", dtype=torch.bfloat16)
    moe.experts, moe.shared_experts = _GpuExperts(h, 6), _GpuShared(h)
    x = torch.randn(64, h, device="cuda", dtype=torch.bfloat16)

    def graphed(two):
        static = x.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            moe(static, defer_shared=defer, two_streams=two)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = moe(static, defer_shared=defer, two_streams=two)
        return g, static, out

    eager = moe(x, defer_shared=defer)
    results = []
    for two in (False, True):
        g, static, out = graphed(two)
        for seed in (1, 2):
            fresh = torch.randn(x.shape, generator=torch.Generator().manual_seed(seed))
            static.copy_(fresh)
            g.replay()
            torch.cuda.synchronize()
            results.append([t.clone() for t in (out if defer else [out])])
        static.copy_(x)
        g.replay()
        torch.cuda.synchronize()
        for got, want in zip(out if defer else [out], eager if defer else [eager]):
            assert torch.equal(got, want)
    one, two = results[:2], results[2:]
    for a, b in zip(one, two):
        assert all(torch.equal(p, q) for p, q in zip(a, b))
