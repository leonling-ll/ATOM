# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Graph-or-eager counts per forward, and the c32 DSpark decode they count.

At the agentic c32 point (`scripts/m31pro/proxy_bench.sh 32`: max-num-seqs
2 x 32, a 16384-token budget, DSpark with 7 drafts, so 8 verify rows a
sequence) every decode batch the scheduler can form has a captured graph on
the default ladder, so no decode step runs eagerly; past the ladder it does,
and the counter says so.
"""

import types
from unittest import mock

import numpy as np
import pytest
from aiter_stub import atom_without_aiter
from prometheus_client import CollectorRegistry

from atom.config import CompilationConfig, Config
from atom.metrics.dispatch import ForwardDispatchMetrics
from atom.model_engine.arg_utils import EngineArgs, parse_size_list
from atom.utils.forward_context import ForwardMode
from atom.utils.region_graph import TokenGraph

C32_MAX_NUM_SEQS = 2 * 32
BUDGET = 16384
VERIFY_ROWS = 7 + 1


@pytest.fixture(scope="module")
def load():
    with atom_without_aiter() as load:
        yield load


@pytest.fixture(scope="module")
def model_runner(load):
    return load("atom.model_engine.model_runner")


@pytest.fixture(scope="module")
def Drafter(load):
    return load("atom.spec_decode.drafter").Drafter


def _samples(registry):
    return {
        (s.labels["model"], s.labels["phase"], s.labels["cudagraph"]): s.value
        for metric in registry.collect()
        for s in metric.samples
        if s.name == "atom:forward_steps_total"
    }


def _c32_ladder(model_runner):
    """The default ladder narrowed as `capture_cudagraph` narrows it."""
    cfg = types.SimpleNamespace(
        compilation_config=CompilationConfig(
            cudagraph_capture_sizes=parse_size_list(EngineArgs.cudagraph_capture_sizes)
        ),
        enable_tbo_decode=False,
    )
    max_bs = model_runner.max_schedulable_decode_bs(
        C32_MAX_NUM_SEQS, BUDGET, VERIFY_ROWS
    )
    sizes = [s for s in Config.capture_sizes.fget(cfg) if s <= max_bs]
    return np.asarray(sorted(sizes), dtype=np.int32)


def _decode(bs, ladder):
    batch = types.SimpleNamespace(
        total_seqs_num=bs,
        total_seqs_num_prefill=0,
        total_tokens_num=bs * VERIFY_ROWS,
        total_tokens_num_prefill=0,
        total_tokens_num_decode=bs * VERIFY_ROWS,
        num_scheduled_tokens=np.full(bs, VERIFY_ROWS, dtype=np.int32),
        has_prefill=False,
        is_dummy_run=False,
        final_prefill_requests=lambda: 0,
    )
    return ForwardMode.decide(
        batch=batch,
        dp_size=1,
        dp_group=None,
        enforce_eager=False,
        step_needs_eager=False,
        capture_sizes=ladder,
        is_block_drafter=True,
        tbo_on=False,
        local_tbo=(False, False, 0, 0),
        max_seqlen_q=VERIFY_ROWS,
        running_tokens_per_dp_rank=np.zeros(1, dtype=np.int32),
    )


def _prefill(tokens, ladder, buckets):
    """One request's prefill of ``tokens`` new rows, with the prefill token-graph
    ``buckets`` recorded (none: every prefill runs eagerly)."""
    batch = types.SimpleNamespace(
        total_seqs_num=1,
        total_seqs_num_prefill=1,
        total_tokens_num=tokens,
        total_tokens_num_prefill=tokens,
        total_tokens_num_decode=0,
        num_scheduled_tokens=np.array([tokens], dtype=np.int32),
        has_prefill=True,
        is_dummy_run=False,
        final_prefill_requests=lambda: 1,
    )
    with mock.patch.object(
        TokenGraph,
        "captured_sizes",
        staticmethod(lambda phase=None: buckets if phase == "prefill" else ()),
    ):
        return ForwardMode.decide(
            batch=batch,
            dp_size=1,
            dp_group=None,
            enforce_eager=False,
            step_needs_eager=False,
            capture_sizes=ladder,
            is_block_drafter=True,
            tbo_on=False,
            local_tbo=(False, False, 0, 0),
            max_seqlen_q=tokens,
            running_tokens_per_dp_rank=np.zeros(1, dtype=np.int32),
        )


def test_every_series_reads_zero_before_any_step():
    registry = CollectorRegistry()
    ForwardDispatchMetrics(registry=registry)
    samples = _samples(registry)
    assert len(samples) == 8 and set(samples.values()) == {0.0}


def test_each_forward_lands_in_its_own_series():
    registry = CollectorRegistry()
    metrics = ForwardDispatchMetrics(registry=registry)
    metrics.record("target", is_prefill=False, cudagraph=True)
    metrics.record("target", is_prefill=False, cudagraph=True)
    metrics.record("target", is_prefill=True, cudagraph=False)
    metrics.record("draft", is_prefill=True, cudagraph=True)
    samples = _samples(registry)
    assert samples[("target", "decode", "true")] == 2
    assert samples[("target", "prefill", "false")] == 1
    assert samples[("draft", "prefill", "true")] == 1
    assert samples[("target", "decode", "false")] == 0
    assert sum(samples.values()) == 4


def test_every_c32_decode_batch_replays_a_graph(model_runner):
    ladder = _c32_ladder(model_runner)
    assert ladder[-1] == C32_MAX_NUM_SEQS
    registry = CollectorRegistry()
    metrics = ForwardDispatchMetrics(registry=registry)
    for bs in range(1, C32_MAX_NUM_SEQS + 1):
        mode = _decode(bs, ladder)
        assert mode.running_bs >= bs
        # The graph holds every verify row of the padded batch.
        assert mode.running_tokens == mode.running_bs * VERIFY_ROWS <= BUDGET
        metrics.record("target", mode.is_prefill, mode.use_cudagraph)
    samples = _samples(registry)
    assert samples[("target", "decode", "true")] == C32_MAX_NUM_SEQS
    assert samples[("target", "decode", "false")] == 0


def test_a_decode_past_the_ladder_is_counted_eager(model_runner):
    ladder = _c32_ladder(model_runner)
    registry = CollectorRegistry()
    metrics = ForwardDispatchMetrics(registry=registry)
    mode = _decode(int(ladder[-1]) + 1, ladder)
    metrics.record("target", mode.is_prefill, mode.use_cudagraph)
    assert _samples(registry)[("target", "decode", "false")] == 1


class _Pass:
    def __init__(self, sizes):
        self.sizes = sizes

    def is_captured(self, running_bs):
        return running_bs in self.sizes


def test_a_draft_replays_only_where_every_declared_pass_was_recorded(Drafter):
    def drafter(*passes):
        return types.SimpleNamespace(draft_graphs=passes)

    both = drafter(_Pass({1, 2, 4}), _Pass({1, 2, 4}))
    assert Drafter.draft_replays(both, 4)
    assert not Drafter.draft_replays(both, 8)
    assert not Drafter.draft_replays(drafter(_Pass({4}), _Pass({2})), 4)
    assert not Drafter.draft_replays(drafter(), 4)


# The default `ATOM_TOKEN_GRAPH_SIZES`, as a backend with `supports_token_graph`
# records them.
PREFILL_BUCKETS = (256, 512, 1024, 2048, 3072)


def test_a_small_prefill_that_replays_its_token_graph_counts_as_a_replay(
    model_runner,
):
    """A warm agentic turn (142 new tokens) pads to the 256 bucket and replays
    it; a 16k chunk is past every bucket and runs eagerly. Both are prefills,
    so only the graph column tells them apart."""
    ladder = _c32_ladder(model_runner)
    registry = CollectorRegistry()
    metrics = ForwardDispatchMetrics(registry=registry)
    for tokens in (142, 3072, BUDGET):
        mode = _prefill(tokens, ladder, PREFILL_BUCKETS)
        assert not mode.use_cudagraph
        metrics.record("target", mode.is_prefill, mode.replays_graph)
    small = _prefill(142, ladder, PREFILL_BUCKETS)
    assert small.prefill_token_graph and small.running_tokens == 256
    samples = _samples(registry)
    assert samples[("target", "prefill", "true")] == 2
    assert samples[("target", "prefill", "false")] == 1


def test_without_recorded_prefill_buckets_every_prefill_counts_eager(model_runner):
    ladder = _c32_ladder(model_runner)
    registry = CollectorRegistry()
    metrics = ForwardDispatchMetrics(registry=registry)
    mode = _prefill(142, ladder, ())
    assert mode.running_tokens == 142
    metrics.record("target", mode.is_prefill, mode.replays_graph)
    assert _samples(registry)[("target", "prefill", "false")] == 1
