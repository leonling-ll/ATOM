# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3-Pro prefills replay token graphs.

Pro's builder declares `supports_token_graph`, so `ModelRunner._record_tr_graphs`
records each `ATOM_TOKEN_GRAPH_SIZES` bucket on the step
`build_for_token_graph_capture` builds, and a prefill of up to the largest
bucket pads to its bucket and replays it. Flash's builder declares none.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def Builder():
    with atom_without_aiter() as load:
        yield load(
            "atom.model_ops.attentions.aiter_attention"
        ).AiterAttentionMetadataBuilder


@pytest.mark.parametrize("pro", [True, False])
def test_only_pro_records_prefill_token_graphs(Builder, pro):
    builder = SimpleNamespace(_has_pro_sparse_attention=pro)
    assert Builder.supports_token_graph.fget(builder) is pro


class _Runner:
    """What `build_for_token_graph_capture` touches on the builder, in call
    order: the query layout it publishes and the step it builds on it."""

    def __init__(self):
        self.block_size = 16
        self.calls = []
        self.model_runner = SimpleNamespace(
            forward_vars={
                "input_ids": SimpleNamespace(gpu=torch.zeros(4096, dtype=torch.int32))
            }
        )

    def publish_query_metadata(self, lengths, running_bs, running_tokens):
        self.calls.append(("publish", lengths.tolist(), running_bs, running_tokens))

    def build(self, *, batch, running_bs, running_tokens, max_seqlen_q, is_prefill):
        self.calls.append(("build", running_bs, running_tokens, max_seqlen_q))
        self.batch, self.is_prefill = batch, is_prefill
        return SimpleNamespace(), torch.arange(running_tokens)

    def prepare_model_inputs(self, input_ids, metadata):
        self.calls.append(("inputs", input_ids.shape[0]))


@pytest.mark.parametrize("tokens", [256, 3072])
def test_a_bucket_is_recorded_on_one_dummy_request_of_its_rows(Builder, tokens):
    """The capture step carries no token inputs, so its one request's query
    layout is published before the step is built on it: built on the previous
    capture's layout, the recording asserted on `cu_seqlens_q`."""
    runner = _Runner()
    _, context = Builder.build_for_token_graph_capture(runner, tokens)
    assert runner.calls == [
        ("publish", [tokens], 1, tokens),
        ("build", 1, tokens, tokens),
        ("inputs", tokens),
    ]
    batch = runner.batch
    assert runner.is_prefill and batch.is_dummy_run
    assert batch.total_seqs_num_prefill == 1
    np.testing.assert_array_equal(batch.num_scheduled_tokens, [tokens])
    # Pro's attention skips a dummy step: no cache write during the recording
    assert context.is_dummy_run and context.is_prefill
    assert context.scheduled_tokens == context.running_tokens == tokens
    assert context.positions.shape[0] == tokens
