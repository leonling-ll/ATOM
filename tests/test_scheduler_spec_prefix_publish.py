# SPDX-License-Identifier: MIT
"""Decode blocks reach the prefix cache only once every token in them has its final KV.

A speculative step (DSpark, k = 7) forwards the last committed token (the anchor) and seven
drafts, writing KV for all eight positions; the drafts it rejects leave KV that no token
owns, and the token it samples last (the bonus) has no KV until the next forward carries
it as its anchor. Deferred output (the default) runs that next forward before the scheduler
sees this step's tokens; without it (pipeline parallel) the scheduler sees them first. Here
the runner is a model of exactly that: what each forward writes, and which ids are final.
Every block the scheduler publishes must hold final KV of final ids, and a second turn
(prompt + answer + more) must hit exactly the blocks that were published.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MockConfig

from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

K = 7
BS = 16
MAX_TOKENS = 64


class Runner:
    """The device side of one sequence: the KV each forward writes, the ids that are final
    (``truth``), and what each forward hands back."""

    def __init__(self, prompt, accepts):
        self.truth = list(prompt)
        self.kv = {}
        self.accepts = accepts
        self.step = 0
        self.fresh = 70_000

    def _new(self):
        self.fresh += 1
        return self.fresh

    def prefill(self):
        for i, t in enumerate(self.truth):
            self.kv[i] = t
        first = self._new()
        self.truth.append(first)  # sampled, no KV yet
        return [first], 0

    def verify(self):
        """The anchor (the last final id) and K drafts at their positions; ``a`` of the
        drafts accepted, then the bonus."""
        anchor = len(self.truth) - 1
        drafts = [self._new() for _ in range(K)]
        self.kv[anchor] = self.truth[anchor]
        for i, d in enumerate(drafts, 1):
            self.kv[anchor + i] = d
        a = self.accepts[self.step % len(self.accepts)]
        self.step += 1
        out = drafts[:a] + [self._new()]
        self.truth += out
        return out, K - a

    def final_upto(self, n):
        """Whether positions ``[0, n)`` hold final KV of final ids."""
        return all(self.kv.get(i) == self.truth[i] for i in range(n))


def _scheduler(deferred):
    sched = Scheduler(
        MockConfig(
            speculative_config=SimpleNamespace(
                num_speculative_tokens=K, use_dspark=lambda: True
            ),
            num_kvcache_blocks=64,
            kv_cache_block_size=BS,
            enable_prefix_caching=True,
            max_model_len=512,
            max_num_batched_tokens=512,
            # undeferred output is what pipeline parallelism runs
            pipeline_parallel_size=1 if deferred else 2,
        )
    )
    assert sched.is_deferred_out == deferred
    return sched


def _output(seq, out, deferred):
    tokens, rejected = out
    return ScheduledBatchOutput(
        [seq.id],
        [tuple(tokens)],
        np.array([rejected]),
        np.array([len(tokens) - 1]),
        np.zeros((1, K), dtype=np.int32),
        is_deferred_out=deferred,
    )


@pytest.mark.parametrize("deferred", [True, False], ids=["deferred", "undeferred"])
@pytest.mark.parametrize("accepts", [[0], [3], [7], [7, 0, 3, 7, 3, 0, 7]])
@pytest.mark.parametrize("prompt_len", [40, 47])
def test_published_decode_blocks_hold_final_kv(
    prompt_len, accepts, deferred, monkeypatch
):
    sched = _scheduler(deferred)
    bm = sched.block_manager
    prompt = list(range(1000, 1000 + prompt_len))
    seq = Sequence(
        prompt,
        BS,
        SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True),
        num_draft_tokens=K,
    )
    run = Runner(prompt, accepts)
    published = []

    def check():
        n = seq.num_hashed_tokens
        assert list(seq.token_ids[:n]) == run.truth[:n], "a block of unfinal ids"
        assert run.final_upto(n), f"published up to {n} over KV not yet final"

    real = bm.hash_decode_blocks

    def spy(s, *a, **kw):
        real(s, *a, **kw)
        check()
        published.append(s.num_hashed_tokens)

    monkeypatch.setattr(bm, "hash_decode_blocks", spy)
    sched.add(seq)
    batch, active = sched.schedule()
    assert batch.total_seqs_num_prefill == 1
    pending = run.prefill()
    if deferred:
        # the first output surfaces one step late
        empty = ScheduledBatchOutput([], [], None, None, None, is_deferred_out=True)
        sched.postprocess(list(active.values()), empty, batch=batch)
    else:
        sched.postprocess(
            list(active.values()), _output(seq, pending, False), batch=batch
        )
    check()
    while not seq.is_finished:
        batch, active = sched.schedule()
        assert seq.id in active
        out = run.verify()
        if deferred:
            # this forward ran before the scheduler reads the previous one's tokens
            out, pending = pending, out
        sched.postprocess(
            list(active.values()), _output(seq, out, deferred), batch=batch
        )
        if not seq.is_finished:
            check()
            # every whole block below the last final id is out
            assert seq.num_hashed_tokens >= (seq.num_finalized_tokens - 1) // BS * BS
    assert published, "no decode step published"
    n = seq.num_prompt_tokens + MAX_TOKENS
    assert run.final_upto(published[-1]) and published[-1] > seq.num_prompt_tokens

    # turn 2: the prompt, the answer and a new question hit what was published, no more
    turn2 = Sequence(run.truth[:n] + list(range(5000, 5000 + 3 * BS)), BS)
    assert bm.can_allocate(turn2) == published[-1] // BS
