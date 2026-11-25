# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3-Pro's forced-acceptance verify head inside the decode forward.

- ``SyntheticVerifyHead`` (``rejection_sampler``): the output ids and verdict of the runner's logits path
  (``index_select`` of the bonus and target rows, the sampler's bonus pick, ``rejection_sample``'s synthetic branch)
  on the same logits and the same draws, eagerly and recorded in a CUDA graph; its draws come from a device step
  counter, so replays draw anew and every rank draws alike.
- ``MiniMaxM3ProForCausalLM.compute_head``: the head serves a decode forward of full draft spans, up to
  ``HEAD_MAX_REQUESTS`` requests, under forced acceptance only.
- ``ModelRunner._head_ids_serve``: a head that draws at T alone (``head_draws_top_p`` False) does not stand for a
  sampled step with a top-p below 1.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter

# ---- the runner (CPU, AITER stubbed)


@pytest.fixture
def runner_module():
    # per test: the stub leaves before the tests that need the real AITER
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner")


def _runner(runner_module, head_draws_top_p):
    runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
    runner.model = SimpleNamespace()
    if head_draws_top_p is not None:
        runner.model.head_draws_top_p = head_draws_top_p
    runner.rejection_sampler = SimpleNamespace(synthetic_acceptance_rates=(1.0, 0.5))
    return runner


def _batch(temps, top_ps):
    n = len(temps)
    return SimpleNamespace(
        total_seqs_num=n,
        temperatures=np.asarray(temps, dtype=np.float32),
        top_ks=np.full(n, -1, dtype=np.int32),
        top_ps=np.asarray(top_ps, dtype=np.float32),
        any_logprobs=False,
        structured_requests={},
        required_channel=None,
        needs_independent_noise=np.zeros(n, dtype=bool),
    )


@pytest.fixture
def spec_step(monkeypatch, runner_module):
    """A verify step of two requests, each verifying the full span of 2 drafts."""
    spec = SimpleNamespace(num_spec_steps=2, num_draft_tokens_np=np.full(2, 2))
    monkeypatch.setattr(
        runner_module,
        "get_forward_context",
        lambda: SimpleNamespace(spec_decode_metadata=spec),
    )
    monkeypatch.setattr(runner_module, "RELAXED_TOP_N", 1)
    monkeypatch.setattr(runner_module, "get_pcp_world_size", lambda: 1)


def _head(requests=2, rows=3):
    return (
        torch.zeros(requests, rows, dtype=torch.int32),
        torch.zeros(3, requests, dtype=torch.int32),
    )


@pytest.mark.parametrize("head_draws_top_p", [None, True, False])
def test_a_head_that_draws_at_t_alone_serves_no_top_p_below_1(
    runner_module, spec_step, head_draws_top_p
):
    runner = _runner(runner_module, head_draws_top_p)
    # sampled at T, every top-p 1 (the agentic proxy's requests): every head
    assert runner._head_ids_serve(_head(), _batch((1.0, 0.7), (1.0, 1.0)))
    # greedy: the argmax whatever the top-p
    assert runner._head_ids_serve(_head(), _batch((0.0, 0.0), (0.5, 1.0)))
    # a top-p below 1 on a sampled row: only a head that draws in the nucleus
    nucleus = _batch((1.0, 0.7), (1.0, 0.9))
    assert runner._head_ids_serve(_head(), nucleus) is (head_draws_top_p is not False)


# ---- the head (AITER, Triton, a GPU)


def _gpu_rejection_sampler():
    pytest.importorskip("aiter")
    pytest.importorskip("triton")
    if not torch.cuda.is_available():
        pytest.skip("the verify head's kernels need a GPU")
    from atom.model_ops import rejection_sampler

    return rejection_sampler


def _draws(rs, step, requests, k, vocab):
    """The head's draws at ``step``: bonus noise [requests, vocab] and the verdict's uniforms [requests * k]."""
    import triton

    counter = torch.tensor([step], dtype=torch.int64, device="cuda")
    noise = torch.empty((requests, vocab), dtype=torch.float32, device="cuda")
    rs._head_noise_kernel[(requests, triton.cdiv(vocab, rs._HEAD_NOISE_BLOCK))](
        noise, counter, vocab, rs._HEAD_NOISE_BASE_SEED, BLOCK=rs._HEAD_NOISE_BLOCK
    )
    uniform = torch.empty(requests * k, dtype=torch.float32, device="cuda")
    rs._head_uniform_kernel[(1,)](
        uniform,
        counter,
        requests * k,
        rs._SYNTHETIC_RNG_BASE_SEED,
        BLOCK=triton.next_power_of_2(requests * k),
    )
    assert counter.item() == step + 1
    return noise, uniform


def _logits_path(rs, logits, input_ids, temps, k, rates, noise, uniform, requests):
    """The runner's logits path (`postprocess`) on the first ``requests`` requests of ``logits``, every one verifying
    the full span, with the bonus race on ``noise`` (the sampler's independent-noise draw) and the synthetic verdict
    on ``uniform`` (in place of its host-seeded generator's)."""
    from aiter.ops.triton.fusions.fused_mixed_sample import fused_mixed_sample

    from atom.model_ops.sampler import SAMPLER_EPS, Sampler

    rows = np.arange(requests * (k + 1)).reshape(requests, k + 1)
    target_rows = torch.tensor(rows[:, :k].ravel(), dtype=torch.int32, device="cuda")
    bonus_rows = torch.tensor(rows[:, k], dtype=torch.int32, device="cuda")
    cu = torch.tensor(np.arange(1, requests + 1) * k, dtype=torch.int32, device="cuda")
    bonus_logits = torch.index_select(logits, 0, bonus_rows)
    target_logits = torch.index_select(logits, 0, target_rows)
    temps = temps[:requests]
    if bool((temps <= SAMPLER_EPS).all()):
        bonus = Sampler()(bonus_logits, temps, None, None, all_greedy=True)
    else:
        bonus = fused_mixed_sample(
            bonus_logits,
            noise[:requests],
            temps,
            eps=SAMPLER_EPS,
            greedy_eps=SAMPLER_EPS,
        )
    real_rand = torch.rand

    def rand(n, **kw):
        assert n == requests * k and kw.get("generator") is not None
        return uniform[:n].clone()

    torch.rand = rand
    try:
        return rs.rejection_sample(
            input_ids,
            target_rows,
            k,
            cu,
            None,
            target_logits,
            bonus.view(-1, 1).contiguous(),
            bonus_rows,
            synthetic_acceptance_rates=tuple(rates),
            synthetic_step=0,
        )
    finally:
        torch.rand = real_rand


def _inputs(requests, k, vocab, temp, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    rows = requests * (k + 1)
    logits = torch.randn(rows, vocab, generator=g, device="cuda").to(torch.bfloat16)
    input_ids = torch.randint(0, vocab, (rows + 8,), generator=g, device="cuda").to(
        torch.int32
    )
    from atom.model_ops.sampler import SAMPLER_EPS

    temps = torch.full((requests,), max(temp, SAMPLER_EPS), device="cuda")
    return logits, input_ids, temps


_VOCAB = 200064  # Pro's


@pytest.mark.parametrize("k, length", [(3, 2.53), (7, 2.94), (7, 5.5)])
@pytest.mark.parametrize("requests, bs", [(1, 1), (2, 2), (3, 3), (16, 16), (4, 2)])
@pytest.mark.parametrize("temp", [0.0, 1.0, 0.7])
def test_the_head_is_the_logits_path(requests, bs, k, length, temp):
    """Ids and verdict ``torch.equal`` to the logits path's on the same draws: greedy (today's argmax bonus), sampled
    (the sampler's race); a padded width's first requests are the step's."""
    rs = _gpu_rejection_sampler()
    rates = rs.acceptance_length_to_rates(length, k)
    logits, input_ids, temps = _inputs(requests, k, _VOCAB, temp, requests * 10 + k)
    head = rs.SyntheticVerifyHead(k, rates, "cuda")
    head.step.fill_(5)
    ids, verdict = head(logits, input_ids, temps)
    noise, uniform = _draws(rs, 5, requests, k, _VOCAB)
    want_ids, want_verdict = _logits_path(
        rs, logits, input_ids, temps, k, rates, noise, uniform, bs
    )
    assert torch.equal(ids[:bs], want_ids)
    assert torch.equal(verdict[:, :bs], want_verdict)
    assert head.step.item() == 6


def test_a_recorded_head_draws_anew_on_every_replay():
    """Recorded in a CUDA graph, each replay is the logits path at the next step: the counter, not the capture,
    seeds the draws."""
    rs = _gpu_rejection_sampler()
    k, requests = 7, 3
    rates = rs.acceptance_length_to_rates(4.5, k)
    logits, input_ids, temps = _inputs(requests, k, _VOCAB, 1.0, 1)
    head = rs.SyntheticVerifyHead(k, rates, "cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        head(logits, input_ids, temps)  # warmup: step 0
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids, verdict = head(logits, input_ids, temps)
    seen = []
    for step in (1, 2, 3):
        fresh, _, _ = _inputs(requests, k, _VOCAB, 1.0, 100 + step)
        logits.copy_(fresh)
        graph.replay()
        noise, uniform = _draws(rs, step, requests, k, _VOCAB)
        want = _logits_path(
            rs, logits, input_ids, temps, k, rates, noise, uniform, requests
        )
        assert torch.equal(ids, want[0]) and torch.equal(verdict, want[1])
        seen.append(uniform)
    assert head.step.item() == 4
    assert not torch.equal(seen[0], seen[1])


def test_every_rank_draws_alike():
    """Eight heads (one a TP rank) on the same logits give the same ids and verdict at every step: the verdict and
    the bonus agree across ranks without a broadcast."""
    rs = _gpu_rejection_sampler()
    k, requests = 7, 4
    rates = rs.acceptance_length_to_rates(2.94, k)
    logits, input_ids, temps = _inputs(requests, k, _VOCAB, 1.0, 2)
    heads = [rs.SyntheticVerifyHead(k, rates, "cuda") for _ in range(8)]
    for _ in range(3):
        outs = [h(logits, input_ids, temps) for h in heads]
        for ids, verdict in outs[1:]:
            assert torch.equal(ids, outs[0][0]) and torch.equal(verdict, outs[0][1])


@pytest.mark.parametrize("k, length", [(7, 2.94), (3, 2.53)])
def test_the_draws_keep_the_forced_acceptance_length(k, length):
    """Over many steps the accepted drafts average ``length - 1`` (the golden AL's schedule), the uniforms are U(0,
    1) and the bonus noise Exp(1)."""
    rs = _gpu_rejection_sampler()
    rates = rs.acceptance_length_to_rates(length, k)
    requests, steps = 16, 400
    logits, input_ids, temps = _inputs(requests, k, 4096, 1.0, 3)
    head = rs.SyntheticVerifyHead(k, rates, "cuda")
    accepted = torch.stack([head(logits, input_ids, temps)[1][0] for _ in range(steps)])
    assert abs(accepted.float().mean().item() - (length - 1)) < 0.025
    noise, uniform = _draws(rs, 0, 64, 16, 4096)
    assert abs(uniform.mean().item() - 0.5) < 0.05
    assert abs(noise.mean().item() - 1.0) < 0.01 and noise.min().item() >= 0


# ---- the model's head (AITER for the model stack)


class _Head:
    """A bound ``SyntheticVerifyHead`` as ``compute_head`` calls it."""

    num_spec_steps = 7

    def __init__(self):
        self.calls = []

    def __call__(self, logits, input_ids, temperatures):
        self.calls.append((logits, input_ids, temperatures))
        return "ids", "verdict"


@pytest.fixture
def pro_head(monkeypatch):
    pytest.importorskip("aiter", reason="pro imports the ATOM model stack")
    from atom.models.minimax_m3 import pro

    head = _Head()
    model = SimpleNamespace(
        _verify_head=head,
        _head_input_ids=torch.arange(400),
        _head_temperatures=torch.ones(32),
        compute_logits=lambda hidden: hidden + 1,
    )

    def run(rows, running_bs, is_prefill=False, is_draft=False):
        context = SimpleNamespace(
            running_bs=running_bs, is_prefill=is_prefill, is_draft=is_draft
        )
        monkeypatch.setattr(
            pro, "get_forward_context", lambda: SimpleNamespace(context=context)
        )
        return pro.MiniMaxM3ProForCausalLM.compute_head(model, torch.zeros(rows, 4))

    return pro, model, run, head.calls


def test_the_model_verifies_a_decode_of_full_spans_up_to_16_requests(pro_head):
    pro, model, run, calls = pro_head
    assert pro.HEAD_MAX_REQUESTS == 16
    assert run(8, 1) == ("ids", "verdict")
    logits, input_ids, temps = calls[-1]
    assert torch.equal(logits, torch.ones(8, 4)) and input_ids.shape == (8,)
    assert temps.shape == (1,)
    assert run(128, 16) == ("ids", "verdict")
    # wider than the small-concurrency range, a query bucket shorter than k + 1, a prefill, a draft: the logits
    for rows, bs, over in [
        (136, 17, {}),
        (16, 4, {}),
        (8, 1, {"is_prefill": True}),
        (8, 1, {"is_draft": True}),
    ]:
        assert run(rows, bs, **over) is None
    assert len(calls) == 2
    # real acceptance binds no head
    model._verify_head = None
    assert run(8, 1) is None


def test_the_model_binds_a_head_under_forced_acceptance_only(pro_head):
    pro = pro_head[0]
    model = SimpleNamespace(_verify_head=None)
    forward_vars = {
        "input_ids": SimpleNamespace(gpu=torch.zeros(64, dtype=torch.int32)),
        "temperatures": SimpleNamespace(gpu=torch.ones(16)),
    }
    pro.MiniMaxM3ProForCausalLM.bind_verify_head(model, forward_vars, 7, None)
    assert model._verify_head is None
    pro.MiniMaxM3ProForCausalLM.bind_verify_head(model, forward_vars, 7, [1.0, 0.5])
    assert model._verify_head.num_spec_steps == 7
    assert model._head_temperatures is forward_vars["temperatures"].gpu
    assert pro.MiniMaxM3ProForConditionalGenerationVL.head_draws_top_p is False
