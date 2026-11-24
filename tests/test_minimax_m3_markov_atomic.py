# SPDX-License-Identifier: MIT
"""The MiniMax-M3.1 / M3-Pro draft's sharded greedy Markov block with one
atomic-max launch a position (`MiniMaxM3DSpark._markov_atomic`), at Pro's TP8
vocab shard and Markov rank: the whole-vocab op's ids (`dspark_markov_argmax`),
ties across ranks to the lowest id, the packed buffers back at their reset
value after each block; and the confidence head run only for the schedule."""

import types

import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="the draft imports the ATOM model stack")
triton = pytest.importorskip("triton")
import triton.language as tl

from atom.models.minimax_m3 import dspark as m3_dspark
from atom.models.minimax_m3.packed_argmax import EMPTY, pack_max_id

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

TP, VOCAB, R, T = 8, 200064, 256, 7
SHARD = VOCAB // TP  # 25008, not a multiple of the V tile


def _pack(val, idx):
    """`pack_max_id` in torch: the word each rank's (max, id) pair packs to."""
    val = torch.where(val == 0, torch.zeros_like(val), val)
    bits = val.contiguous().view(torch.int32)
    ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return (ordered.to(torch.int64) << 32) | (0xFFFFFFFF - idx.to(torch.int64))


class _LockstepTp:
    """One rank's view of a TP group whose other ranks run in lockstep: each
    all-gather of this rank's packed words is answered with every rank's word
    for the same block position, from `dspark_markov_argmax_tp`'s per-shard
    (max, id) over that rank's shard and the ids the block has so far."""

    def __init__(self, rank, shard_logits, anchors, w1, w2):
        self.rank, self.world_size = rank, len(shard_logits)
        self._logits, self._w1, self._w2 = shard_logits, w1, w2
        self._prev, self.calls = anchors.clone(), 0

    def all_gather(self, words, dim, use_custom):
        from atom.model_ops.dspark_markov_argmax import _launch, argmax_gather

        assert dim == 0 and words.dtype == torch.float32 and words.shape[1] == 2
        pairs, packed = [], []
        for j, logits in enumerate(self._logits):
            pair = torch.empty_like(words)
            base = logits[:, self.calls]
            _launch(
                base, self._prev, self._w1, self._w2, torch.empty_like(self._prev),
                j * base.shape[1], pair,
            )  # fmt: skip
            pairs.append(pair)
            packed.append(_pack(pair[:, 0], pair[:, 1].long()))
            if j == self.rank:
                # the rank under test folded its shard to the same word
                assert torch.equal(words.view(torch.int64).view(-1), packed[-1])
        argmax_gather(torch.stack(pairs), self._prev)
        self.calls += 1
        return torch.cat(packed).view(torch.float32).view(-1, 2)


def _draft(rank, shard_w, markov, max_num_seqs=64):
    draft = m3_dspark.MiniMaxM3DSpark.__new__(m3_dspark.MiniMaxM3DSpark)
    nn.Module.__init__(draft)
    draft.lm_head = types.SimpleNamespace(
        weight=shard_w[rank], vocab_start_idx=rank * SHARD, tp_size=TP, bias=None
    )
    draft.markov_head, draft.confidence_head = markov, None
    draft.emits_confidence, draft.max_num_seqs = False, max_num_seqs
    return draft


def _whole_vocab_ids(logits, anchors, w1, w2):
    from atom.model_ops.dspark_markov_argmax import dspark_markov_argmax

    ref = anchors.new_empty(anchors.shape[0], T + 1)
    ref[:, 0] = anchors
    for k in range(T):
        dspark_markov_argmax(logits[:, k], ref[:, k], w1, w2, ref[:, k + 1])
    return ref[:, 1:]


@needs_gpu
@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("bs", [1, 3, 8, 32, 64])
def test_atomic_block_has_the_whole_vocab_ids(monkeypatch, bs, tie):
    """Two blocks in a row on each rank's packed buffers (the second reads
    what the first left), ids equal to the whole-vocab op's. With ``tie``,
    each position's maximum is tied between an id of the first shard and one
    of the last, and the lower id wins."""
    from atom.models.dspark_draft import DSparkMarkovHead

    monkeypatch.setenv("ATOM_DSPARK_FUSED_MARKOV_SAMPLE", "1")
    torch.manual_seed(bs)
    kw = {"dtype": torch.bfloat16, "device": "cuda"}
    hidden_size = 512
    weight = torch.randn(VOCAB, hidden_size, **kw) * 0.05
    markov = DSparkMarkovHead(VOCAB, R).to(**kw)
    with torch.no_grad():
        markov.markov_w1.weight.normal_(std=0.5)
        markov.markov_w2.weight.normal_(std=0.5)
    w1, w2 = markov.markov_w1.weight, markov.markov_w2.weight
    low, high = 12345, (TP - 1) * SHARD + 12345
    if tie:
        with torch.no_grad():
            w2[high] = w2[low]
    shard_w = [weight[j * SHARD : (j + 1) * SHARD] for j in range(TP)]
    drafts = {r: _draft(r, shard_w, markov) for r in (0, 3, TP - 1)}
    for _ in range(2):
        hidden = torch.randn(bs * T, hidden_size, **kw)
        if tie:
            hidden = hidden[:1].expand(bs * T, -1).contiguous()
            with torch.no_grad():
                weight[low] = weight[high] = hidden[0]
        anchors = torch.randint(0, VOCAB, (bs,), dtype=torch.int32, device="cuda")
        # the per-shard GEMM the head runs; the whole vocab is their concat
        shard_logits = [m3_dspark.tgemm.mm(hidden, w).view(bs, T, -1) for w in shard_w]
        ref = _whole_vocab_ids(torch.cat(shard_logits, -1), anchors, w1, w2)
        if tie:
            assert ref.eq(low).all()
        for r, draft in drafts.items():
            group = _LockstepTp(r, shard_logits, anchors, w1, w2)
            monkeypatch.setattr(m3_dspark, "get_tp_group", lambda g=group: g)
            ids, confidence = draft.head_and_sample(hidden, anchors, T)
            assert confidence is None and group.calls == T
            assert torch.equal(ids, ref), r
            assert bool((draft._mk_packed == EMPTY).all())


@triton.jit
def _pack_kernel(val_ptr, idx_ptr, out_ptr):
    i = tl.program_id(0)
    tl.store(out_ptr + i, pack_max_id(tl.load(val_ptr + i), tl.load(idx_ptr + i)))


@needs_gpu
def test_packed_words_order_by_value_then_lowest_id():
    vals = torch.tensor([-0.0, 0.0, -0.0, 0.0, -1.5, 2.0, 2.0, -3.0], device="cuda")
    idx = torch.tensor([5, 5, 7, 7, 1, 9, 4, 0], device="cuda", dtype=torch.int32)
    out = torch.empty(8, dtype=torch.int64, device="cuda")
    _pack_kernel[(8,)](vals, idx, out)
    assert torch.equal(out, _pack(vals, idx))
    # signed zeros are one value; at equal values the lower id packs larger
    assert out[0] == out[1] and out[2] == out[3] and out[0] > out[3]
    assert out[6] > out[5] > out[0] > out[4] > out[7] > EMPTY


class _Confidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden, embeds):
        self.calls += 1
        return hidden.new_ones(hidden.shape[0], hidden.shape[1])


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("schedule", [False, True])
def test_the_confidence_head_runs_only_for_the_schedule(monkeypatch, schedule, sharded):
    """Off the schedule the head is not run and the block is the atomic one
    under TP; on it, the head runs, over the sharded block that keeps the W1
    rows it reads."""
    from atom.models.dspark_draft import DSparkMarkovHead

    monkeypatch.setenv("ATOM_DSPARK_FUSED_MARKOV_SAMPLE", "0")
    bs, vocab, hidden_size = 3, 64, 16
    draft = m3_dspark.MiniMaxM3DSpark.__new__(m3_dspark.MiniMaxM3DSpark)
    nn.Module.__init__(draft)
    draft.markov_head = DSparkMarkovHead(vocab, 8)
    draft.confidence_head = _Confidence()
    draft.emits_confidence = schedule
    head_w = torch.randn(vocab, hidden_size)
    draft.lm_head = lambda x: x @ head_w.t()
    hidden = torch.randn(bs * T, hidden_size)
    anchors = torch.tensor([1, 2, 3])
    taken = []
    monkeypatch.setattr(
        m3_dspark.MiniMaxM3DSpark,
        "_samples_off_the_vocab_shard",
        lambda self, head: sharded,
    )
    monkeypatch.setattr(
        m3_dspark.MiniMaxM3DSpark,
        "_markov_atomic",
        lambda self, *a: taken.append("atomic") or anchors.new_zeros(bs, T),
    )
    monkeypatch.setattr(
        m3_dspark.MiniMaxM3DSpark,
        "_sample_block_off_the_vocab_shard",
        lambda self, *a: taken.append("sharded")
        or (anchors.new_zeros(bs, T), torch.zeros(bs, T, 8)),
    )
    ids, confidence = draft.head_and_sample(hidden, anchors, T)
    assert ids.shape == (bs, T)
    assert (confidence is not None) == schedule
    assert draft.confidence_head.calls == int(schedule)
    if sharded:
        assert taken == ["sharded" if schedule else "atomic"]
    else:
        assert not taken
