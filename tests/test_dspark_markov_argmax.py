# SPDX-License-Identifier: MIT
"""The Markov sampling op's contract: the destination, and the row it biased by.

The W1 gather lives inside stage 1 -- every tile program reads those rows to
form the bias, so a separate `index_select` would be a launch that reads what
the kernel is about to read anyway. That makes the gathered row the op's only
return value, and the confidence head downstream consumes it, so it has to be
the same row the bias was formed from rather than merely a plausible one.

The ids leave through `out` and nothing else. `out` is required for that
reason: `torch.compile` may functionalize the call, and the handle it would
hand back then is a copy, so a returned id tensor could not be trusted.
"""

import pytest
import torch

pytest.importorskip("aiter", reason="the Markov sampler is a Triton/AITER op")

from atom.model_ops.dspark_markov_argmax import (
    _torch_dspark_markov_argmax,
    dspark_markov_argmax,
)


def _tables(rows, vocab, rank, device, seed=1201):
    torch.manual_seed(seed)
    kwargs = {"dtype": torch.bfloat16, "device": device}
    return (
        torch.randn(rows, vocab, **kwargs),
        torch.randint(0, vocab, (rows,), dtype=torch.int64, device=device),
        torch.randn(vocab, rank, **kwargs),
        torch.randn(vocab, rank, **kwargs),
    )


def _ids(rows, device, dtype=torch.int64):
    return torch.empty(rows, dtype=dtype, device=device)


def test_torch_body_returns_the_row_it_gathered():
    """The fallback every non-CUDA caller takes, and CI's only path here."""
    base, prev, w1, w2 = _tables(5, 64, 16, "cpu")
    ids = _ids(5, "cpu")
    embed = _torch_dspark_markov_argmax(base, prev, w1, w2, ids)
    assert torch.equal(embed, w1[prev])
    assert torch.equal(ids, (base + w1[prev].float() @ w2.float().t()).argmax(-1))


def test_shapes_are_checked_against_each_other():
    base, prev, w1, w2 = _tables(5, 64, 16, "cpu")
    ids = _ids(5, "cpu")
    with pytest.raises(ValueError, match="prev_ids"):
        dspark_markov_argmax(base, prev[:-1], w1, w2, ids)
    with pytest.raises(ValueError, match="markov_w1"):
        dspark_markov_argmax(base, prev, w1[:, :-1], w2, ids)
    with pytest.raises(ValueError, match="markov_w2"):
        dspark_markov_argmax(base, prev, w1, w2[:-1], ids)
    embed = dspark_markov_argmax(base[:0], prev[:0], w1, w2, ids[:0])
    assert embed.shape == (0, 16)


def test_ids_are_rejected_at_both_ends_unless_an_id_block_could_hold_them():
    """`prev_ids` indexes a table and `out` is indexed by one, so both are ids.

    A float `prev_ids` would gather by a reinterpreted address in the kernel
    and by an index error in the fallback -- two failures for one mistake, and
    only one of them loud.
    """
    base, prev, w1, w2 = _tables(5, 64, 16, "cpu")
    with pytest.raises(ValueError, match="prev_ids"):
        dspark_markov_argmax(base, prev.float(), w1, w2, _ids(5, "cpu"))
    with pytest.raises(ValueError, match="out"):
        dspark_markov_argmax(base, prev, w1, w2, _ids(4, "cpu"))
    with pytest.raises(ValueError, match="out"):
        dspark_markov_argmax(base, prev, w1, w2, _ids(5, "cpu", torch.float32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_fused_gathers_the_same_row_it_biases_with():
    """Fused against the torch body: the ids, and the row bit for bit.

    The row is the strict half -- a gather is exact, so anything but equality
    means stage 1 published a different row than it biased with, and the
    confidence head would drift with nothing pointing back here. The ids are
    equal up to accumulation order (see the op's module docstring); a tie
    closer than that would be the one thing this could not distinguish, which
    random logits make vanishingly unlikely.
    """
    base, prev, w1, w2 = _tables(7, 2048, 128, "cuda")
    ids, ref_ids = _ids(7, "cuda"), _ids(7, "cuda")
    embed = dspark_markov_argmax(base, prev, w1, w2, ids)
    ref_embed = _torch_dspark_markov_argmax(base, prev, w1, w2, ref_ids)
    # Armed: a batch whose rows all shared a previous token would pass with
    # the gather reading any single row.
    assert len(set(prev.tolist())) > 1
    assert torch.equal(embed, ref_embed)
    assert torch.equal(embed, w1[prev])
    assert torch.equal(ids, ref_ids)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_fused_reads_and_writes_the_strided_columns_it_is_given(dtype):
    """The production spelling: both ids are columns of one `[B, T + 1]` block.

    `out_ids[:, k]` carries stride `T + 1`, so a kernel that assumed unit
    stride would bias each row with another position's token on the way in and
    land each id in another position's slot on the way out -- in bounds, never
    a fault, and visible only as a draft that gets rejected more often.

    int32 is the one that matters: a draft's id block comes from
    `anchor_ids.new_empty`, so the store is narrowing as well as moving, and a
    kernel that only stored int64 would corrupt the neighbouring column.
    """
    torch.manual_seed(5)
    rows, width, vocab, rank = 6, 5, 2048, 128
    out_ids = torch.randint(0, vocab, (rows, width + 1), dtype=dtype, device="cuda")
    base = torch.randn(rows, vocab, dtype=torch.bfloat16, device="cuda")
    w1 = torch.randn(vocab, rank, dtype=torch.bfloat16, device="cuda")
    w2 = torch.randn(vocab, rank, dtype=torch.bfloat16, device="cuda")
    for k in range(width):
        prev = out_ids[:, k]
        # Armed: a contiguous column would make this the test above again.
        assert not prev.is_contiguous() and prev.stride(0) == width + 1
        ref_ids = _ids(rows, "cuda", dtype)
        ref_embed = _torch_dspark_markov_argmax(base, prev, w1, w2, ref_ids)
        embed = dspark_markov_argmax(base, prev, w1, w2, out_ids[:, k + 1])
        assert torch.equal(embed, ref_embed), k
        assert torch.equal(out_ids[:, k + 1], ref_ids), k


class _TpGroup:
    """`dspark_markov_argmax_tp`'s group, over shards run one after another:
    `all_gather` returns every shard's `(max, id)` pack, rank order."""

    def __init__(self, packs):
        self.world_size = len(packs)
        self._packs = packs

    def all_gather(self, packed, dim, use_custom):
        assert dim == 0
        return torch.cat(self._packs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("tp", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 32, 160])
def test_sharded_sampling_matches_the_whole_vocab(tp, rows):
    """Each rank's `(max, id)` over its vocab shard, reduced across ranks, is
    the whole-vocab op's id and gathered row, bit for bit -- including a
    maximum tied across two shards, where the lower id must win."""
    from atom.model_ops.dspark_markov_argmax import _launch, dspark_markov_argmax_tp

    vocab, rank = 129280, 256
    base, prev, w1, w2 = _tables(rows, vocab, rank, "cuda")
    w1, w2 = w1 * 0.05, w2 * 0.05
    # an exact tie between ids in the first and last shard: same base logit,
    # same Markov column, so the same biased logit -- the lower id must win
    base[0, 1000] = base[0, vocab - 7] = 80.0
    w2[vocab - 7] = w2[1000]
    ref = _ids(rows, "cuda", torch.int32)
    ref_embed = dspark_markov_argmax(base, prev, w1, w2, ref)
    assert ref[0].item() == 1000
    shard = vocab // tp
    packs = []
    for r in range(tp):
        packed = torch.empty(rows, 2, dtype=torch.float32, device="cuda")
        _launch(
            base[:, r * shard : (r + 1) * shard],
            prev,
            w1,
            w2,
            ref,
            r * shard,
            packed,
        )
        packs.append(packed)
    for r in range(tp):
        out = _ids(rows, "cuda", torch.int32)
        embed = dspark_markov_argmax_tp(
            base[:, r * shard : (r + 1) * shard],
            prev,
            w1,
            w2,
            out,
            r * shard,
            _TpGroup(packs),
        )
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        torch.testing.assert_close(embed, ref_embed, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("tp", [1, 4])
def test_an_all_nan_row_drafts_id_0_at_a_vocab_off_the_tile_width(tp):
    """An all-NaN row drafts id 0, as `torch.argmax` does, also where the
    (shard) vocab is not a multiple of the V tile, so its tail tile has no
    hit: one GPU at V = 129 * 128 + 64, and four shards of that width (V4.1
    at TP4 has 32320, %128 = 64). Other rows are unaffected."""
    from atom.model_ops.dspark_markov_argmax import _launch, dspark_markov_argmax_tp

    shard, rows, rank = 129 * 128 + 64, 3, 32
    vocab = shard * tp
    base, prev, w1, w2 = _tables(rows, vocab, rank, "cuda")
    w1, w2 = w1 * 0.05, w2 * 0.05
    base[1] = float("nan")
    ref = _ids(rows, "cuda", torch.int32)
    dspark_markov_argmax(base[[0, 2]], prev[[0, 2]], w1, w2, ref[:2])
    if tp == 1:
        out = _ids(rows, "cuda", torch.int32)
        dspark_markov_argmax(base, prev, w1, w2, out)
    else:
        packs = []
        for r in range(tp):
            packed = torch.empty(rows, 2, dtype=torch.float32, device="cuda")
            _launch(
                base[:, r * shard : (r + 1) * shard],
                prev,
                w1,
                w2,
                ref,
                r * shard,
                packed,
            )
            packs.append(packed)
        out = _ids(rows, "cuda", torch.int32)
        dspark_markov_argmax_tp(base[:, :shard], prev, w1, w2, out, 0, _TpGroup(packs))
    assert out[1].item() == 0
    assert out[0].item() == ref[0].item() and out[2].item() == ref[1].item()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("num_j", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 32, 160, 2048])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_argmax_gather_is_the_three_ops_it_replaces(num_j, rows, dtype):
    """`argmax_gather` against `argmax` over j, the id gather at the winner
    and the copy -- with ties across j and NaN values, written into a strided
    integer column."""
    from atom.model_ops.dspark_markov_argmax import argmax_gather

    g = torch.Generator(device="cuda").manual_seed(rows * num_j)
    vals = torch.randint(0, 4, (num_j, rows), device="cuda", generator=g).float()
    ids = torch.randint(0, 129280, (num_j, rows), device="cuda", generator=g).float()
    vals[torch.rand(num_j, rows, device="cuda", generator=g) < 0.1] = float("nan")
    pairs = torch.stack((vals, ids), dim=-1).contiguous()
    winner = pairs[:, :, 0].argmax(dim=0)
    ref = pairs[:, :, 1].gather(0, winner.unsqueeze(0)).squeeze(0).to(dtype)

    block = torch.full((rows, 3), -1, dtype=dtype, device="cuda")
    argmax_gather(pairs, block[:, 1])
    assert torch.equal(block[:, 1], ref)
    # the neighbouring columns are untouched
    assert torch.all(block[:, 0] == -1) and torch.all(block[:, 2] == -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("rows", [1, 5, 160])
def test_confidence_head_matches_the_fp32_reference(rows):
    """The head's GPU path (concat, BF16 GEMV with FP32 output, sigmoid)
    against the FP32 reference it replaces, on a [B, T, H] strided hidden."""
    from atom.models.deepseek_v4_dspark import DSparkConfidenceHead

    hidden_size, rank = 5120, 256
    head = DSparkConfidenceHead(hidden_size, rank).cuda().to(torch.bfloat16)
    with torch.no_grad():
        head.proj.weight.normal_(std=0.02)
    g = torch.Generator(device="cuda").manual_seed(rows)
    big = torch.randn(rows, 5, 2, hidden_size, device="cuda", generator=g)
    hidden = big.to(torch.bfloat16)[:, :, 1]
    embeds = torch.randn(rows, 5, rank, device="cuda", generator=g).bfloat16()
    ref = torch.sigmoid(
        torch.nn.functional.linear(
            torch.cat([hidden, embeds], dim=-1).float(), head.proj.weight.float()
        ).squeeze(-1)
    )
    out = head(hidden, embeds)
    assert out.dtype == torch.float32 and out.shape == (rows, 5)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_sharded_step_writes_its_embed_column_in_place():
    """`embed_out` receives the step's W1[x] rows: a [B, T, r] block filled
    column by column equals the stack of fresh results."""
    from atom.model_ops.dspark_markov_argmax import dspark_markov_argmax_tp

    rows, steps, vocab, rank, tp = 32, 5, 129280, 256, 2
    base, prev, w1, w2 = _tables(rows, vocab, rank, "cuda")
    shard = vocab // tp
    block = w1.new_empty(rows, steps, rank)
    fresh = []
    for k in range(steps):
        packed = torch.empty(rows, 2, dtype=torch.float32, device="cuda")
        from atom.model_ops.dspark_markov_argmax import _launch

        _launch(
            base[:, :shard], prev, w1, w2, _ids(rows, "cuda", torch.int32), 0, packed
        )
        group = _TpGroup([packed, packed])
        fresh.append(
            dspark_markov_argmax_tp(
                base[:, :shard], prev, w1, w2, _ids(rows, "cuda", torch.int32), 0, group
            )
        )
        dspark_markov_argmax_tp(
            base[:, :shard],
            prev,
            w1,
            w2,
            _ids(rows, "cuda", torch.int32),
            0,
            group,
            embed_out=block[:, k],
        )
    assert torch.equal(block, torch.stack(fresh, dim=1))


class _LockstepTp:
    """One rank's view of a TP group whose other ranks run in lockstep: each
    `all_gather` of this rank's `(max, id)` pack is answered with every other
    rank's pack for the same block position, computed off their own LM-head
    shards from the ids the block has gathered so far."""

    def __init__(self, rank, shard_logits, anchors, w1, w2):
        self.rank, self.world_size = rank, len(shard_logits)
        self._logits, self._w1, self._w2 = shard_logits, w1, w2
        self._prev, self.calls = anchors.clone(), 0

    def all_gather(self, packed, dim, use_custom):
        from atom.model_ops.dspark_markov_argmax import _launch, argmax_gather

        assert dim == 0
        packs = []
        for j, logits in enumerate(self._logits):
            pack = torch.empty_like(packed)
            scratch = torch.empty_like(self._prev)
            base = logits[:, self.calls]
            _launch(
                base, self._prev, self._w1, self._w2, scratch, j * base.shape[1], pack
            )
            if j == self.rank:
                # the rank under test reduced its own shard to the same pair
                assert torch.equal(packed, pack), self.calls
            packs.append(pack)
        gathered = torch.cat(packs)
        argmax_gather(gathered.view(self.world_size, -1, 2), self._prev)
        self.calls += 1
        return gathered


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("bs", [1, 4])
def test_m3_draft_samples_its_block_off_the_vocab_shard(monkeypatch, bs, tie):
    """MiniMax-M3.1 / M3-Pro's draft under TP8 at Pro's vocab and Markov rank:
    the block sampled off each rank's LM-head shard (no logits all-gather) has
    the whole-vocab head's ids and confidences, bit for bit. The whole-vocab
    head's logits are the all-gather of the same per-shard GEMMs, as
    `ParallelLMHead.forward` produces them; it is what a head of TP 1 runs.
    With ``tie``, every position's maximum is tied between an id of the first
    shard and one of the last, and the lower id wins."""
    from aiter.dist import parallel_state
    from aiter.tuned_gemm import tgemm
    from torch import nn

    from atom.models.dspark_draft import DSparkConfidenceHead, DSparkMarkovHead
    from atom.models.minimax_m3 import dspark as m3_dspark

    tp, vocab, rank, hidden_size, T = 8, 200064, 256, 1024, 7
    shard = vocab // tp  # 25008: not a multiple of the V tile
    torch.manual_seed(bs)
    kw = {"dtype": torch.bfloat16, "device": "cuda"}
    weight = torch.randn(vocab, hidden_size, **kw) * 0.05
    hidden = torch.randn(bs * T, hidden_size, **kw)
    anchors = torch.randint(0, vocab, (bs,), dtype=torch.int32, device="cuda")
    monkeypatch.setenv("ATOM_DSPARK_FUSED_MARKOV_SAMPLE", "1")
    markov = DSparkMarkovHead(vocab, rank).to(**kw)
    confidence = DSparkConfidenceHead(hidden_size, rank, bias=True).to(**kw)
    with torch.no_grad():
        markov.markov_w1.weight.normal_(std=0.5)
        markov.markov_w2.weight.normal_(std=0.5)
        confidence.proj.weight.normal_(std=0.02)
        confidence.proj.bias.normal_()
    w1, w2 = markov.markov_w1.weight, markov.markov_w2.weight
    if tie:
        # One hidden row everywhere, and two vocab rows, at the same column of
        # the first and the last shard, that copy it: their logits, the same
        # GEMM at the same column, dwarf every other, and their Markov rows
        # are equal, so each position's maximum is a tie between them.
        low = 12345
        high = (tp - 1) * shard + low
        hidden = hidden[:1].expand(bs * T, -1).contiguous()
        with torch.no_grad():
            weight[low] = weight[high] = hidden[0]
            w2[high] = w2[low]
    shard_w = [weight[j * shard : (j + 1) * shard] for j in range(tp)]
    shard_logits = [tgemm.mm(hidden, w).view(bs, T, -1) for w in shard_w]

    class _Head:
        """A rank's `ParallelLMHead`: its shard, and the gathered logits."""

        bias = None

        def __init__(self, r, tp_size=tp):
            self.weight, self.vocab_start_idx = shard_w[r], r * shard
            self.tp_size = tp_size

        def __call__(self, x):
            return torch.cat([tgemm.mm(x, w) for w in shard_w], dim=-1)

    def draft_on(head):
        draft = m3_dspark.MiniMaxM3DSpark.__new__(m3_dspark.MiniMaxM3DSpark)
        nn.Module.__init__(draft)
        draft.lm_head = head
        draft.markov_head, draft.confidence_head = markov, confidence
        return draft

    group = _LockstepTp(0, shard_logits, anchors, w1, w2)
    monkeypatch.setattr(parallel_state, "get_tp_group", lambda: group)
    ref_ids, ref_conf = draft_on(_Head(0, tp_size=1)).head_and_sample(
        hidden, anchors, T
    )
    assert group.calls == 0  # the whole-vocab head: nothing crossed ranks
    if tie:
        assert torch.equal(shard_logits[0][..., low], shard_logits[-1][..., low])
        assert ref_ids.eq(low).all()
    for r in (0, 3, tp - 1):
        group = _LockstepTp(r, shard_logits, anchors, w1, w2)
        ids, conf = draft_on(_Head(r)).head_and_sample(hidden, anchors, T)
        assert group.calls == T
        assert torch.equal(ids, ref_ids), r
        assert torch.equal(conf, ref_conf), r
