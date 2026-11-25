# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro under decode context parallelism, against the single-rank kernel path.

The ranks of a DCP group are simulated in one process: a global paged cache is split
into per-rank caches the way DCP stores them (a whole 16-token block on rank ``j % W``,
at block-table entry ``j // W``), and the collectives are concatenations. Each rank's
indexer scores over its own blocks, gathered, must select what the scores over the whole
cache select; each rank's attention over its share of the selection, merged by
log-sum-exp, must give the attention over the whole selection; a prefill's gathered
caches must hold every block of its sequences. Without a GPU the kernels run in Triton's
CPU interpreter (fp32), with ``TRITON_INTERPRET=1`` set before Triton is imported.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Triton's interpreter has to be on before Triton is imported (here, or by conftest)
INTERPRETABLE = (
    torch.cuda.is_available()
    or os.environ.get("TRITON_INTERPRET") == "1"
    or "triton" not in sys.modules
)
if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

from atom.config import DCPConfig, admit_minimax_m3_pro_dcp
from atom.model_ops.attentions.token_layout.slots import dcp_slot_mapping

PG, D, NI, DI = 16, 256, 16, 128
GPU = torch.cuda.is_available()
DEV = "cuda" if GPU else "cpu"
DT = torch.bfloat16 if GPU else torch.float32
ATOL = 3e-2 if GPU else 2e-4


@pytest.fixture(scope="module")
def K():
    pytest.importorskip("triton")
    if not INTERPRETABLE:
        pytest.skip(
            "Triton was imported before TRITON_INTERPRET=1: set it to run on CPU"
        )
    from atom.model_ops.minimax_m3 import pro_kernels

    return pro_kernels


class Split:
    """Sequences of ``lens`` tokens in a global cache (K-only pages and index pages, at
    shuffled block ids) and the same blocks split over ``world`` ranks' caches."""

    def __init__(self, lens, world, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.lens, self.world = lens, world
        nb = [-(-n // PG) for n in lens]
        self.nb, self.ne = max(nb), max(-(-n // world) for n in nb)
        total = sum(nb)
        perm = torch.randperm(total, generator=g)
        self.bt = torch.zeros(len(lens), self.nb, dtype=torch.int32)
        c = 0
        for s, n in enumerate(nb):
            self.bt[s, :n] = perm[c : c + n].to(torch.int32)
            c += n
        # K-only pages as the kernels address them: [block, 1, page, 256]
        self.kc = (torch.randn(total, 1, PG, D, generator=g) * 0.5).to(DT)
        self.ic = (torch.randn(total, PG, DI, generator=g) * 0.5).to(DT)
        # rank r: every entry of every sequence, block j = e * world + r where it exists
        self.ranks = []
        for r in range(world):
            ents = [(s, e) for s, n in enumerate(nb) for e in range(-(-n // world))]
            order = torch.randperm(len(ents), generator=g)
            bt = torch.zeros(len(lens), self.ne, dtype=torch.int32)
            kc = torch.zeros(len(ents), 1, PG, D, dtype=DT)
            ic = torch.zeros(len(ents), PG, DI, dtype=DT)
            for i, (s, e) in enumerate(ents):
                phys = int(order[i])
                bt[s, e] = phys
                j = e * world + r
                if j < nb[s]:
                    kc[phys], ic[phys] = self.kc[self.bt[s, j]], self.ic[self.bt[s, j]]
            self.ranks.append(
                SimpleNamespace(bt=bt.to(DEV), kc=kc.to(DEV), ic=ic.to(DEV))
            )
        self.bt, self.kc, self.ic = self.bt.to(DEV), self.kc.to(DEV), self.ic.to(DEV)

    def rows(self, per_seq):
        """The last ``per_seq`` tokens of each sequence: ``(seq, positions)``."""
        seq = [s for s, n in enumerate(self.lens) for _ in range(per_seq)]
        pos = [n - per_seq + i for n in self.lens for i in range(per_seq)]
        return torch.tensor(seq, device=DEV), torch.tensor(pos, device=DEV)


# (block score, head reduction, index scale), as ``pro_attn`` maps the variants
READINGS = {
    "pool4/max": (1, 1, 1.0),
    "max/sum": (0, 0, 1.0),
    "lse/lse": (3, 2, DI**-0.5),
}


@pytest.mark.parametrize("reading", list(READINGS))
@pytest.mark.parametrize("tq", [0, 2])
# a decode's two rows, and a k=5 verify's six (`_dcp_decode`'s tile: 2)
@pytest.mark.parametrize("per_seq", [2, 6])
def test_scores_select_as_single_rank(K, reading, tq, per_seq):
    world, topk, init, local = 4, 8, 1, 2
    c = Split([300, 517, 70], world)
    seq, pos = c.rows(per_seq)
    bmode, hmode, sc = READINGS[reading]
    g = torch.Generator().manual_seed(1)
    iq = torch.randn(len(pos), NI, DI, generator=g).to(DT).to(DEV)
    cq = torch.randn(DI, generator=g).to(DT).to(DEV)
    kw = {
        "block_mode": bmode,
        "head_mode": hmode,
        "pool_temp": DI**-0.5,
        "causal": True,
        "tq": tq,
    }
    ref = K.scores(iq, c.ic, c.bt, seq, pos, c.nb, cq, init, local, sc, **kw)
    parts = []
    for r, rk in enumerate(c.ranks):
        qpos = K.dcp_local_positions(pos, r, world)
        s = K.scores(iq, rk.ic, rk.bt, seq, qpos, c.ne, cq, 0, 0, sc, **kw)
        parts.append(K.dcp_pin(s, pos, r, world, init, local))
    full = K.dcp_interleave(torch.cat(parts), world)
    assert full.shape == (len(pos), c.ne * world)
    for t, p in enumerate(pos.tolist()):
        causal = p // PG + 1
        torch.testing.assert_close(full[t, :causal], ref[t, :causal], atol=ATOL, rtol=0)
    sel = K.select_logical(full, pos, topk)
    want = K.select_logical(ref, pos, topk)
    assert torch.equal(sel.sort(dim=1).values, want.sort(dim=1).values)


@pytest.mark.parametrize("init, local", [(1, 2), (0, 4), (3, 0), (2, 11), (0, 0)])
def test_pins_a_ranks_causal_columns_by_global_block(K, init, local):
    """``dcp_pin``: on every causal column of a rank, the init blocks 1e30 and the local
    window 1e29 (an init block inside it too), by global block, the rest untouched."""
    world = 4
    pos = torch.tensor([0, 5, 16, 37, 70, 300, 517, 1000], device=DEV)
    nbl = 1000 // PG // world + 2
    g = torch.Generator().manual_seed(7)
    for r in range(world):
        s = torch.randn(len(pos), nbl, generator=g).to(DEV)
        j = torch.arange(nbl, device=DEV) * world + r
        qb = (pos // PG)[:, None]
        want = s.clone().masked_fill_(j < init, 1e30)
        want.masked_fill_(j >= qb + 1 - local, 1e29)
        causal = j <= qb
        got = K.dcp_pin(s, pos, r, world, init, local)
        assert torch.equal(got[causal], want[causal])


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("per_seq", [1, 6])
def test_topk_merge_selects_as_the_gathered_scores(K, split, per_seq):
    """``dcp_topk_merge``: each rank's best k, all-gathered as sort keys and merged, is
    the gathered scores' selection cut to the rank, order included: with ties, -inf
    blocks, pins, junk past a rank's causal blocks (pinned too), rows with fewer than k
    causal blocks and rows with none on a rank."""
    world, k, init, local = 4, 8, 1, 2
    c = Split([300, 517, 70, 20], world)
    seq, pos = c.rows(per_seq)
    g = torch.Generator().manual_seed(5)
    parts, qpos = [], []
    for r in range(world):
        s = torch.randint(0, 6, (len(pos), c.ne), generator=g).float()
        s[torch.rand(s.shape, generator=g) < 0.1] = float("-inf")
        parts.append(K.dcp_pin(s.to(DEV), pos, r, world, init, local))
        qpos.append(K.dcp_local_positions(pos, r, world))
    assert (qpos[world - 1] < 0).any()  # the 20-token sequence: nothing on rank 3
    full = K.dcp_interleave(torch.cat(parts), world)
    want = K.select_logical(full, pos, k, split=split)
    valid = (want >= 0).sum(dim=1)
    assert valid.max() == k and valid.min() < k
    picks = [K.select_logical(parts[r], qpos[r], k, split=split) for r in range(world)]
    keys = [
        K.dcp_local_keys(parts[r], picks[r], qpos[r], r, world) for r in range(world)
    ]
    gathered = torch.cat(keys)  # the all-gather, rank-major
    for r in range(world):

        def gather(x, r=r):
            assert torch.equal(x, keys[r])
            return gathered

        got = K.dcp_select(parts[r], qpos[r], k, r, world, gather, split=split)
        assert torch.equal(got, K.dcp_local_selection(want, r, world))


@pytest.mark.parametrize("splits", [1, 4])
# "spread": every rank holds some of each token's selection; "empty": a sequence shorter
# than `world` blocks and a selection that skips the last rank, which then sees no key
@pytest.mark.parametrize("case", ["spread", "empty"])
@pytest.mark.parametrize("per_seq", [2, 6])
def test_attention_merge_as_single_rank(K, splits, case, per_seq):
    world, topk = 4, 8
    c = Split([300, 517, 70] if case == "spread" else [40, 517], world, seed=2)
    seq, pos = c.rows(per_seq)
    g = torch.Generator().manual_seed(3)
    q = torch.randn(len(pos), 16, D, generator=g).to(DT).to(DEV)
    # a selection as the indexer leaves it: valid blocks first (own block included), -1 after
    sel = torch.full((len(pos), topk), -1, dtype=torch.int32)
    for t, p in enumerate(pos.tolist()):
        causal = torch.randperm(p // PG, generator=g)
        if case == "empty":
            causal = causal[causal % world != world - 1]
        causal = causal[: topk - 1]
        sel[t, : len(causal) + 1] = torch.cat([causal, torch.tensor([p // PG])])
    sel = sel.to(DEV)
    scale = D**-0.5
    want = q.new_empty(len(pos), 16 * 192)
    K.attention(q, c.kc, c.bt, seq, pos, sel, want, scale, splits, valid_first=True)
    outs, lses = [], []
    for r, rk in enumerate(c.ranks):
        out = q.new_empty(len(pos), 16 * 192)
        lse = torch.empty(len(pos), 16, device=DEV)
        mine = K.dcp_local_selection(sel, r, world)
        assert int((mine >= 0).sum()) == int(((sel >= 0) & (sel % world == r)).sum())
        qpos = K.dcp_local_positions(pos, r, world)
        K.attention(
            q,
            rk.kc,
            rk.bt,
            seq,
            qpos,
            mine,
            out,
            scale,
            splits,
            valid_first=True,
            lse=lse,
        )
        assert torch.isfinite(out).all()
        empty = (mine < 0).all(dim=1)
        if per_seq == 2:
            # six rows' random selections also skip some rank in "spread"
            assert empty.any() == (case == "empty" and r == world - 1)
        assert (out[empty] == 0).all() and (lse[empty] == float("-inf")).all()
        outs.append(out.float().view(len(pos), 16, 192))
        lses.append(lse)
    lse = torch.stack(lses)
    total = torch.logsumexp(lse, dim=0)
    merged = sum(torch.exp(lse[r] - total)[..., None] * outs[r] for r in range(world))
    torch.testing.assert_close(merged.flatten(1), want.float(), atol=ATOL, rtol=0)


def test_prefill_gather_holds_every_block(K):
    world = 4
    c = Split([300, 517, 70], world, seed=4)
    plans = [K.dcp_prefill_plan(rk.bt, c.lens, world) for rk in c.ranks]
    for (rows, tables), rk in zip(plans, c.ranks):
        # every rank's gather returns the ranks' rows in rank order, as an all-gather does
        def gather(x, rk=rk, rows=rows):
            assert torch.equal(x, rk.kc[rows])
            return torch.cat([o.kc[p[0]] for o, p in zip(c.ranks, plans)])

        kc = K.dcp_gather_blocks(rk.kc, rows, gather, world)
        assert torch.equal(tables, plans[0][1])
        for s, n in enumerate(c.lens):
            nb = -(-n // PG)
            torch.testing.assert_close(
                kc[tables[s, :nb].long()], c.kc[c.bt[s, :nb].long()]
            )


@pytest.mark.parametrize("interleave", [1, 16])
def test_decode_slots_follow_the_dcp_layout(interleave):
    rng = np.random.default_rng(0)
    world, bs, lens = 4, 16, [70, 513, 1]
    tables = [list(rng.permutation(64)[: -(-n // (bs * world)) + 1]) for n in lens]
    packed = np.zeros((len(lens), 16), dtype=np.int32)
    for s, row in enumerate(tables):
        packed[s, : len(row)] = row
    for rank in range(world):
        # three tokens of each sequence, as a verify writes them
        positions = np.concatenate([np.arange(n, n + 3) for n in lens])
        got = dcp_slot_mapping(
            positions, np.full(len(lens), 3), packed, bs, world, rank, interleave
        )
        # token by token: chunks of `interleave` tokens dealt round-robin over the ranks,
        # an entry of the table holding `bs` local tokens of each rank
        want = []
        for s, n in enumerate(lens):
            for p in range(n, n + 3):
                if (p // interleave) % world != rank:
                    want.append(-1)
                    continue
                local = (p // (interleave * world)) * interleave + p % interleave
                want.append(tables[s][p // (bs * world)] * bs + local % bs)
        assert got.tolist() == want


def _config(**kw):
    base = {
        "dcp_config": DCPConfig(),
        "kv_cache_block_size": 16,
        "speculative_config": None,
    }
    return SimpleNamespace(**{**base, **kw})


def test_config_interleaves_whole_blocks():
    cfg = _config()
    admit_minimax_m3_pro_dcp(cfg)
    assert cfg.dcp_config.interleave_size == 16
    with pytest.raises(ValueError, match="interleave_size"):
        admit_minimax_m3_pro_dcp(_config(dcp_config=DCPConfig(interleave_size=4)))


def test_config_admits_the_dspark_draft_only():
    cfg = _config(speculative_config=SimpleNamespace(method="dspark"))
    admit_minimax_m3_pro_dcp(cfg)
    assert cfg.dcp_config.interleave_size == 16
    for method in ("mtp", "eagle3"):
        with pytest.raises(ValueError, match="DSpark only"):
            admit_minimax_m3_pro_dcp(
                _config(speculative_config=SimpleNamespace(method=method))
            )


def test_compressed_index_pooled_per_rank_gathers_to_the_single_ranks(K):
    """``index_compress`` under DCP: every rank runs the step's keys, pooling and ringing
    only the blocks it holds (its slots, -1 elsewhere); a prefill's gather of those
    caches (`_dcp_prefill_caches`, after `compress_insert`) holds every closed group's
    row as the single rank's pooling of the whole cache does, through prefill chunks,
    a k=5 verify and single decodes."""
    world = 4
    PGc, G, RR = K.PAGE, K.GROUP, K.RING_ROWS
    lens = [300, 517, 70]
    c = Split(lens, world, seed=5)
    g = torch.Generator().manual_seed(6)
    cq = torch.randn(DI, generator=g).to(DEV)
    B = len(lens)

    def caches(n):
        return SimpleNamespace(
            ic=torch.full(
                (n, PGc // G, DI), float("nan"), dtype=torch.bfloat16, device=DEV
            ),
            ring=torch.zeros(B * RR, DI, dtype=torch.bfloat16, device=DEV),
            ring_pos=torch.full((B * RR,), -1, dtype=torch.int32, device=DEV),
        )

    one = caches(c.kc.shape[0])
    ranks = [caches(rk.kc.shape[0]) for rk in c.ranks]
    base = torch.arange(B, dtype=torch.int32, device=DEV) * RR
    # each request's step boundaries: prefill chunks, then a verify's 6 rows and decodes
    cuts = [[0, 150, 290, 296, 297, 300], [0, 400, 511, 517], [0, 40, 64, 70]]
    for step in range(max(len(x) for x in cuts) - 1):
        runs = [(x[min(step, len(x) - 1)], x[min(step + 1, len(x) - 1)]) for x in cuts]
        pos = [p for a, b in runs for p in range(a, b)]
        seq = [s for s, (a, b) in enumerate(runs) for _ in range(a, b)]
        ik = (torch.randn(len(pos), DI, generator=g) * 2).to(torch.bfloat16).to(DEV)

        def insert(cache, slots, pos=pos, seq=seq, ik=ik):
            K.compress_insert(
                ik,
                torch.tensor(pos, dtype=torch.int64, device=DEV),
                torch.tensor(seq, dtype=torch.int32, device=DEV),
                torch.tensor(slots, dtype=torch.int64, device=DEV),
                base,
                cache.ring,
                cache.ring_pos,
                cq,
                DI**-0.5,
                cache.ic,
            )

        insert(one, [int(c.bt[s, p // PGc]) * PGc + p % PGc for s, p in zip(seq, pos)])
        for r, (rk, own) in enumerate(zip(c.ranks, ranks)):
            j = [p // PGc for p in pos]
            insert(
                own,
                [
                    int(rk.bt[s, b // world]) * PGc + p % PGc if b % world == r else -1
                    for s, b, p in zip(seq, j, pos)
                ],
            )
        done = [b for _, b in runs]
        plans = [K.dcp_prefill_plan(rk.bt, done, world) for rk in c.ranks]
        for (rows, tables), own in zip(plans, ranks):

            def gather(x, rows=rows, plans=plans):
                return torch.cat([o.ic[p[0]] for o, p in zip(ranks, plans)])

            ic = K.dcp_gather_blocks(own.ic, rows, gather, world)
            for s, n in enumerate(done):
                for g0 in range(0, n - G + 1, G):
                    j, row = g0 // PGc, (g0 % PGc) // G
                    torch.testing.assert_close(
                        ic[int(tables[s, j]), row],
                        one.ic[int(c.bt[s, j]), row],
                        atol=0,
                        rtol=0,
                    )
