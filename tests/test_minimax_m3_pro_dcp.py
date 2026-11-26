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


@pytest.mark.parametrize("fold", [False, True])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("per_seq", [1, 6])
def test_topk_merge_selects_as_the_gathered_scores(K, split, per_seq, fold):
    """``dcp_topk_merge``: each rank's best k, all-gathered as sort keys and merged, is
    the gathered scores' selection cut to the rank, order included: with ties, -inf
    blocks, pins, junk past a rank's causal blocks (pinned too), rows with fewer than k
    causal blocks and rows with none on a rank. ``fold``: the selection's own keys are
    ``dcp_local_keys``' (the all-gather is handed exactly those)."""
    world, k, init, local = 4, 8, 1, 2
    c = Split([300, 517, 70, 20], world)
    _, pos = c.rows(per_seq)
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

        got = K.dcp_select(
            parts[r], qpos[r], k, r, world, gather, split=split, fold=fold
        )
        assert torch.equal(got, K.dcp_local_selection(want, r, world))


@pytest.mark.parametrize("fused", [True, False])
def test_the_folded_keys_over_spans_are_dcp_local_keys(K, fused):
    """``dcp_fold``'s keys from a selection over several spans (merged by the last span
    or a second launch) and from the one-launch one are ``dcp_local_keys``' of their
    ids, bit for bit: ties, -inf blocks, a row with no causal block on the rank, one
    with fewer than k."""
    world, r, k, nb = 4, 2, 8, 200
    g = torch.Generator().manual_seed(11)
    pos = torch.tensor([-16, 5, 100, 1000, 3199, 2000, 47]).to(DEV)
    s = torch.randint(0, 6, (len(pos), nb), generator=g).float()
    s[torch.rand(s.shape, generator=g) < 0.1] = float("-inf")
    s = s.to(DEV)
    split = {"parts": 3, "chunk": 32, "fused": fused, "logical": True}
    ids = K.select_rows_split(s, s, pos, pos, k, **split)[0]
    got = K.select_rows_split(s, s, pos, pos, k, dcp_keys=(r, world), **split)[0]
    assert got.dtype == torch.int64
    assert torch.equal(got, K.dcp_local_keys(s, ids, pos, r, world))
    ids = K.select_logical(s, pos, k)
    got = K.select_logical(s, pos, k, dcp_keys=(r, world))
    assert torch.equal(got, K.dcp_local_keys(s, ids, pos, r, world))


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


@pytest.mark.parametrize("splits", [1, 4])
def test_an_fp8_query_attends_as_its_values_do(K, splits):
    """``dcp_q_fp8``'s gathered query, a plain e4m3 cast, attends bit for bit as the
    same values in the output's dtype: output and LSE."""
    world = 4
    c = Split([300, 517, 70], world, seed=2)
    seq, pos = c.rows(6)
    g = torch.Generator().manual_seed(8)
    q8 = (torch.randn(len(pos), 16, D, generator=g) * 3).to(torch.float8_e4m3fn)
    q8 = q8.to(DEV)
    rk, r = c.ranks[1], 1
    # local blocks, valid first; the kernel drops the ones past a row's own
    sel = torch.stack([torch.randperm(8, generator=g) for _ in pos]).int().to(DEV)
    qpos = K.dcp_local_positions(pos, r, world)
    got = []
    for q in (q8, q8.to(DT)):
        out = torch.empty(len(pos), 16 * 192, dtype=DT, device=DEV)
        lse = torch.empty(len(pos), 16, device=DEV)
        K.attention(
            q,
            rk.kc,
            rk.bt,
            seq,
            qpos,
            sel,
            out,
            D**-0.5,
            splits,
            valid_first=True,
            lse=lse,
        )
        got.append((out, lse))
    assert torch.equal(got[0][0], got[1][0]) and torch.equal(got[0][1], got[1][1])


@pytest.mark.parametrize("splits", [1, 4])
def test_the_attention_stores_the_lse_repeated(K, splits):
    """``lse_rep``: each head's LSE stored that many times in a row, bit for bit the
    plain store's ``repeat_interleave``, the output as it was; -inf where a row sees no
    key on the rank."""
    world, r = 4, 3
    c = Split([40, 517, 70], world, seed=2)
    seq, pos = c.rows(4)
    g = torch.Generator().manual_seed(9)
    q = torch.randn(len(pos), 16, D, generator=g).to(DT).to(DEV)
    rk = c.ranks[r]
    sel = torch.stack([torch.randperm(8, generator=g) for _ in pos]).int().to(DEV)
    qpos = K.dcp_local_positions(pos, r, world)
    got = []
    for rep in (1, 3):
        out = torch.empty(len(pos), 16 * 192, dtype=DT, device=DEV)
        lse = torch.empty(len(pos), 16 * rep, device=DEV)
        K.attention(
            q,
            rk.kc,
            rk.bt,
            seq,
            qpos,
            sel,
            out,
            D**-0.5,
            splits,
            valid_first=True,
            lse=lse,
            lse_rep=rep,
        )
        got.append((out, lse))
    (o1, l1), (o3, l3) = got
    assert torch.equal(o1, o3) and torch.equal(l3, l1.repeat_interleave(3, dim=1))
    assert (l1 == float("-inf")).any() and torch.isfinite(l1).any()


def test_the_all_to_all_merge_writes_the_callers_buffer(monkeypatch):
    """``dcp_fold``'s merge: the a2a combine into the caller's ``[B, H_local, D]`` (the
    layer's output viewed as 48 heads of 64), bit for bit its own buffer's, returned."""
    from conftest import gpu_stack_import

    dcp_ops = gpu_stack_import("atom.model_ops.dcp_ops")

    def all_to_all(recv, send, group=None):  # every rank sends what this one does
        recv.copy_(send)

    monkeypatch.setattr(torch.distributed, "all_to_all_single", all_to_all)
    group = SimpleNamespace(world_size=8, device_group=None)
    g = torch.Generator().manual_seed(4)
    of = torch.randn(5, 8 * 48, 64, generator=g).to(DT).to(DEV)
    lse = torch.randn(5, 8 * 48, generator=g).to(DEV)
    lse[1] = float("-inf")  # a row no rank holds a key of
    want = dcp_ops.cp_lse_a2a(of, lse, group)
    for merge in (dcp_ops.cp_lse_a2a, dcp_ops.dcp_lse_merge):
        out = torch.full((5, 16 * 192), float("nan"), dtype=DT, device=DEV)
        got = merge(of, lse, group, out=out.view(5, 48, 64))
        assert got.data_ptr() == out.data_ptr() and torch.equal(got, want)


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


# -- the decode merge: aiter's mailbox combine, else all-to-all ----------------------


@pytest.fixture
def mailbox(monkeypatch):
    """A DCP8 Pro layer (16 heads x 192 a rank, 64 sequences x k=5) between stub
    collectives: aiter's mailbox workspace and combine, and the all-to-all, record what
    they are given. ``(impl, config, built workspaces, calls, group)``."""
    from conftest import gpu_stack_import

    pro_attn = gpu_stack_import("atom.model_ops.minimax_m3.pro_attn")
    from atom.model_ops import dcp_ops

    built, calls = [], {}

    class Workspace:
        def __init__(self, group, device, heads, dim, max_tokens):
            built.append((heads, dim, max_tokens))
            self.max_tokens = max_tokens
            self.peers = SimpleNamespace(bytes=torch.empty(0))

    def combine(ws, out, lse, base_e=True, row_counts=None, res=None):
        calls["mailbox"] = (out, lse, res)
        return res.fill_(7)

    def a2a(of, lse, group, owned_counts=None, quant_dtype=None, out=None):
        calls["a2a"] = (of, lse, out)
        merged = torch.arange(of.shape[0] * of.shape[1] // 8 * 64.0).view(
            of.shape[0], -1, 64
        )
        return merged if out is None else out.copy_(merged)

    monkeypatch.setitem(
        sys.modules,
        "aiter.ops.flydsl.dcp_mailbox",
        SimpleNamespace(CombineWorkspace=Workspace, dcp_lse_combine=combine),
    )
    monkeypatch.setattr(dcp_ops, "_MAILBOX_WORKSPACES", {})
    monkeypatch.setattr(dcp_ops, "_MAILBOX_LAUNCHES", {})
    monkeypatch.setattr(dcp_ops, "cp_lse_a2a", a2a)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    group = SimpleNamespace(world_size=8, rank_in_group=3, cpu_group="gloo")
    monkeypatch.setattr("atom.distributed.dcp_utils.get_dcp_group", lambda: group)
    config = SimpleNamespace(
        dcp_config=SimpleNamespace(comm_backend="mailbox"),
        max_num_seqs=64,
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    monkeypatch.setattr("atom.config.get_current_atom_config", lambda: config)
    impl = object.__new__(pro_attn.ProSparseAttentionImpl)
    impl.num_heads, impl.value_head_dim, impl.dcp_mailbox = 16, 192, False
    return impl, config, built, calls, group


def test_the_mailbox_holds_the_variants_rows_of_a_ranks_heads(mailbox):
    """``dcp_mailbox`` rows (at most the widest decode step, 64 x 6) of 16 heads x 192,
    past the default cap (85 rows at that width); one workspace for all the layers, and
    none without the variant or the mailbox backend."""
    from atom.model_ops import dcp_ops

    impl, config, built, _, _ = mailbox
    assert dcp_ops._MAILBOX_MAX_ELEMS // (8 * 16 * 192) == 85
    assert all(impl._prepare_dcp_mailbox(256) for _ in range(3))  # three layers
    assert built == [(16, 192, 256)]
    dcp_ops._MAILBOX_WORKSPACES.clear()
    assert impl._prepare_dcp_mailbox(4096) and built[-1] == (16, 192, 384)
    dcp_ops._MAILBOX_WORKSPACES.clear()
    assert not impl._prepare_dcp_mailbox(0)
    config.dcp_config.comm_backend = "a2a"
    assert not impl._prepare_dcp_mailbox(256)
    assert len(built) == 2


@pytest.mark.parametrize(
    "on, rows, capturing, takes",
    [
        (True, 8, True, True),
        (True, 9, True, False),  # wider than the workspace
        (True, 8, False, False),  # eager
        (False, 8, True, False),
    ],
)
def test_a_decode_merge_combines_in_the_mailbox_else_by_all_to_all(
    mailbox, monkeypatch, on, rows, capturing, takes
):
    """The mailbox takes a captured step it holds, at the heads' own 192 and LSEs, into
    the layer's output; anything else merges by all-to-all as before, at three heads of
    64 a head with its LSE repeated, copied into the output."""
    impl, _, _, calls, group = mailbox
    assert impl._prepare_dcp_mailbox(8)
    impl.dcp_mailbox = on
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    g = torch.Generator().manual_seed(0)
    of = torch.randn(rows, 8 * 16 * 192, generator=g).to(torch.bfloat16)
    lse = torch.randn(rows, 8 * 16, generator=g)
    out = torch.zeros(rows, 16 * 192, dtype=torch.bfloat16)
    assert impl._dcp_merge(group, of, lse, out) is out
    if takes:
        o, l, res = calls["mailbox"]
        assert o.shape == (rows, 128, 192) and o.data_ptr() == of.data_ptr()
        assert l.data_ptr() == lse.data_ptr()
        assert res.shape == (rows, 16, 192) and res.data_ptr() == out.data_ptr()
        assert (out == 7).all() and "a2a" not in calls
    else:
        o, l, buf = calls["a2a"]
        assert o.shape == (rows, 384, 64) and o.data_ptr() == of.data_ptr()
        assert torch.equal(l, lse.repeat_interleave(3, dim=1)) and buf is None
        want = torch.arange(rows * 48 * 64.0).to(torch.bfloat16).view(rows, -1)
        assert torch.equal(out, want) and "mailbox" not in calls


@pytest.mark.parametrize("on", [False, True])
def test_dcp_fold_merges_a_repeated_lse_into_the_output(mailbox, monkeypatch, on):
    """``dcp_fold``: the attention's LSE comes repeated once a 64-wide part, which the
    all-to-all takes as it is (and the mailbox, reading each head's once, leaves), and
    its combine writes the layer's output."""
    impl, _, _, calls, group = mailbox
    assert impl._prepare_dcp_mailbox(8)
    impl.dcp_mailbox = on
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    g = torch.Generator().manual_seed(0)
    of = torch.randn(8, 8 * 16 * 192, generator=g).to(torch.bfloat16)
    lse = torch.randn(8, 8 * 16 * 3, generator=g)
    out = torch.zeros(8, 16 * 192, dtype=torch.bfloat16)
    assert impl._dcp_merge(group, of, lse, out, fold=True) is out
    o, l, buf = calls["a2a"]
    assert o.shape == (8, 384, 64) and l is lse and "mailbox" not in calls
    assert buf.shape == (8, 48, 64) and buf.data_ptr() == out.data_ptr()
    want = torch.arange(8 * 48 * 64.0).to(torch.bfloat16).view(8, -1)
    assert torch.equal(out, want)


@pytest.fixture
def decode(monkeypatch):
    """``ProSparseAttentionImpl._dcp_decode`` of rank 3 of 8, 12 rows, between stubs
    that record what reaches the query all-gather, the attention and the merge.
    ``(run(**variants) -> (q, out), calls)``."""
    from conftest import gpu_stack_import

    pro_attn = gpu_stack_import("atom.model_ops.minimax_m3.pro_attn")
    from atom.model_ops import dcp_ops

    P, calls = pro_attn.pro_kernels, {}
    group = SimpleNamespace(world_size=8, rank_in_group=3)
    monkeypatch.setattr("atom.distributed.dcp_utils.get_dcp_group", lambda: group)
    monkeypatch.setattr(
        pro_attn,
        "get_forward_context",
        lambda: SimpleNamespace(attn_metadata=SimpleNamespace()),
    )
    monkeypatch.setattr(P, "dcp_local_positions", lambda pos, r, w: pos)
    monkeypatch.setattr(P, "scores", lambda iq, *a, **k: torch.zeros(iq.shape[0], 8))
    monkeypatch.setattr(P, "dcp_pin", lambda score, *a: score)

    def select(score, qpos, k, *a, **kw):
        calls["select"] = kw
        return torch.zeros(score.shape[0], k, dtype=torch.int32)

    def gather_q(group, q):
        calls["gather"] = q
        return torch.empty(q.shape[0], 8 * 16, q.shape[2], dtype=q.dtype)

    def attention(q, kc4, bt, seq, pos, sel, out, *a, **kw):
        calls["attention"] = (q, out, kw)

    def attention_day0(q, kc4, bt, seq, pos, sel, out, scale, lse, rep):
        calls["day0"] = (q, kc4, seq, pos, out, lse, rep)

    monkeypatch.setattr(P, "dcp_select", select)
    monkeypatch.setattr(dcp_ops, "dcp_all_gather_query_heads", gather_q)
    monkeypatch.setattr(P, "attention", attention)
    monkeypatch.setattr(P, "attention_dcp_day0", attention_day0)
    monkeypatch.setattr(P, "dcp_day0_decode_available", lambda: True)
    impl = object.__new__(pro_attn.ProSparseAttentionImpl)
    impl.dcp_world, impl.topk, impl.scale, impl.value_head_dim = 8, 128, 0.1, 192
    impl.k_only, impl.page, impl.num_heads = True, 16, 16

    def merge(group, of, lse, out, fold=False):
        calls["merge"] = (of, lse, fold)
        return out

    impl._dcp_merge = merge

    def run(mailbox=False, kc4=None, **variants):
        impl.dcp_mailbox = mailbox
        v = pro_attn.ProVariants(dcp_topk_merge=True, topk="split", **variants)
        n = 12
        q = torch.randn(n, 16, 256).to(torch.bfloat16)
        out = torch.empty(n, 16 * 192, dtype=torch.bfloat16)
        pos = torch.arange(n) + 100
        tables = torch.zeros(2, 64, dtype=torch.int32)
        seq = torch.zeros(n, dtype=torch.int32)
        iq = torch.zeros(n, 16, 128)
        score_args = (False, 1, 2, 1.0, 0, 0)
        got = impl._dcp_decode(
            q, iq, pos, seq, tables, kc4, None, None, v, out, score_args
        )
        assert got is out
        return q, out

    return run, calls


@pytest.mark.parametrize("fp8", [False, True])
def test_dcp_q_fp8_gathers_the_query_as_its_e4m3_cast(decode, fp8):
    """The all-gather takes the query's plain e4m3 cast (half the bytes), the attention
    reads that, and the partial outputs stay in the output's dtype."""
    run, calls = decode
    q, out = run(dcp_q_fp8=fp8)
    sent = calls["gather"]
    if fp8:
        assert sent.dtype == torch.float8_e4m3fn and sent.is_contiguous()
        want = q.to(torch.float8_e4m3fn)
        assert torch.equal(sent.view(torch.uint8), want.view(torch.uint8))
    else:
        assert sent is q
    qa, of, _ = calls["attention"]
    assert qa.dtype == sent.dtype and qa.shape == (12, 128, 256)
    assert of.dtype == out.dtype == torch.bfloat16 and of.shape == (12, 128 * 192)
    assert calls["merge"][0] is of


@pytest.mark.parametrize("fold, mailbox, rep", [(0, 0, 1), (1, 0, 3), (1, 1, 1)])
def test_dcp_fold_asks_for_the_keys_the_lse_as_merged_and_the_output(
    decode, fold, mailbox, rep
):
    """``dcp_fold`` has the selection store its keys, the attention the LSE as the
    all-to-all reads it (once a head under the mailbox merge) and the merge the
    output."""
    run, calls = decode
    run(mailbox=bool(mailbox), dcp_fold=bool(fold))
    assert calls["select"]["fold"] is bool(fold)
    _, of, kw = calls["attention"]
    assert kw["lse_rep"] == rep and kw["lse"].shape == (12, 128 * rep)
    m_of, m_lse, m_fold = calls["merge"]
    assert m_of is of and m_lse is kw["lse"] and m_fold is bool(fold)


@pytest.mark.parametrize("fold, mailbox, rep", [(0, 0, 1), (1, 0, 3), (1, 1, 1)])
def test_dcp_day0_decode_attends_the_e4m3_query_in_the_tritons_stead(
    decode, fold, mailbox, rep
):
    """``dcp_day0_decode``: the all-gather takes the query's e4m3 cast, aiter's DCP decode
    attends it over the plain fp8 cache (the rows' positions and sequences as int32), and
    its output and LSE (repeated as the merge reads it) go to the merge."""
    run, calls = decode
    kc4 = torch.zeros(4, 1, 16, 256, dtype=torch.float8_e4m3fn)
    q, _ = run(
        mailbox=bool(mailbox),
        kc4=kc4,
        dcp_day0_decode=True,
        k_plain_fp8=True,
        dcp_fold=bool(fold),
    )
    assert "attention" not in calls
    sent = calls["gather"]
    assert torch.equal(
        sent.view(torch.uint8), q.to(torch.float8_e4m3fn).view(torch.uint8)
    )
    qa, k, seq, pos, of, lse, r = calls["day0"]
    assert qa.dtype == torch.float8_e4m3fn and qa.shape == (12, 128, 256) and k is kc4
    assert seq.dtype == pos.dtype == torch.int32
    assert torch.equal(pos, torch.arange(12, dtype=torch.int32) + 100)
    assert of.shape == (12, 128 * 192) and r == rep and lse.shape == (12, 128 * rep)
    m_of, m_lse, m_fold = calls["merge"]
    assert m_of is of and m_lse is lse and m_fold is bool(fold)


@pytest.mark.parametrize(
    "plain, dtype", [(False, torch.float8_e4m3fn), (True, torch.bfloat16)]
)
def test_dcp_day0_decode_keeps_triton_off_the_plain_fp8_cache(decode, plain, dtype):
    """Without ``k_plain_fp8``, or on a bf16 cache, the Triton attention serves."""
    run, calls = decode
    run(
        kc4=torch.zeros(4, 1, 16, 256, dtype=dtype),
        dcp_day0_decode=True,
        k_plain_fp8=plain,
    )
    assert "attention" in calls and "day0" not in calls
