"""MiniMax-M3-Pro's compressed index cache (``index_compress``): ``prep`` hands the step's index
keys to ``compress_insert``, which softmax-pools every 4 into one bf16 row of ``ic_c`` and keeps
each request's trailing keys in its ring. Checked against ``ProSparseAttentionImpl._centroids``'
pooling of the final keys, through chunked prefill, decode, and DSpark verifies whose rejected
drafts are written over."""

import os

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

pytest.importorskip("triton")

from atom.model_ops.minimax_m3 import pro_kernels as K

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PG, G, RR, DI = K.PAGE, K.GROUP, K.RING_ROWS, K.IDX_DIM
POOL_SCALE = DI**-0.5


def _pool(keys, cq):
    """The reference: ``_centroids`` of 4 keys (bf16 in, fp32 math)."""
    k = keys.float()
    w = torch.softmax((k * cq.float()).sum(-1) * POOL_SCALE, dim=-1)
    return (w[:, None] * k).sum(0)


class Stream:
    """Two requests' index-key streams through the cache, as the runner drives them: each
    step writes a contiguous run of positions per request; a verify's rejected drafts are
    rewritten by the next step from the first rejected position."""

    def __init__(self, seed=0, ring=True):
        self.g = torch.Generator().manual_seed(seed)
        self.cq = torch.randn(DI, generator=self.g).to(DEV)
        self.nblk = 16
        self.bt = torch.arange(2 * self.nblk, dtype=torch.int32).view(2, self.nblk)
        self.bt = (
            self.bt[:, torch.randperm(self.nblk, generator=self.g)].contiguous().to(DEV)
        )
        self.ic = torch.full(
            (2 * self.nblk, PG // G, DI), float("nan"), dtype=torch.bfloat16, device=DEV
        )
        self.ring = torch.zeros(4 * 2 * RR, DI, dtype=torch.bfloat16, device=DEV)
        self.ring_pos = torch.full((4 * 2 * RR,), -1, dtype=torch.int32, device=DEV)
        self.base = torch.tensor(
            [5 * RR, 2 * RR] if ring else [-1, -1], dtype=torch.int32, device=DEV
        )
        self.keys = [{}, {}]  # position -> the key that stands there now
        self.length = [0, 0]  # committed positions

    def key(self):
        return (torch.randn(DI, generator=self.g) * 2).to(torch.bfloat16).to(DEV)

    def step(self, runs):
        """``runs``: per request ``(start, n)`` (n may be 0); returns the step's keys."""
        ik, pos, seq, slot = [], [], [], []
        for s, (start, n) in enumerate(runs):
            for p in range(start, start + n):
                k = self.key()
                self.keys[s][p] = k
                ik.append(k)
                pos.append(p)
                seq.append(s)
                slot.append(int(self.bt[s, p // PG]) * PG + p % PG)
        ik = torch.stack(ik).contiguous()
        as_t = lambda x, dt: torch.tensor(x, dtype=dt, device=DEV)
        K.compress_insert(
            ik,
            as_t(pos, torch.int64),
            as_t(seq, torch.int32),
            as_t(slot, torch.int64),
            self.base,
            self.ring,
            self.ring_pos,
            self.cq,
            POOL_SCALE,
            self.ic,
        )

    def commit(self, s, length):
        self.length[s] = length
        for p in [p for p in self.keys[s] if p >= length]:
            del self.keys[s][p]  # a rejected draft: nothing stands there any more

    def check(self):
        """Every closed, committed group's row is the pooling of the keys there now."""
        n = 0
        for s in range(2):
            for g0 in range(0, self.length[s] - G + 1, G):
                want = _pool(
                    torch.stack([self.keys[s][g0 + i] for i in range(G)]), self.cq
                )
                got = self.ic[int(self.bt[s, g0 // PG]), (g0 % PG) // G].float()
                torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
                n += 1
        return n


def test_groups_pool_through_chunks_decodes_and_rejected_verifies():
    st = Stream(seed=1)
    # chunked prefill, chunks not group-aligned, two requests in one step
    st.step([(0, 37), (0, 13)])
    st.commit(0, 37), st.commit(1, 13)
    st.step([(37, 21), (13, 6)])
    st.commit(0, 58), st.commit(1, 19)
    assert st.check() == 14 + 4
    # plain decodes: every group closes one token at a time across steps
    for _ in range(5):
        st.step([(st.length[0], 1), (st.length[1], 1)])
        st.commit(0, st.length[0] + 1), st.commit(1, st.length[1] + 1)
    st.check()
    # DSpark k7 verifies: 8 rows each; request 0 accepts 2 drafts, request 1 all 7 (+bonus)
    for _ in range(3):
        a, b = st.length
        st.step([(a, 8), (b, 8)])
        st.commit(
            0, a + 3
        )  # rows a..a+2 stand; a+3..a+7 were rejected and get rewritten
        st.commit(1, b + 8)
        st.check()
    assert st.length == [72, 48]


def test_a_request_without_a_ring_pools_what_its_step_holds():
    """No ring slot (or one taken over): a group whose earlier keys a past step wrote is
    pooled from the keys this step has, never from another request's ring rows."""
    st = Stream(seed=2, ring=False)
    st.step([(0, 6), (0, 0)])  # group 0 closes in-step; group 1 holds 4, 5
    st.step([(6, 2), (0, 0)])  # group 1 closes from 6, 7 alone
    want = _pool(torch.stack([st.keys[0][6], st.keys[0][7]]), st.cq)
    got = st.ic[int(st.bt[0, 0]), 1].float()
    torch.testing.assert_close(got, want, rtol=2e-2, atol=2e-2)
    want0 = _pool(torch.stack([st.keys[0][i] for i in range(4)]), st.cq)
    torch.testing.assert_close(
        st.ic[int(st.bt[0, 0]), 0].float(), want0, rtol=2e-2, atol=2e-2
    )


def test_pad_rows_write_nothing():
    st = Stream(seed=3)
    before = st.ring.clone(), st.ring_pos.clone(), st.ic.clone()
    ik = torch.zeros(3, DI, dtype=torch.bfloat16, device=DEV)
    pos = torch.tensor([3, 7, 15], dtype=torch.int64, device=DEV)
    seq = torch.tensor(
        [99, 99, 99], dtype=torch.int32, device=DEV
    )  # pad rows: no batch row
    slot = torch.full((3,), -1, dtype=torch.int64, device=DEV)
    K.compress_insert(
        ik, pos, seq, slot, st.base, st.ring, st.ring_pos, st.cq, POOL_SCALE, st.ic
    )
    assert torch.equal(st.ring, before[0]) and torch.equal(st.ring_pos, before[1])
    assert torch.equal(st.ic.isnan(), before[2].isnan())


def test_prep_hands_over_the_keys_it_would_cache():
    """``ik_out`` is the bf16 index key ``prep`` writes to a bf16 index cache."""
    pytest.importorskip("aiter", reason="the prep test case builds an attention impl")
    from test_minimax_m3_pro_kernels import Case, H

    c = Case([21, 6], seed=5)
    c.run_prep()
    slot = c.slot.long()
    cached = c.ic.view(-1, DI)[slot]
    ik = torch.empty(c.T, DI, dtype=c.qkv.dtype, device=DEV)
    c2 = Case([21, 6], seed=5)
    K.prep(
        c2.qkv,
        c2.pos,
        c2.slot,
        c2.cos,
        c2.sin,
        1.0,
        c2.wq,
        c2.wk,
        c2.wiq,
        c2.wik,
        c2.eps,
        c2.kc4,
        c2.ic,
        H,
        vc4=c2.vc4,
        k_scale=c2.ks,
        v_scale=c2.vs,
        ik_out=ik,
    )
    assert torch.equal(ik, cached)
    assert not c2.ic.any()  # the index cache itself is not written


def test_layer_selects_the_blocks_the_pooled_keys_score_highest(monkeypatch):
    """``_kernel_attention`` with ``index_compress`` over two steps (a prefill of each request's
    first rows, then the rest): every row of the second step attends the blocks a torch
    reference picks from the centroids of the very bf16 keys ``prep`` produced (max over a
    block's 4 rows, then over heads; sink and local pinned), its scores reading rows both
    steps wrote."""
    pytest.importorskip("aiter", reason="the layer test case builds an attention impl")
    from test_minimax_m3_pro_kernels import DT, VD, Case, H, TestImplGlue

    from atom.model_ops.minimax_m3.pro_attn import ProVariants

    c = Case([150, 97], topk=8, seed=7)
    v = ProVariants(index_compress=True, topk="fused")
    TestImplGlue()._impl_with_rope(c, v)
    ic_c = torch.zeros(c.ic.shape[0], PG // G, DI, dtype=torch.bfloat16, device=DEV)
    ring = (torch.tensor([0, RR], dtype=torch.int32, device=DEV), 4 * RR)
    keys = torch.zeros(c.T, DI, dtype=DT, device=DEV)
    iqs = torch.zeros(c.T, 16, DI, dtype=DT, device=DEV)
    sel_seen = {}
    real_prep, real_attention = K.prep, K.attention
    rows_now = {}

    def prep_spy(*a, **k):
        q, iq = real_prep(*a, **k)
        keys[rows_now["r"]] = k["ik_out"]
        iqs[rows_now["r"]] = iq
        return q, iq

    def attention_spy(q, kc4, bt, seq, pos, sel, out, scale, *a, **k):
        for i, t in enumerate(rows_now["r"].tolist()):
            sel_seen[t] = set(x for x in sel[i].tolist() if x >= 0)
        return real_attention(q, kc4, bt, seq, pos, sel, out, scale, *a, **k)

    monkeypatch.setattr(K, "prep", prep_spy)
    monkeypatch.setattr(K, "attention", attention_spy)
    a0, a1 = c.seq_slice(0), c.seq_slice(1)
    steps = [
        torch.cat([torch.arange(a0[0], a0[0] + 120), torch.arange(a1[0], a1[0] + 80)]),
        torch.cat([torch.arange(a0[0] + 120, a0[1]), torch.arange(a1[0] + 80, a1[1])]),
    ]
    for r in steps:
        r = r.to(DEV)
        rows_now["r"] = r
        out = torch.zeros(r.numel(), H * VD, dtype=DT, device=DEV)
        c.impl._kernel_attention(
            c.qkv[r],
            c.pos[r],
            c.seq[r],
            c.slot[r],
            c.bt,
            c.layer,
            ic_c,
            c.bt.shape[1],
            v,
            out,
            ring=ring,
        )
    # the reference: centroids of each request's groups, then each row's top-k blocks
    cq = c.impl.index_compress_query.data.float()
    for t in steps[1].tolist():
        s, p = int(c.seq[t]), int(c.pos[t])
        start = c.seq_slice(s)[0]
        n = (
            (p + 1) // G * G
        )  # this row's closed groups (its own block is pinned anyway)
        g = keys[start : start + n].float().view(-1, G, DI)
        w = torch.softmax((g * cq).sum(-1) * v.pool_scale, dim=-1)
        cent = (w[..., None] * g).sum(1).to(torch.bfloat16).float()  # [groups, DI]
        dots = torch.einsum("hd,cd->hc", iqs[t].float(), cent) * v.index_scale
        qb = p // PG
        nb = qb + 1
        score = torch.full((nb,), float("-inf"))
        for j in range(nb):
            rows = dots[:, j * (PG // G) : (j + 1) * (PG // G)]
            if rows.shape[1]:
                score[j] = rows.amax(-1).amax().item()
        init, local = c.impl.init_blocks, c.impl.local_blocks
        for j in range(nb):
            if j < init:
                score[j] = 1e30
            elif j >= qb + 1 - local:
                score[j] = 1e29
        want = set(score.topk(min(c.topk, nb)).indices.tolist())
        assert sel_seen[t] == want, (t, p, sorted(sel_seen[t] ^ want))
