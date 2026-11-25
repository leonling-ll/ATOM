"""``compress_insert`` in one launch (``max_rows`` at most ``FUSED_INSERT_MAX_ROWS``): the
compressed cache, the rings and their positions are the two-launch path's bytes, step for
step, through unaligned prefill chunks, one-token decodes, DSpark verifies whose rejected
drafts are written over, and groups that close across steps; a step of more rows takes the
two launches."""

import contextlib
import functools

import pytest
import torch
from test_minimax_m3_pro_compress import K, Stream

MAX = K.FUSED_INSERT_MAX_ROWS


class Launches:
    """Records each launch of the insert kernel as its ``(POOL, STORE)`` flags."""

    def __init__(self, kernel):
        self.kernel, self.flags = kernel, []

    def __getitem__(self, grid):
        def launch(*a, **k):
            self.flags.append((k["POOL"], k["STORE"]))
            return self.kernel[grid](*a, **k)

        return launch


@contextlib.contextmanager
def bound(monkeypatch, max_rows):
    """``Stream.step`` calls ``compress_insert`` with ``max_rows``; yields its launches."""
    launches = Launches(K._compress_insert_kernel)
    with monkeypatch.context() as m:
        m.setattr(K, "_compress_insert_kernel", launches)
        m.setattr(
            K,
            "compress_insert",
            functools.partial(K.compress_insert, max_rows=max_rows),
        )
        yield launches


def assert_same(a, b):
    assert torch.equal(a.ic.view(torch.int16), b.ic.view(torch.int16))  # NaN rows too
    assert torch.equal(a.ring, b.ring) and torch.equal(a.ring_pos, b.ring_pos)


@pytest.mark.parametrize("k", [3, 5, 7, 12])
def test_one_launch_writes_the_two_launch_bytes(monkeypatch, k):
    two, one = Stream(seed=10 + k), Stream(seed=10 + k)
    rng = torch.Generator().manual_seed(k)

    def step(runs, accept):
        two.step(runs)
        rows = max(n for _, n in runs)
        with bound(monkeypatch, rows) as launches:
            one.step(runs)
        assert launches.flags == [(True, True)]
        for st in (two, one):
            for s, (start, n) in enumerate(runs):
                st.commit(s, start + accept[s])
        assert_same(two, one)

    # unaligned prefill chunks (their own launches), then the same into both
    for st in (two, one):
        st.step([(0, 37), (0, 13)]), st.commit(0, 37), st.commit(1, 13)
    assert_same(two, one)
    step([(37, 6), (13, 3)], [6, 3])  # a short chunk under the bound, unaligned
    for _ in range(6):  # one-token decodes: groups close one token at a time
        step([(two.length[0], 1), (two.length[1], 1)], [1, 1])
    for _ in range(8):  # verifies: k drafts + 1, a random number accepted
        acc = torch.randint(1, k + 2, (2,), generator=rng).tolist()
        step([(two.length[0], k + 1), (two.length[1], k + 1)], acc)
    assert two.check() == one.check() > 0


def test_a_step_past_the_bound_takes_two_launches(monkeypatch):
    two, one = Stream(seed=4), Stream(seed=4)
    for st in (two, one):
        st.step([(0, 5), (0, 2)]), st.commit(0, 5), st.commit(1, 2)
    two.step([(5, MAX + 1), (2, 3)])
    with bound(monkeypatch, MAX + 1) as launches:
        one.step([(5, MAX + 1), (2, 3)])
    assert launches.flags == [(True, False), (False, True)]
    assert_same(two, one)
    assert MAX == 13  # a ring of 16 rows, groups of 4


def test_the_layer_passes_its_rows_bound_only_under_the_variant(monkeypatch):
    pytest.importorskip("aiter", reason="the layer test case builds an attention impl")
    from test_minimax_m3_pro_kernels import DT, VD, Case, H, TestImplGlue

    from atom.model_ops.minimax_m3.pro_attn import ProVariants

    seen = []
    real = K.compress_insert
    monkeypatch.setattr(
        K, "compress_insert", lambda *a, **kw: seen.append(kw) or real(*a, **kw)
    )
    c = Case([6, 6], topk=8, seed=7)
    for fused in (False, True):
        v = ProVariants(index_compress=True, topk="fused", index_compress_fused=fused)
        TestImplGlue()._impl_with_rope(c, v)
        ic_c = torch.zeros(c.ic.shape[0], K.PAGE // K.GROUP, K.IDX_DIM).to(c.qkv)
        ring = (torch.tensor([0, K.RING_ROWS], dtype=torch.int32).to(c.seq.device), 64)
        out = torch.zeros(c.T, H * VD, dtype=DT, device=c.qkv.device)
        args = (c.qkv, c.pos, c.seq, c.slot, c.bt, c.layer, ic_c, c.bt.shape[1], v)
        c.impl._kernel_attention(*args, out, static=True, ring=ring, max_rows=6)
    assert [kw["max_rows"] for kw in seen] == [0, 6]
