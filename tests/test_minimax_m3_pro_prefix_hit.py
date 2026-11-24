# SPDX-License-Identifier: MIT
"""A prefix-cache hit on MiniMax-M3-Pro's caches computes what a cold prefill computes.

A hit skips the forward of the cached tokens, so every per-token side effect they had
(K row, K scale, index-K row) must already sit in the shared blocks, and nothing a new
token computes may depend on anything else the skipped forward did. The layout comes from
the real ``BlockManager``: request A prefills every token and publishes its blocks, then a
second request is admitted through the prefix walk, claims A's blocks for the hit and
fresh blocks (holding a recycled pool's garbage) for the rest, and runs
``ProSparseAttentionImpl._kernel_attention`` on its uncached rows only:

  B  the same prompt again (a full hit: the walk keeps the last block back), or a prompt
     that leaves A's after ``keep`` tokens (a half hit);
  C  A's rows re-prefilled cold in chunks that are not block aligned;
  V  after a half hit, the last ``k + 1`` rows as a static DSpark verify (k = 7): first
     with junk rows (a rejected draft), then with the real ones.

Each must select the same blocks for a row as A did (a verify: as a verify on A's own
cache), write A's bytes for its new tokens, leave the shared blocks byte for byte as they
were, and give A's attention output within ``test_minimax_m3_pro_kernels._close``.

Without a GPU the kernels run in Triton's interpreter (the small variants); the served
variants (top-128 over contexts past 128 blocks, day0 prefill, aiter's K-only decode,
FlyDSL's sparse decode, the indexer split over 8 emulated ranks) need a GPU and the
packages they run on.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

if not torch.cuda.is_available():
    os.environ["TRITON_INTERPRET"] = "1"

pytest.importorskip("triton")
pytest.importorskip("aiter", reason="pro_attn imports the ATOM attention base")

from types import SimpleNamespace

import test_minimax_m3_pro_kernels as T
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.sequence import Sequence
from atom.model_ops.minimax_m3 import pro_kernels as K
from atom.model_ops.minimax_m3.pro_attn import ProVariants, row_split_index
from atom.sampling_params import SamplingParams

DEV, DT, H, VD, PG, FP8 = T.DEV, T.DT, T.H, T.VD, T.PG, T.FP8
NV = 8  # rows of a DSpark verify: k = 7 drafts plus the anchor

# Sizes. ``lens`` holds one length that is a whole number of blocks, so a walk that hands
# back the last block leaves nothing to recompute; ``keep`` (whole blocks) is where a half
# hit's prompt leaves A's.
SMALL = {"lens": [200, 192], "keep": [96, 64], "topk": 8}
# Past 128 blocks, so the top-128 selection runs.
L8 = [2600, 2208, 2321, 2100, 2477, 2064, 2333, 2401]
K8 = [1600, 1024, 2048, 1056, 2400, 1040, 1200, 2000]
BIG2 = {"lens": L8[:2], "keep": K8[:2], "topk": K.TOPK}
BIG3 = {"lens": L8[:3], "keep": K8[:3], "topk": K.TOPK}
BIG8 = {"lens": L8, "keep": K8, "topk": K.TOPK}

KONLY8 = {"k_only": True, "kv": FP8, "ic": FP8}  # what the Agentic recipe serves
SERVED = {
    "attn": "aiter",
    "topk": "split",
    "score_tq": 8,
    "k_plain_fp8": True,
    "prefill_attn": "day0",
}

# name: (sizes, Case kwargs, ProVariants kwargs, GPU-only spec or None). A spec names the
# packages the variant needs, the attention the prefill and the verify must reach, and the
# emulated TP ranks of the prefill / verify indexer split.
VARIANTS = {
    "bf16_torch_topk": (SMALL, {}, {}, None),
    "konly_split": (SMALL, {"k_only": True}, {"topk": "split", "attn": "aiter"}, None),
    "konly_split_tq8": (
        SMALL,
        {"k_only": True},
        {"topk": "split", "attn": "aiter", "score_tq": 8},
        None,
    ),
    "konly_fp8plain_fp8idx_tq8": (
        SMALL,
        KONLY8,
        {"topk": "split", "attn": "aiter", "score_tq": 8, "k_plain_fp8": True},
        None,
    ),
    "day0_prefill": (
        BIG2,
        KONLY8,
        {**SERVED, "day0_decode_rows": 0},
        {"needs": ("day0",), "pre": "attention_day0", "ver": "attention"},
    ),
    "aiter_verify": (
        BIG3,
        KONLY8,
        {**SERVED, "day0_decode_rows": 0},
        {"needs": ("day0", "rows"), "pre": "attention_day0", "ver": "attention_aiter"},
    ),
    "flydsl_verify": (
        BIG3,
        KONLY8,
        SERVED,
        {
            "needs": ("day0", "day0_decode"),
            "pre": "attention_day0",
            "ver": "attention_day0_decode",
        },
    ),
    "index_split8": (
        BIG8,
        KONLY8,
        {**SERVED, "index_split": True, "index_split_decode": 64},
        {
            "needs": ("day0", "day0_decode"),
            "pre": "attention_day0",
            "ver": "attention_day0_decode",
            "ranks": 8,
        },
    ),
}
_NEEDS = {
    "day0": K.day0_prefill_available,
    "day0_decode": K.day0_decode_available,
    "rows": K.aiter_value_rows_available,
}


def _params():
    # No `needs_fp8` skip: the interpreter's fp8 rounding bug (31.5 -> 16) is the same in
    # every run here, and these tests compare runs with each other, not with torch.
    out = []
    for name, (size, ck, vk, spec) in VARIANTS.items():
        marks = [] if spec is None else [pytest.mark.skipif(not T.GPU, reason="GPU")]
        out.append(pytest.param(size, ck, vk, spec, id=name, marks=marks))
    return out


class Spy:
    """Records the selection every attention launch is handed, then runs it."""

    NAMES = ("attention", "attention_aiter", "attention_day0", "attention_day0_decode")

    def __init__(self, monkeypatch):
        self.calls = []
        for name in self.NAMES:
            monkeypatch.setattr(K, name, self._wrap(name, getattr(K, name)))

    def _wrap(self, name, real):
        def run(*a, **kw):
            if name == "attention_aiter":
                sel = (a[3].clone(), a[4].clone())
            else:
                sel = None if a[5] is None else a[5].clone()
            self.calls.append((name, sel))
            return real(*a, **kw)

        return run

    def take(self):
        calls, self.calls = self.calls, []
        return calls


def _blocks(call, bt, seq, pos):
    """A launch as ``(kernel, each row's sorted logical blocks)``: aiter's page tables are
    mapped back through ``bt``; only causal blocks count (what the attention reads);
    ``None`` for a dense launch."""
    name, sel = call
    if sel is None:
        return name, None
    seq, pos = seq.tolist(), pos.tolist()
    if name != "attention_aiter":
        rows = sel.tolist()
        return name, [
            sorted(x for x in r if 0 <= x <= p // PG) for r, p in zip(rows, pos)
        ]
    pages, ctx = (t.tolist() for t in sel)
    last = {}
    for s, p in zip(seq, pos):
        last[s] = max(last.get(s, 0), p // PG)
    inv = {
        s: {b: j for j, b in enumerate(bt[s, : n + 1].tolist())}
        for s, n in last.items()
    }
    return name, [
        sorted(inv[s][b] for b in pr[: -(-c // PG)])
        for s, pr, c in zip(seq, pages, ctx)
    ]


def _same(a, b):
    return all(x is None or torch.equal(x, y) for x, y in zip(a, b))


def _garbage_like(t, g):
    """Finite garbage of ``t``'s shape and dtype: what a recycled block holds."""
    if t.element_size() == 1:  # fp8
        x = torch.randn(t.shape, generator=g) * 4
        return x.clamp(-400, 400).to(t.dtype).to(DEV)
    if t.dtype == torch.float32 and t.dim() == 3 and t.shape[1] == 1:  # scale planes
        return (torch.rand(t.shape, generator=g) + 0.5).to(DEV)
    return (torch.randn(t.shape, generator=g) * 3).to(t.dtype).to(DEV)


class Pool:
    """A's case on a ``BlockManager``'s pool: caches of the pool's size full of garbage,
    A's sequences admitted (cold) and their block table in the case."""

    def __init__(self, size, ck, seed=21):
        c = T.Case(
            size["lens"],
            topk=size["topk"],
            seed=seed,
            kv_dtype=ck.get("kv"),
            ic_dtype=ck.get("ic"),
            k_only=ck.get("k_only", False),
        )
        c.impl.k_only = ck.get("k_only", False)
        T.TestImplGlue()._impl_with_rope(c, ProVariants())
        nblk = sum(-(-n // PG) for n in c.lens)
        self.n_blocks = 2 * nblk + 4
        self.bm = BlockManager(
            MockConfig(
                kv_cache_block_size=PG,
                num_kvcache_blocks=self.n_blocks,
                enable_prefix_caching=True,
                max_model_len=max(c.lens) + PG,
            )
        )
        g = torch.Generator().manual_seed(seed + 1)

        for name in ("kc4", "ks", "ic", "vc4", "vs"):
            t = getattr(c, name)
            if t is not None:
                shape = (self.n_blocks, *t.shape[1:])
                t = _garbage_like(torch.empty(shape, dtype=t.dtype), g)
            setattr(c, name, t)
        c.layer = SimpleNamespace(
            k_cache=c.kc4.unsqueeze(1),
            v_cache=(c.kc4 if c.vc4 is None else c.vc4).unsqueeze(1),
            k_scale=c.ks,
            v_scale=c.ks if c.vc4 is None else c.vs,
        )
        c.impl.index_cache = c.ic
        self.toks = [
            list(range(10_000 * (s + 1), 10_000 * (s + 1) + n))
            for s, n in enumerate(c.lens)
        ]
        self.a = self.admit(self.toks)
        assert all(s.num_cached_tokens == 0 for s in self.a)
        c.bt = self.table(self.a)
        c.slot = self.slots(c.bt, c.seq, c.pos)
        self.c = c
        self.nb = max(-(-n // PG) for n in c.lens)

    def admit(self, prompts):
        seqs = []
        for p in prompts:
            s = Sequence(p, PG, sampling_params=SamplingParams())
            hit = self.bm.can_allocate(s)
            assert hit >= 0, "the pool is too small"
            self.bm.allocate(s, hit)
            seqs.append(s)
        return seqs

    @staticmethod
    def table(seqs):
        bt = torch.zeros(
            len(seqs), max(len(s.block_table) for s in seqs), dtype=torch.int32
        )
        for i, s in enumerate(seqs):
            bt[i, : len(s.block_table)] = torch.tensor(list(s.block_table))
        return bt.to(DEV)

    def half_prompts(self, keep):
        """A's prompts up to ``keep[s]`` tokens, new tokens after."""
        return [
            t[:k] + list(range(90_000 + 10_000 * s, 90_000 + 10_000 * s + len(t) - k))
            for s, (t, k) in enumerate(zip(self.toks, keep))
        ]

    @staticmethod
    def shared(seqs):
        """The block ids ``seqs`` hold from the prefix cache."""
        ids = {
            i for s in seqs for i in list(s.block_table)[: s.num_cached_tokens // PG]
        }
        return torch.tensor(sorted(ids), device=DEV)

    @staticmethod
    def slots(bt, seq, pos):
        return bt[seq, pos // PG].long() * PG + pos % PG

    def publish(self, seqs):
        """What the scheduler does once a prefill forward has run."""
        for s in seqs:
            self.bm.hash_blocks(s, s.num_prompt_tokens - s.num_cached_tokens)

    def rows(self, lo, hi=None):
        """Global row indices of tokens ``[lo[s], hi[s])`` of each sequence, batch order."""
        c, idx = self.c, []
        for s in range(c.S):
            a, b = c.seq_slice(s)
            idx += list(range(a + lo[s], b if hi is None else a + hi[s]))
        return torch.tensor(idx, device=DEV, dtype=torch.long)

    def bytes_of(self, slot):
        """Per-token cache bytes at ``slot``: the K row, its scale, the index row."""
        c, blk, off = self.c, slot // PG, slot % PG
        k = c.kc4[blk, :, off, :].reshape(blk.shape[0], -1).view(torch.uint8)
        s = None if c.ks is None else c.ks[blk, 0, off].clone()
        i = c.ic[blk, off].reshape(blk.shape[0], -1).view(torch.uint8)
        return k.clone(), s, i.clone()

    def snapshot(self, ids):
        c = self.c
        return [
            None if t is None else t[ids].view(torch.uint8).clone()
            for t in (c.kc4, c.ks, c.ic, c.vc4, c.vs)
        ]


def _run(p, v, idx, bt, slot, spy, qkv=None, static=False, ranks=0, monkeypatch=None):
    """``_kernel_attention`` over rows ``idx`` (``qkv``: their rows, A's by default) with
    block table ``bt``: ``(output, the launch's selection as logical blocks)``. With
    ``ranks``, as that many TP ranks run it (the prefill split or, ``static``, the decode
    split, each on its own share), the all-gather emulated: every rank must agree."""
    c = p.c
    qkv = c.qkv[idx] if qkv is None else qkv
    seq, pos = c.seq[idx], c.pos[idx]
    nb = bt.shape[1] if static else p.nb

    def call(split=None, dsplit=None):
        out = torch.zeros(idx.shape[0], H * VD, dtype=DT, device=DEV)
        c.impl._kernel_attention(
            qkv,
            pos,
            seq,
            slot,
            bt,
            c.layer,
            c.ic,
            nb,
            v,
            out,
            static=static,
            split=split,
            dsplit=dsplit,
        )
        return out

    if not ranks:
        spy.take()
        out = call()
        (launch,) = spy.take()
        return out, _blocks(launch, bt, seq, pos)
    n = idx.shape[0]

    def rank(r):
        if static:
            per = n // ranks
            return call(dsplit=(r * per, (r + 1) * per))
        ridx, seg = row_split_index(n, ranks, r, DEV)
        return call(split=(ridx, seg, n, ranks))

    comm = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "aiter.dist.communication_op", comm)
    shares = []
    comm.tensor_model_parallel_all_gather = lambda x, dim=0: (
        shares.append(x.clone()),
        x.repeat(ranks, *[1] * (x.dim() - 1)),
    )[1]
    for r in range(ranks):
        rank(r)
    mine = torch.cat(shares)
    comm.tensor_model_parallel_all_gather = lambda x, dim=0: mine
    spy.take()
    outs = [rank(r) for r in range(ranks)]
    sels = [_blocks(launch, bt, seq, pos) for launch in spy.take()]
    assert len(sels) == ranks and all(s == sels[0] for s in sels[1:])
    assert all(torch.equal(o, outs[0]) for o in outs)
    return outs[0], sels[0]


def _setup(size, ck, vk, spec, monkeypatch):
    if spec is None:
        if not T.GPU:
            # aiter's paged attention and top-k need a GPU: their torch stand-ins, and
            # the K-only cache on the Triton attention
            monkeypatch.setattr(K, "_pa_decode", T.RefPA())
            monkeypatch.setattr(K, "_default_partitions", lambda t: 3)
            monkeypatch.setattr(K, "_topk_rows", T.ref_topk_rows)
            monkeypatch.setattr(K, "aiter_value_rows_available", lambda: False)
    else:
        for need in spec["needs"]:
            if not _NEEDS[need]():
                pytest.skip(f"needs {need}")
    p, v, spy = Pool(size, ck), ProVariants(**vk), Spy(monkeypatch)
    # A: every token, cold; its blocks are published once its forward has run
    p.full, p.full_sel = _run(
        p, v, torch.arange(p.c.T, device=DEV), p.c.bt, p.c.slot, spy
    )
    if spec is not None:
        assert p.full_sel[0] == spec["pre"]
    p.publish(p.a)
    return p, v, spy


def _check_new_rows(p, idx, slot):
    for a, b in zip(p.bytes_of(p.c.slot[idx]), p.bytes_of(slot)):
        assert (a is None) == (b is None)
        assert a is None or torch.equal(a, b), "a new token's cache bytes differ"


def _sub(rows, idx):
    """Rows ``idx`` of a launch over every token (``None``: dense)."""
    return None if rows is None else [rows[i] for i in idx.tolist()]


@pytest.mark.parametrize("half", [False, True], ids=["hit", "half"])
@pytest.mark.parametrize("size,ck,vk,spec", _params())
def test_prefix_hit_equals_cold_prefill(size, ck, vk, spec, half, monkeypatch):
    p, v, spy = _setup(size, ck, vk, spec, monkeypatch)
    c = p.c
    keep = size["keep"]
    b = p.admit(p.half_prompts(keep) if half else p.toks)
    cached = [s.num_cached_tokens for s in b]
    for n, k, got in zip(c.lens, keep, cached):
        if half:
            assert got == k
        else:
            # the walk keeps the last block back: the sampler needs the last row
            assert n - PG <= got < n, (n, got)
    for sa, sb, got in zip(p.a, b, cached):
        assert list(sb.block_table)[: got // PG] == list(sa.block_table)[: got // PG]
    bt = p.table(b)
    shared = p.shared(b)
    before = p.snapshot(shared)
    idx = p.rows(cached)
    slot = p.slots(bt, c.seq[idx], c.pos[idx])
    got, sel = _run(
        p,
        v,
        idx,
        bt,
        slot,
        spy,
        ranks=(spec or {}).get("ranks", 0),
        monkeypatch=monkeypatch,
    )
    assert not torch.isnan(got).any()
    assert _same(p.snapshot(shared), before), "a hit prefill wrote into a shared block"
    if spec is not None:
        assert sel[0] == spec["pre"]
    assert sel[1] == _sub(p.full_sel[1], idx)
    _check_new_rows(p, idx, slot)
    assert T._close(got, p.full[idx])


@pytest.mark.parametrize("size,ck,vk,spec", _params())
def test_chunked_cold_prefill_equals_one_prefill(size, ck, vk, spec, monkeypatch):
    p, v, spy = _setup(size, ck, vk, spec, monkeypatch)
    c = p.c
    fresh = [
        list(range(500_000 + 10_000 * s, 500_000 + 10_000 * s + n))
        for s, n in enumerate(c.lens)
    ]
    b = p.admit(fresh)
    assert all(s.num_cached_tokens == 0 for s in b)
    bt = p.table(b)
    cuts = [[0, n // 3 + 5, 2 * n // 3 + 3, n] for n in c.lens]  # not block aligned
    got = torch.zeros_like(p.full)
    for step in range(3):
        idx = p.rows([cu[step] for cu in cuts], [cu[step + 1] for cu in cuts])
        slot = p.slots(bt, c.seq[idx], c.pos[idx])
        got[idx], sel = _run(
            p,
            v,
            idx,
            bt,
            slot,
            spy,
            ranks=(spec or {}).get("ranks", 0),
            monkeypatch=monkeypatch,
        )
        assert sel[1] == _sub(p.full_sel[1], idx)
        _check_new_rows(p, idx, slot)
    assert T._close(got, p.full)


@pytest.mark.parametrize("size,ck,vk,spec", _params())
def test_rejected_verify_after_hit_leaves_no_trace(size, ck, vk, spec, monkeypatch):
    p, v, spy = _setup(size, ck, vk, spec, monkeypatch)
    c = p.c
    ranks = (spec or {}).get("ranks", 0)
    tail = p.rows([n - NV for n in c.lens])
    seq, pos = c.seq[tail], c.pos[tail]
    # the verify on A's own cache (its prep rewrites A's bytes): the cold twin
    wide_a = torch.cat([c.bt, torch.zeros_like(c.bt)], dim=1)  # a graph's table width
    cold, cold_sel = _run(p, v, tail, wide_a, c.slot[tail], spy, static=True)
    if spec is not None:
        assert cold_sel[0] == spec["ver"]
    keep = size["keep"]
    b = p.admit(p.half_prompts(keep))
    assert [s.num_cached_tokens for s in b] == keep
    bt = p.table(b)
    shared = p.shared(b)
    before = p.snapshot(shared)
    head = p.rows(keep, [n - NV for n in c.lens])
    _run(p, v, head, bt, p.slots(bt, c.seq[head], c.pos[head]), spy)
    wide = torch.cat([bt, torch.zeros_like(bt)], dim=1)
    slot = p.slots(bt, seq, pos)
    junk = (
        (torch.randn(c.qkv[tail].shape, generator=torch.Generator().manual_seed(5)) * 2)
        .to(DT)
        .to(DEV)
    )
    _run(
        p,
        v,
        tail,
        wide,
        slot,
        spy,
        qkv=junk,
        static=True,
        ranks=ranks,
        monkeypatch=monkeypatch,
    )
    assert _same(p.snapshot(shared), before), "a rejected verify wrote a shared block"
    got, sel = _run(
        p, v, tail, wide, slot, spy, static=True, ranks=ranks, monkeypatch=monkeypatch
    )
    assert _same(p.snapshot(shared), before)
    assert sel[1] == cold_sel[1]
    if spec is not None:
        assert sel[0] == spec["ver"]
    _check_new_rows(p, tail, slot)
    assert T._close(got, cold)
