# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro prep: aiter's ``fused_qk_norm_rope_cache_mqa`` (``pro_kernels.prep_aiter``)
against the Triton ``prep`` and the torch reference, on a GPU.

Both kernels get the same raw rows, Pro's own YaRN rope (x48 of 24576, theta 5e5, rotary
64; the "hf" magnitude ratio) and caches pre-filled with garbage, over TP head counts 16 /
128, bf16 / fp8 KV and index caches, decode and prefill token counts, padding slots (-1)
and positions past 24576, and the recipe's K-only plain fp8 cache. Checked: q / index q, the
K / V / index-K rows and fp8 scales the caches received, and that nothing else in the caches
changed.

    pytest -q tests/test_minimax_m3_pro_prep_aiter.py
    python tests/test_minimax_m3_pro_prep_aiter.py --bench [--heads 16] [--kv fp8] [--k-only]
"""

from __future__ import annotations

import argparse
import copy
import functools
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("triton")
pytest.importorskip("aiter")

from aiter import dtypes

from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.minimax_m3 import pro_kernels as K
from atom.model_ops.minimax_m3.pro_attn import ProSparseAttentionImpl

if not K.aiter_prep_available():
    pytest.skip(
        "this aiter has no fused_qk_norm_rope_cache_mqa", allow_module_level=True
    )

D, NI, DI, PG, VD, ROT = 256, 16, 128, 16, 192, 64
DEV = "cuda"
DT = torch.bfloat16
FP8 = dtypes.fp8
EPS = 1e-6
MAXPOS = 65536  # past YaRN's original 24576


def _rope_tables():
    from aiter.rotary_embedding import YaRNScalingRotaryEmbedding

    r = YaRNScalingRotaryEmbedding(D, ROT, 24576, 500000, True, 48.0, DT)
    cos = r.cos_cache.view(r.cos_cache.shape[0], -1)[:MAXPOS].contiguous().to(DEV)
    sin = r.sin_cache.view(r.sin_cache.shape[0], -1)[:MAXPOS].contiguous().to(DEV)
    return cos, sin, 1.0 / r.mscale  # ProVariants(mscale="hf"): target 1.0


_TABLES = None


def tables():
    global _TABLES
    if _TABLES is None:
        _TABLES = _rope_tables()
    return _TABLES


def _garbage(shape, dt, g):
    """Random bytes (NaN patterns included) for a cache the kernels must partly leave alone."""
    if dt == torch.float32:
        return torch.rand(shape, generator=g).to(DEV)
    n = dt.itemsize
    raw = torch.randint(
        0, 256, (*shape[:-1], shape[-1] * n), generator=g, dtype=torch.int32
    )
    return raw.to(torch.uint8).view(dt).to(DEV)


class Case:
    def __init__(
        self,
        t,
        heads,
        kv="bf16",
        ic="bf16",
        packed=False,
        write_v=True,
        slot_dtype=torch.int64,
        pad=0.15,
        seed=0,
        k_only=False,
    ):
        """``k_only``: the recipe's K-only row cache (``ProKOnlyLayout``: x = 256, no V), an
        fp8 one a plain cast (``k_plain_fp8``: ``prep``'s ``k_unit``)."""
        g = torch.Generator().manual_seed(seed)
        self.t, self.heads = t, heads
        rn = lambda *s: (torch.randn(*s, generator=g) * 2).to(DT).to(DEV)
        widths = (heads * D, D, NI * DI, DI)
        if packed:
            self.qkv = rn(t, sum(widths))
            self.parts = K._split_rows(self.qkv, heads)
        else:
            self.parts = tuple(rn(t, w) for w in widths)
            self.qkv = self.parts
        nblk = max(4, 2 * ((t + PG - 1) // PG))
        slot = torch.randperm(nblk * PG, generator=g)[:t]
        drop = torch.rand(t, generator=g) < pad
        drop[0] = pad >= 1.0
        slot[drop] = -1
        self.slot = slot.to(slot_dtype).to(DEV)
        self.pos = torch.randint(0, MAXPOS, (t,), generator=g).to(DEV)
        self.fp8 = kv == "fp8"
        self.k_unit = k_only and self.fp8
        kvdt = FP8 if self.fp8 else DT
        x = 16 if self.fp8 else 8
        kx = D if k_only else x
        self.kc4 = _garbage((nblk, D // kx, PG, kx), kvdt, g)
        write_v = write_v and not k_only
        self.vc4 = _garbage((nblk, PG // x, D, x), kvdt, g) if write_v else None
        self.ks = _garbage((nblk, 1, PG), torch.float32, g) if self.fp8 else None
        self.vs = _garbage((nblk, 1, PG), torch.float32, g) if self.fp8 else None
        self.ic = _garbage((nblk, PG, DI), FP8 if ic == "fp8" else DT, g)
        mk = lambda n: (torch.randn(n, generator=g) * 0.1).to(DT).to(DEV)
        self.wq, self.wk, self.wiq, self.wik = mk(D), mk(D), mk(DI), mk(DI)
        self.x = x

    def caches(self):
        c = lambda a: None if a is None else a.clone()
        return c(self.kc4), c(self.vc4), c(self.ks), c(self.vs), c(self.ic)

    def run(self, fn):
        kc4, vc4, ks, vs, ic = self.caches()
        cos, sin, ratio = tables()
        q, iq = fn(
            self.qkv,
            self.pos,
            self.slot,
            cos,
            sin,
            ratio,
            self.wq,
            self.wk,
            self.wiq,
            self.wik,
            EPS,
            kc4,
            ic,
            self.heads,
            vc4=vc4,
            k_scale=ks,
            v_scale=vs,
            k_unit=self.k_unit,
        )
        torch.cuda.synchronize()
        return SimpleNamespace(q=q, iq=iq, kc4=kc4, vc4=vc4, ks=ks, vs=vs, ic=ic)

    # ---- the torch reference (pro_attn's) ------------------------------------------------
    def ref(self):
        m = object.__new__(ProSparseAttentionImpl)
        m.rotary_dim = ROT
        cos, sin, ratio = tables()
        c, s = cos[self.pos].float() * ratio, sin[self.pos].float() * ratio
        q, k, iq, ik = self.parts
        t, h = self.t, self.heads
        nr = lambda w, x: GemmaRMSNorm.forward_static(w, EPS, x, None)
        return SimpleNamespace(
            q=m._rope(nr(self.wq, q[:, : h * D].reshape(t, h, D)), c, s),
            k=m._rope(nr(self.wk, k[:, :D].reshape(t, 1, D)), c, s).reshape(t, D),
            iq=m._rope(nr(self.wiq, iq[:, : NI * DI].reshape(t, NI, DI)), c, s),
            ik=m._rope(nr(self.wik, ik[:, :DI].reshape(t, 1, DI)), c, s).reshape(t, DI),
        )

    # ---- what the caches hold for the written tokens -------------------------------------
    def live(self):
        return (self.slot >= 0).nonzero().flatten()

    def read(self, r):
        """K [n, 256], V [n, 192], index K [n, 128] (cache dtype) and the scales of the
        written tokens."""
        tok = self.live()
        s = self.slot[tok].long()
        blk, off = s // PG, s % PG
        k = r.kc4[blk, :, off, :].reshape(-1, D)
        v = None if r.vc4 is None else r.vc4[blk, off // self.x, :VD, off % self.x]
        ik = r.ic[blk, off]
        ks = None if r.ks is None else r.ks[blk, 0, off]
        vs = None if r.vs is None or r.vc4 is None else r.vs[blk, 0, off]
        return SimpleNamespace(k=k, v=v, ik=ik, ks=ks, vs=vs, tok=tok)

    def written_masks(self):
        """Per cache, the elements a token writes."""
        n = self.kc4.shape[0]
        w = torch.zeros(n * PG, dtype=torch.bool, device=DEV)
        w[self.slot[self.live()].long()] = True
        w = w.view(n, PG)
        x = self.x
        out = {
            "kc4": w[:, None, :, None].expand_as(self.kc4),
            "ic": w[:, :, None].expand_as(self.ic),
        }
        if self.vc4 is not None:
            vm = w.view(n, PG // x, 1, x).expand(n, PG // x, D, x).clone()
            vm[:, :, VD:, :] = False
            out["vc4"] = vm
        if self.ks is not None:
            out["ks"] = w[:, None, :].expand_as(self.ks)
            if self.vc4 is not None:
                out["vs"] = out["ks"]
        return out


def _raw(x):
    return x.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[x.element_size()])


def _close16(a, b, what, frac=1e-3):
    """bf16 / fp16 results: at most ``frac`` differ, and those by a rounding step or two
    (summation order / fma contraction may move a value across a rounding boundary)."""
    a, b = a.float(), b.float()
    diff = (a != b).float().mean().item()
    assert diff <= frac, f"{what}: {diff:.2e} of the values differ"
    torch.testing.assert_close(
        a, b, rtol=2**-7, atol=2**-6, msg=lambda m: f"{what}: {m}"
    )


def _close8(a, b, what, frac=1e-2):
    """fp8 codes (as their values): mostly equal, else one fp8 step apart."""
    a, b = a.float(), b.float()
    diff = (a != b).float().mean().item()
    assert diff <= frac, f"{what}: {diff:.2e} of the codes differ"
    torch.testing.assert_close(
        a, b, rtol=0.13, atol=2**-6, msg=lambda m: f"{what}: {m}"
    )


def _close(a, b, what):
    if a.numel() == 0:
        return
    diff = (a.float() != b.float()).float().mean().item()
    if diff:
        print(f"[diff] {what} {tuple(a.shape)}: {diff:.3e} differ")
    (_close8 if a.element_size() == 1 else _close16)(a, b, what)


def compare(c: Case, a, b, what):
    """``a`` / ``b``: two ``Case.run`` results."""
    _close(a.q, b.q, f"{what} q")
    _close(a.iq, b.iq, f"{what} index q")
    ra, rb = c.read(a), c.read(b)
    _close(ra.k, rb.k, f"{what} K")
    _close(ra.ik, rb.ik, f"{what} index K")
    if ra.v is not None:
        _close(ra.v, rb.v, f"{what} V")
    if ra.ks is not None:
        torch.testing.assert_close(
            ra.ks, rb.ks, rtol=2**-7, atol=0, msg=lambda m: f"{what} k_scale: {m}"
        )
    if ra.vs is not None:
        torch.testing.assert_close(
            ra.vs, rb.vs, rtol=2**-7, atol=0, msg=lambda m: f"{what} v_scale: {m}"
        )


def check_untouched(c: Case, r, what):
    for name, mask in c.written_masks().items():
        got, orig = _raw(getattr(r, name)), _raw(getattr(c, name))
        assert torch.equal(
            got[~mask], orig[~mask]
        ), f"{what}: {name} changed outside the written rows"
    for name in ("ks", "vs"):
        if getattr(c, name) is not None and name not in c.written_masks():
            assert torch.equal(
                _raw(getattr(r, name)), _raw(getattr(c, name))
            ), f"{what}: {name} written"


def check_vs_ref(c: Case, r, what):
    ref = c.ref()
    _close16(r.q, ref.q, f"{what} q vs torch")
    _close16(r.iq, ref.iq, f"{what} index q vs torch")
    rd = c.read(r)
    k = ref.k[rd.tok]
    if c.fp8:
        k, _ = K.quant_fp8_rows(k, FP8, plain=c.k_unit)
    _close(rd.k, k.to(rd.k.dtype), f"{what} K vs torch")
    ik = ref.ik[rd.tok]
    ik = (
        ik.float().clamp(-torch.finfo(rd.ik.dtype).max, torch.finfo(rd.ik.dtype).max)
        if rd.ik.element_size() == 1
        else ik
    )
    _close(rd.ik, ik.to(rd.ik.dtype), f"{what} index K vs torch")


TOKENS = [1, 3, 64, 1031]


@pytest.mark.parametrize("ic", ["bf16", "fp8"])
@pytest.mark.parametrize("kv", ["bf16", "fp8"])
@pytest.mark.parametrize("heads", [16, 128])
@pytest.mark.parametrize("t", TOKENS)
def test_aiter_prep_matches_triton_and_torch(t, heads, kv, ic):
    c = Case(t, heads, kv, ic, seed=t + heads)
    got = c.run(K.prep_aiter)
    check_untouched(c, got, "aiter")
    check_vs_ref(c, got, "aiter")
    try:
        want = c.run(K.prep)
    except (
        Exception
    ) as e:  # pragma: no cover - Triton's prep holds every head in one program
        if heads <= 32:
            raise
        pytest.skip(f"Triton prep at {heads} heads: {type(e).__name__}: {e}")
    check_untouched(c, want, "triton")
    compare(c, got, want, "aiter vs triton")


@pytest.mark.parametrize("kv", ["bf16", "fp8"])
def test_packed_rows_equal_separate_rows(kv):
    """The ``[q | k | index q | index k]`` rows of one tensor (each part read at its own
    offset / the packed stride) give what the four separate projections give."""
    a = Case(77, 16, kv, packed=True, seed=3)
    b = copy.copy(a)
    widths = (16 * D, D, NI * DI, DI)
    b.parts = tuple(p[:, :w].contiguous() for p, w in zip(a.parts, widths))
    b.qkv = b.parts
    ra, rb = a.run(K.prep_aiter), b.run(K.prep_aiter)
    for name in ("q", "iq", "kc4", "vc4", "ks", "vs", "ic"):
        x, y = getattr(ra, name), getattr(rb, name)
        assert (x is None and y is None) or torch.equal(_raw(x), _raw(y)), name


@pytest.mark.parametrize("kv", ["bf16", "fp8"])
def test_without_v_cache(kv):
    c = Case(40, 16, kv, write_v=False, seed=5)
    got, want = c.run(K.prep_aiter), c.run(K.prep)
    check_untouched(c, got, "aiter")
    compare(c, got, want, "aiter vs triton (no V)")


def test_int32_slots_and_prefill_4096():
    c = Case(4096, 16, "fp8", "fp8", slot_dtype=torch.int32, seed=6)
    got, want = c.run(K.prep_aiter), c.run(K.prep)
    check_untouched(c, got, "aiter")
    compare(c, got, want, "aiter vs triton (int32 slots)")


def test_int32_positions_and_slots_equal_int64():
    c = Case(64, 16, "fp8", "bf16", seed=7)
    c32 = copy.copy(c)
    c32.pos, c32.slot = c.pos.int(), c.slot.int()
    a, b = c.run(K.prep_aiter), c32.run(K.prep_aiter)
    for name in ("q", "iq", "kc4", "vc4", "ks", "vs", "ic"):
        assert torch.equal(_raw(getattr(a, name)), _raw(getattr(b, name))), name


@pytest.mark.parametrize("ic", ["bf16", "fp8"])
@pytest.mark.parametrize("t", [1, 8, 16, 24, 64, 1031])
def test_k_only_plain_fp8_matches_triton(t, ic):
    """The recipe's K-only plain fp8 cache (``k_plain_fp8``), packed rows as the model passes
    them: aiter's ``kv_unit_scale`` writes the bytes of Triton's ``k_unit`` -- K codes, the
    1.0 scales, index K -- at decode row counts, within ``_close`` at prefill sizes."""
    if not K.aiter_prep_available(k_unit=True):
        pytest.skip("this aiter's fused_qk_norm_rope_cache_mqa has no kv_unit_scale")
    c = Case(t, 16, "fp8", ic, packed=True, seed=t + 11, k_only=True)
    got, want = c.run(K.prep_aiter), c.run(K.prep)
    check_untouched(c, got, "aiter")
    check_untouched(c, want, "triton")
    check_vs_ref(c, got, "aiter")
    rg, rw = c.read(got), c.read(want)
    assert torch.equal(rg.ks, torch.ones_like(rg.ks)), "k_scale is not 1.0"
    for name in ("q", "iq"):
        a, b = getattr(got, name), getattr(want, name)
        print(f"[frac] t={t} ic={ic} {name}: {(a != b).float().mean().item():.3e}")
    if t <= 64:
        for name in ("k", "ks", "ik"):
            a, b = _raw(getattr(rg, name)), _raw(getattr(rw, name))
            print(f"[frac] t={t} ic={ic} {name}: {(a != b).float().mean().item():.3e}")
            assert torch.equal(a, b), f"{name} differs from Triton's"
    compare(c, got, want, "aiter vs triton (K-only plain fp8)")


def test_all_padding_writes_nothing():
    c = Case(9, 16, "fp8", "fp8", pad=1.1, seed=8)
    got = c.run(K.prep_aiter)
    for name in ("kc4", "vc4", "ks", "vs", "ic"):
        assert torch.equal(_raw(getattr(got, name)), _raw(getattr(c, name))), name


# --------------------------------------------------------------------------- bench --
def _time(fn, iters):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    eager = s.elapsed_time(e) * 1e3 / iters
    per_graph = 20
    g = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    with torch.cuda.graph(g):
        for _ in range(per_graph):
            fn()
    g.replay()
    torch.cuda.synchronize()
    reps = max(1, iters // per_graph)
    s.record()
    for _ in range(reps):
        g.replay()
    e.record()
    torch.cuda.synchronize()
    return eager, s.elapsed_time(e) * 1e3 / (reps * per_graph)


def bench(heads, kv, ic, tokens, iters, k_only=False):
    cos, sin, ratio = tables()
    layout = ", K-only (packed rows)" if k_only else ""
    print(
        f"heads {heads}, KV cache {kv}{layout}, index cache {ic}  (us per call: eager / in a CUDA graph)"
    )
    print(f"{'tokens':>7} {'triton':>17} {'aiter':>17} {'speedup(graph)':>15}")
    for t in tokens:
        c = Case(t, heads, kv, ic, packed=k_only, pad=0.0, seed=t, k_only=k_only)
        kc4, vc4, ks, vs, icc = c.caches()
        args = (
            c.qkv,
            c.pos,
            c.slot,
            cos,
            sin,
            ratio,
            c.wq,
            c.wk,
            c.wiq,
            c.wik,
            EPS,
            kc4,
            icc,
            heads,
        )
        kw = {"vc4": vc4, "k_scale": ks, "v_scale": vs, "k_unit": c.k_unit}
        te, tg = _time(functools.partial(K.prep, *args, **kw), iters)
        ae, ag = _time(functools.partial(K.prep_aiter, *args, **kw), iters)
        print(f"{t:>7} {te:>8.1f} /{tg:>7.1f} {ae:>8.1f} /{ag:>7.1f} {tg / ag:>14.2f}x")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv", default="bf16", choices=["bf16", "fp8"])
    ap.add_argument("--ic", default="bf16", choices=["bf16", "fp8"])
    ap.add_argument("--tokens", default="1,4,16,64,256,1024,4096,16384")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument(
        "--k-only",
        action="store_true",
        help="the recipe's K-only cache (plain fp8 with --kv fp8), packed rows",
    )
    a = ap.parse_args()
    if a.bench:
        tokens = [int(x) for x in a.tokens.split(",")]
        bench(a.heads, a.kv, a.ic, tokens, a.iters, a.k_only)
    else:
        raise SystemExit(pytest.main([__file__, "-q"]))


@pytest.mark.parametrize("wide", [False, True])
@pytest.mark.parametrize("t", [1, 8, 24, 32, 64])
def test_q8_is_the_cast_of_the_bf16_q(t, wide):
    """``prep_aiter(q8=True)`` (``prep_q8``): the E4M3 q the day0 decode reads,
    written by the prep itself, holds the bytes of ``q.to(torch.float8_e4m3fn)`` of its
    bf16 q (the decode's cast, which then is a no-op); every cache and the index q are
    unchanged. ``wide``: q-norm gains from 1e-7 to 3e3, so fp8 denormals and the NaN
    range (>= 480) are hit too."""
    if not K.aiter_prep_q8_available():
        pytest.skip("this aiter's fused_qk_norm_rope_cache_mqa has no E4M3 q_out")
    c = Case(t, 16, "fp8", "fp8", packed=True, seed=t + 21, k_only=True)
    if wide:
        c.wq = (10 ** torch.linspace(-7, 3.5, D) - 1).to(DT).to(DEV)
    want, got = c.run(K.prep_aiter), c.run(functools.partial(K.prep_aiter, q8=True))
    assert got.q.dtype == torch.float8_e4m3fn and got.q.shape == want.q.shape
    assert torch.equal(_raw(got.q), _raw(want.q.to(torch.float8_e4m3fn)))
    for name in ("iq", "kc4", "ks", "ic"):
        assert torch.equal(_raw(getattr(got, name)), _raw(getattr(want, name))), name
    if wide:
        codes = _raw(got.q).view(torch.uint8) & 0x7F
        assert (codes == 0x7F).any() and ((codes > 0) & (codes < 8)).any()
