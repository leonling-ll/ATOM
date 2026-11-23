# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pro mono phase 1's layer kernel, its plain-Python half: the peer regions
and their indexing, the column ownership the o_proj push and the all-reduce
must agree on, the build key, and the torch reference the GPU test holds the
kernel to (checked here against independent formulas)."""

import functools
import itertools
import math

import pytest
import torch
from minimax_m3_pro_mono_ref import (
    RES_RATIOS,
    ar_sum,
    case_inputs,
    code_distance,
    dequant,
    layer_failures,
    layer_reference,
    layer_report,
    mxfp8,
    mxfp8_codes,
    o_partial,
    rows,
    token_sizes,
)

from atom.models.minimax_m3.pro_mono import layout as L
from atom.models.minimax_m3.pro_mono.config import HIDDEN, MAX_ROWS, MAX_ROWS_LAYOUT


def test_tp8_geometry():
    assert (L.O_K, L.OWN, L.NT) == (3072, 896, 56)
    assert (L.O_TASKS, L.O_TASKS_OWN, L.O_CPW, L.O_SCALES_PW) == (224, 28, 12, 24)
    # one o task a CTA, and every AR task on a CTA of its own
    assert L.O_TASKS <= 256 and L.NT <= 256
    # the push's thread maps (token, row of 32) and (token, row pair) fit a CTA
    assert L.O_ROWS * MAX_ROWS <= 512


def test_the_push_and_the_reduce_agree_on_who_owns_a_column():
    """o task t's 32 rows (hidden columns) go to rank t // 28; the AR task j
    owning column c is rank j // 7. Every column must have one owner both ways."""
    owner_by_o = [L.o_owner(c // L.O_ROWS) for c in range(HIDDEN)]
    owner_by_ar = [L.ar_owner(c // 128) for c in range(HIDDEN)]
    assert owner_by_o == owner_by_ar
    assert owner_by_o == [c // L.OWN for c in range(HIDDEN)]


def test_peer_regions_are_aligned_disjoint_and_sized_for_the_layout_rows():
    lay = L.peer_layout()
    spans = sorted(lay.values())
    for (o1, n1), (o2, _) in itertools.pairwise(spans):
        assert o1 + n1 <= o2
    assert all(off % 16 == 0 for off, _ in spans)
    assert L.peer_bytes() % 16 == 0 and L.peer_bytes() >= spans[-1][0] + spans[-1][1]
    # the largest index each region takes, at the layout's row count
    t, src = MAX_ROWS_LAYOUT - 1, L.TP - 1
    assert (L.rs_pair(src, t, L.OWN - 2) + 1) * 8 == lay["rs"][1]
    assert (L.ag_pair(t, HIDDEN - 2) + 1) * 8 == lay["ag"][1]
    assert (L.agsq_pair(L.NT - 1, t) + 2) * 8 == lay["agsq"][1]


def test_pair_indices_are_injective():
    """No two (rank, token, column pair) cells share a mailbox pair."""
    seen = {
        L.rs_pair(src, t, c)
        for src in range(L.TP)
        for t in range(MAX_ROWS_LAYOUT)
        for c in range(0, L.OWN, 2)
    }
    assert len(seen) == L.TP * MAX_ROWS_LAYOUT * L.OWN // 2
    cells = {L.agsq_pair(j, t) for j in range(L.NT) for t in range(MAX_ROWS_LAYOUT)}
    assert len(cells) == L.NT * MAX_ROWS_LAYOUT and all(c % 2 == 0 for c in cells)


def test_build_key():
    assert L.P1Build(tokens=MAX_ROWS).upto == "norm"
    for bad in ({"tokens": 0}, {"tokens": MAX_ROWS + 1}, {"tokens": 1, "upto": "x"}):
        with pytest.raises(ValueError):
            L.P1Build(**bad)
    names = [d.name for d in L.decls(L.P1Build(tokens=1))]
    assert names == list(L.NORM_REGIONS)


def test_debug_scratch_holds_the_wait_records():
    from atom.mono.runtime.debug import DIAG_BYTES

    assert L.scratch_layout(L.P1Build(tokens=1)) == {}
    assert L.scratch_layout(L.P1Build(tokens=1, debug=True))["diag"] == (0, DIAG_BYTES)
    assert L.scratch_bytes(L.P1Build(tokens=1, debug=True)) >= DIAG_BYTES


# ------------------------------------------------------------ the reference


def _code_frexp(amax):
    a = torch.tensor(max(float(amax), 1e-10), dtype=torch.float32) * torch.tensor(
        1.0 / 448.0, dtype=torch.float32
    )
    m, e = math.frexp(float(a))
    return min(254, max(1, 127 + (e - 1 if m == 0.5 else e)))


def test_reference_scale_codes_at_the_edges():
    edges = [0.0, 1e-10, 2.0**-20, 0.5, 1.0, 447.0, 448.0, 448.0001, 896.0, 3.0e38]
    got = mxfp8_codes(torch.tensor(edges)).tolist()
    assert got == [_code_frexp(a) for a in edges]
    assert got[6] == 127  # amax 448 is scale 1


def test_reference_quantization_error_is_bounded_by_the_block_max():
    x = rows(token_sizes(4, 3), HIDDEN, 3)
    q, s = mxfp8(x)
    xf = x.float().view(4, -1, 32)
    err = (dequant(q, s).view(4, -1, 32) - xf).abs()
    assert float((err / xf.abs().amax(-1, keepdim=True)).max()) <= 2.0**-4
    assert int(code_distance(q, q).max()) == 0


def test_reference_norm_and_lookahead():
    """The norm is Gemma (1 + w) over the unrounded v, the lookahead the same
    over bf16(v) with the reading layer's weight; with w2 = w and equal eps they
    differ only by that rounding."""
    s = 3
    parts = [rows(token_sizes(s, 1), HIDDEN, 100 + r) for r in range(8)]
    res = rows(token_sizes(s, 1), HIDDEN, 7)
    w = (torch.randn(HIDDEN, generator=torch.Generator().manual_seed(1)) * 0.3).to(
        torch.bfloat16
    )
    out = layer_reference(parts, res, w, w, 1e-6, 1e-6)
    v = sum(p.float() for p in parts).to(torch.bfloat16).float() + res.float()
    assert torch.equal(out["h_mid"], v.to(torch.bfloat16))
    rstd = torch.rsqrt((v.double() ** 2).mean(-1, keepdim=True)).float()
    want = (v * rstd * (w.float() + 1)).to(torch.bfloat16)
    assert torch.allclose(out["x"].float(), want.float(), rtol=2**-7, atol=0)
    assert int(code_distance(out["x8s"], out["la8s"]).max()) <= 1
    # where rounding v moved no scale, it moves an element by at most one code
    same = (out["x8s"] == out["la8s"]).repeat_interleave(32, dim=-1)
    assert int((code_distance(out["x8"], out["la8"]) * same).max()) <= 1
    assert "la8" not in layer_reference(parts, res, w, None, 1e-6, 1e-6)


# -------------------------------------------- the GPU test's inputs and checks

_K = 256  # the GPU test's checks do not depend on K; a short one keeps this fast


@functools.cache
def _weights():
    def one(rank):
        g = torch.Generator().manual_seed(7000 + rank)
        w = torch.randn(HIDDEN, _K, generator=g) * 0.02
        w[torch.randint(0, HIDDEN, (64,), generator=g)] *= 64.0
        return mxfp8(w.to(torch.bfloat16))

    return [one(r) for r in range(8)]


@functools.cache
def _case(s):
    g = torch.Generator().manual_seed(17)
    wp = (torch.randn(HIDDEN, generator=g) * 0.3).to(torch.bfloat16)
    wa = (torch.randn(HIDDEN, generator=g) * 0.3).to(torch.bfloat16)
    return (*case_inputs(s, _weights(), 1000 * s), wp, wa)


def test_case_inputs_keep_every_rank_and_the_residual_in_play():
    """Every rank's partial is a material share of its token's all-reduced row
    (sizes drawn a rank apart over 6 decades left some at 1e-6), and a token's
    residual row is RES_RATIOS of that row's norm."""
    _, parts, res = _case(5)[:3]
    ar = ar_sum(parts).float().norm(dim=-1)
    share = torch.stack([p.float().norm(dim=-1) for p in parts]) / ar
    assert float(share.min()) >= 0.03
    ratio = res.float().norm(dim=-1) / ar
    want = torch.tensor([RES_RATIOS[t % len(RES_RATIOS)] for t in range(5)])
    assert torch.allclose(ratio, want, rtol=2**-6)


def _mutations(s):
    """Bugs the GPU test must catch, as the outputs a kernel with them writes."""
    cores, parts, res, wp, wa = _case(s)

    def ref(p=parts, r=res, w=wp, w2=wa):
        return layer_reference(p, r, w, w2, 1e-6, 1e-6)

    want = ref()
    out = {}
    for r in range(8):
        out[f"rank {r}'s partial lost"] = ref(
            [torch.zeros_like(p) if q == r else p for q, p in enumerate(parts)]
        )
    for t in range(s):
        p = [x.clone() for x in parts]
        p[3][t] = 0
        out[f"rank 3's token {t} lost"] = ref(p)
    c = cores[3].clone()
    c[:, 64:96] = 0
    out["a K block of rank 3's core misread"] = ref(
        parts[:3] + [o_partial(c, *_weights()[3])] + parts[4:]
    )
    ws = _weights()[3][1].clone()
    ws[:, 2] += 1
    out["a scale column of rank 3's weight misread"] = ref(
        parts[:3] + [o_partial(cores[3], _weights()[3][0], ws)] + parts[4:]
    )
    out["no residual"] = ref(r=torch.zeros_like(res))
    out["residual columns shifted by 2"] = ref(r=res.roll(2, -1))
    if s > 1:
        out["residual of the next token"] = ref(r=res.roll(1, 0))
    out["post norm without + 1"] = ref(w=(wp.float() - 1).to(torch.bfloat16))
    out["lookahead with the post weight"] = ref(w2=wp)
    m = dict(want, x8s=want["x8s"].clone())
    m["x8s"][0, 0] += 1
    out["an x8 scale code 1 off"] = m
    m = dict(want, la8s=want["la8s"].clone())
    m["la8s"][-1, -1] -= 1
    out["an la8 scale code 1 off"] = m
    m = dict(want, h_mid=ar_sum(parts))
    out["h_mid without the residual"] = m
    return want, wa, out


@pytest.mark.parametrize("s", [1, 3, 16])
def test_the_gpu_tests_checks_catch_injected_bugs(s):
    want, wa, out = _mutations(s)
    assert layer_failures(layer_report(want, want, wa, 1e-6)) == []
    missed = [
        k
        for k, got in out.items()
        if not layer_failures(layer_report(got, want, wa, 1e-6))
    ]
    assert missed == []


def test_the_gpu_tests_bounds_pass_every_partial_one_ulp_off():
    """Design section 12's tolerance: a partial may differ in its last bf16 bit."""
    _, parts, res, wp, wa = _case(3)
    want = layer_reference(parts, res, wp, wa, 1e-6, 1e-6)
    g = torch.Generator().manual_seed(5)
    off = []
    for p in parts:
        step = torch.where(torch.rand(p.shape, generator=g) < 0.5, 1, -1)
        off.append((p.view(torch.int16) + step.to(torch.int16)).view(torch.bfloat16))
    rep = layer_report(layer_reference(off, res, wp, wa, 1e-6, 1e-6), want, wa, 1e-6)
    assert layer_failures(rep) == [], rep
