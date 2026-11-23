# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pro mono phase 1's layer kernel, its plain-Python half: the peer regions
and their indexing, the column ownership the o_proj push and the all-reduce
must agree on, the build key, and the torch reference the GPU test holds the
kernel to (checked here against independent formulas)."""

import itertools
import math

import pytest
import torch
from minimax_m3_pro_mono_ref import (
    code_distance,
    dequant,
    layer_reference,
    mxfp8,
    mxfp8_codes,
    rows,
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
    for bad in ({"tokens": 0}, {"tokens": MAX_ROWS + 1}, {"tokens": 1, "upto": "ffn"}):
        with pytest.raises(ValueError):
            L.P1Build(**bad)
    names = [d.name for d in L.decls(L.P1Build(tokens=1))]
    assert names == list(L.REGIONS)


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
    x = rows(4, HIDDEN, 3)
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
    parts = [rows(s, HIDDEN, 100 + r) for r in range(8)]
    res = rows(s, HIDDEN, 7)
    w = (torch.randn(HIDDEN, generator=torch.Generator().manual_seed(1)) * 0.3).to(
        torch.bfloat16
    )
    out = layer_reference(parts, res, w, w, 1e-6, 1e-6)
    v = sum(p.float() for p in parts).to(torch.bfloat16).float() + res.float()
    assert torch.equal(out["h_mid"], v.to(torch.bfloat16))
    rstd = torch.rsqrt((v.double() ** 2).mean(-1, keepdim=True)).float()
    want = (v * rstd * (w.float() + 1)).to(torch.bfloat16)
    assert torch.allclose(out["x"].float(), want.float(), rtol=2**-7, atol=0)
    assert int(code_distance(out["x8"], out["la8"]).max()) <= 1
    assert int(code_distance(out["x8s"], out["la8s"]).max()) <= 1
    assert "la8" not in layer_reference(parts, res, w, None, 1e-6, 1e-6)
