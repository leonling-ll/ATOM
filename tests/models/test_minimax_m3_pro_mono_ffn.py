# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pro mono phase 1's ``upto = "ffn"`` build, its plain-Python half: the FFN
stages' widths, regions and contract, and the torch reference the 8-GPU test
(``test_minimax_m3_pro_mono_ffn_gpu``) holds the kernel to -- its MXFP4 rule,
its router, and that the GPU test's bounds pass summation-order noise and
catch every injected bug (on a narrow case: hidden 1024, intermediate 96 of
128, 16 experts).

At Pro's widths (hidden 7168, 640 of 768, 24 experts, the same generator and
seeds; CPU, about 15 s) the noise model below gives at most 0.016 per row at
2^-12 and 0.021 at 2^-9, and the weakest mutation 0.066 (an x8 scale code
+1), against ``FFN_BOUND`` 0.031."""

import itertools

import pytest
import torch
from minimax_m3_pro_mono_ref import (
    FFN_BOUND,
    ROUTE_SCALE,
    TOPK,
    case_ffn,
    dq_mxfp4,
    ffn_case,
    ffn_failures,
    ffn_mutations,
    mxfp4,
    qdq_mxfp4,
    route_failures,
    route_reference_of_logits,
    row_relnorm,
    swiglu_oai,
)

from atom.models.minimax_m3.pro_mono import layout as L
from atom.models.minimax_m3.pro_mono.config import (
    EXPERTS,
    HIDDEN,
    MAX_ROWS,
    MAX_ROWS_LAYOUT,
)
from atom.mono.plan.execution import BLOCKS, LDS_BYTES, THREADS, WAVES
from atom.mono.plan.trace import Space

NARROW = {"h": 1024, "i_valid": 96, "i_pad": 128, "e": 16, "i_sh": 96}


# ------------------------------------------------------------ the layout


def test_ffn_geometry():
    assert (L.E_RANK, L.R_GROUPS, L.R_PARTS, L.R_KLEN, L.R_TASKS) == (
        48,
        3,
        7,
        1024,
        21,
    )
    assert (L.INTER_V, L.INTER_R, L.UG_GROUPS, L.W13_HALF, L.W2_STEPS) == (
        640, 768, 20, 28, 5,
    )  # fmt: skip
    assert (L.INTER_S, L.SH_TASKS, L.SH_CPW, L.SD_STEPS, L.DOWN_TASKS) == (
        640, 80, 7, 5, 448,
    )  # fmt: skip
    # the pad is whole MXFP4 blocks: skipping it skips whole zero blocks
    assert L.INTER_V % 32 == 0 and L.INTER_R % 32 == 0
    # a ug task: 4 row groups x 2 K halves, one a wave
    assert 4 * 2 == WAVES and 2 * L.W13_HALF * 128 == HIDDEN
    # the shared gate_up's K split over the waves, the down's over 5 of them
    assert L.SH_CPW * WAVES * 128 == HIDDEN and L.SD_STEPS <= WAVES
    # thread maps of the epilogues fit a CTA at the widest build
    assert L.UG_COLS * MAX_ROWS <= THREADS
    assert (L.DOWN_ROWS // 2) * MAX_ROWS <= THREADS
    assert L.R_ROWS * MAX_ROWS <= THREADS
    assert MAX_ROWS * (L.INTER_S // 32) <= THREADS  # load_smid: a block a thread
    # the route's 6 logits a lane
    assert EXPERTS % 64 == 0


def test_ffn_placement_keeps_the_roles_apart():
    """Router, route and shared gate_up on disjoint CTAs, none on a norm CTA
    (whose x every one of them waits for)."""
    router = set(range(L.R0, L.R0 + L.R_TASKS))
    route = set(range(L.RT0, L.RT0 + MAX_ROWS))
    shared = set(range(L.SH0, L.SH0 + L.SH_TASKS))
    norm = set(range(L.NT))
    for a, b in itertools.combinations((router, route, shared, norm), 2):
        assert not a & b
    assert max(router) == BLOCKS - 1
    # the router CTAs have no o task: their weights load from the start
    assert min(router) >= L.O_TASKS


def _spans(lay):
    return sorted(lay.values())


@pytest.mark.parametrize("debug", [False, True])
def test_ffn_regions_are_disjoint_and_sized(debug):
    key = L.P1Build(tokens=MAX_ROWS, upto="ffn", debug=debug)
    for lay in (L.scratch_layout(key), L.plain_layout(key)):
        spans = _spans(lay)
        for (o1, n1), (o2, _) in itertools.pairwise(spans):
            assert o1 + n1 <= o2
        assert all(o % 16 == 0 for o, _ in spans)
    s = L.scratch_layout(key)
    t = MAX_ROWS_LAYOUT - 1
    assert (L.rpart_pair(L.R_TASKS - 1, t, L.R_ROWS - 1) + 1) * 8 == s["rpart"][1]
    assert (L.route_pair(t, 2 * L.TOPK - 1) + 1) * 8 == s["route"][1]
    assert (L.smid_pair(t, L.INTER_S - 2) + 1) * 8 == s["smid"][1]
    assert s["ugf"][1] == L.max_u(MAX_ROWS_LAYOUT) * L.UG_GROUPS * 8
    assert s["xrdy"][1] == L.NT * 8
    p = L.plain_layout(key)
    assert p["x4"][1] == MAX_ROWS_LAYOUT * HIDDEN // 2
    assert p["inter"][1] == MAX_ROWS_LAYOUT * L.TOPK * L.INTER_V // 2
    peer = L.peer_layout()
    assert (L.rlog_pair(L.TP - 1, t, L.E_RANK - 1) + 1) * 8 == peer["rlog"][1]
    # a norm build has none of them
    norm = L.P1Build(tokens=MAX_ROWS)
    assert L.plain_layout(norm) == {} and "route" not in L.scratch_layout(norm)


def test_ffn_build_key_and_contract_table():
    key = L.P1Build(tokens=1, upto="ffn")
    assert key.upto == "ffn"
    with pytest.raises(ValueError):
        L.P1Build(tokens=1, upto="moe")
    got = {d.name: (d.space, d.writers, d.exchange) for d in L.decls(key)}
    assert set(got) == set(L.REGIONS)
    assert got["rlog"] == (Space.PEER, ("router",), False)
    assert got["rpart"] == (Space.SCRATCH, ("router",), True)
    assert got["ugf"][:2] == (Space.SCRATCH, ("ug",))
    assert [d.name for d in L.decls(L.P1Build(tokens=1))] == list(L.NORM_REGIONS)


def test_moe_weights_table_names_the_binders_fields():
    import dataclasses

    from atom.models.minimax_m3.pro_mono.weights import ProMoeLayer

    fields = {f.name for f in dataclasses.fields(ProMoeLayer)}
    assert set(L.MOE_WEIGHTS) <= fields
    assert {"moe_w", "ffn_out", "plain"} <= set(L.ABI_NAMES)


def test_ffn_lds_fits_at_the_widest_build():
    """The FFN build's LDS (``layer.build``'s structs, counted here) under the
    CU's: the contract run checks the binary's."""
    s = MAX_ROWS
    words = (
        s * L.O_ROW + s * L.OSC_ROW + WAVES * 64 * 4 + L.O_ROWS * s + 2 * 11
        + s * L.X4_ROW + s * L.XSC_WORDS + s * L.SMSC_WORDS
        + s * L.TOPK * L.DOWN_ROWS + s * 2 * L.TOPK + 2 * EXPERTS
        + 2 * L.max_u(s) + 1 + s * L.TOPK + 3 * (128 + 16) + WAVES
    )  # fmt: skip
    assert 4 * words <= LDS_BYTES


# ------------------------------------------------------------ the reference


def test_mxfp4_rule():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(64, 256, generator=g) * torch.exp2(
        torch.randint(-6, 6, (64, 8), generator=g).float()
    ).repeat_interleave(32, 1)
    x = x.to(torch.bfloat16).float()
    nib, code = mxfp4(x)
    scale = (code << 23).view(torch.float32).repeat_interleave(32, 1)
    amax = x.view(64, 8, 32).abs().amax(-1).repeat_interleave(32, 1)
    # RoundUp: the block's max fits 6 x scale, half the scale would not
    assert (amax <= 6 * scale).all() and (amax > 3 * scale).all()
    # RNE onto the e2m1 grid: within half a grid step (at most 1 x scale)
    err = (dq_mxfp4(nib, code) - x).abs()
    assert (err <= 0.5 * torch.where(x.abs() >= 4 * scale, 2.0, 1.0) * scale).all()
    # a block of zeros is zeros; amax = 6 x 2^k gives scale 2^k exactly
    z = torch.zeros(1, 32)
    assert not qdq_mxfp4(z).any()
    for k in range(-4, 5):
        b = torch.zeros(1, 32)
        b[0, 7] = 6.0 * 2.0**k
        assert int(mxfp4(b)[1][0, 0]) == 127 + k


def test_swiglu_oai():
    g = torch.tensor([-10.0, 0.0, 2.0, 9.0])
    u = torch.tensor([-9.0, 1.0, 3.0, 8.0])
    gc = torch.tensor([-10.0, 0.0, 2.0, 7.0])
    uc = torch.tensor([-7.0, 1.0, 3.0, 7.0])
    want = gc * torch.sigmoid(1.702 * gc) * (uc + 1)
    assert torch.equal(swiglu_oai(g, u), want)


def test_route_reference():
    logits = torch.tensor([[0.5, 0.5, 0.25, -1.0, 2.0, 0.5]]).to(torch.bfloat16)
    bias = torch.tensor([0.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    _, picks, w, _ = route_reference_of_logits(logits, bias)
    # the bias pulls expert 3 in; equal scores: the lower expert first
    assert picks[0].tolist() == [3, 4, 0, 1]
    sig = torch.sigmoid(logits.float())[0, picks[0]]
    assert torch.allclose(w[0], sig / sig.sum() * ROUTE_SCALE)
    assert abs(float(w.sum()) - ROUTE_SCALE) < 1e-6


def test_route_checks_catch_a_wrong_router():
    c = ffn_case(5, **NARROW, seed=3)
    logits, picks, w = c["logits"], c["picks"], c["weights"]
    assert not route_failures(logits, picks, w, logits, c["bias"])
    # logits one bf16 step off are sum order (the picks and weights then
    # follow from them); two steps are not
    for steps, want in ((1, []), (2, ["logits"])):
        off = (logits.view(torch.int16) + steps).view(torch.bfloat16)
        _, p, ww, _ = route_reference_of_logits(off, c["bias"])
        assert route_failures(off, p, ww, logits, c["bias"]) == want
    bad = {
        "unscaled": (picks, w / ROUTE_SCALE),
        "biased weights": (picks, w * 1.01),
        "5th pick": (
            torch.sort(
                torch.sigmoid(logits.float()) + c["bias"], -1, descending=True
            ).indices[:, 1 : TOPK + 1],
            w,
        ),
        "no bias": (torch.topk(logits.float(), TOPK).indices, w),
    }
    for name, (p, ww) in bad.items():
        assert route_failures(logits, p, ww, logits, c["bias"]), name


@pytest.mark.parametrize("s", [1, 3, 16])
def test_the_gpu_tests_ffn_bound_passes_sum_order_noise(s):
    """gate / up off by 2^-12 relative, at random (f32 sum order is ~2^-20):
    the bound holds. Off by 2^-20 nothing moves at all."""
    c = ffn_case(s, **NARROW, seed=11 + s)
    want = case_ffn(c)
    g = torch.Generator().manual_seed(s)

    def noisy(rel):
        def f(gg, uu):
            def jitter(t):
                sign = torch.randint(0, 2, t.shape, generator=g) * 2 - 1
                return t * (1 + rel * sign)

            return swiglu_oai(jitter(gg), jitter(uu))

        return f

    assert torch.equal(case_ffn(c, swiglu=noisy(2.0**-20)), want)
    assert not ffn_failures(case_ffn(c, swiglu=noisy(2.0**-12)), want)


@pytest.mark.parametrize("s", [1, 3, 16])
def test_the_gpu_tests_ffn_bound_catches_injected_bugs(s):
    c = ffn_case(s, **NARROW, seed=11 + s)
    want = case_ffn(c)
    muts = ffn_mutations(
        c["x"], c["x8"], c["x8s"], c["picks"], c["weights"], c["routed"],
        c["shared"], NARROW["e"],
    )  # fmt: skip
    assert len(muts) >= 13
    missed = {
        k: row_relnorm(v, want) for k, v in muts.items() if not ffn_failures(v, want)
    }
    assert not missed, missed
    # and each by a margin
    assert min(row_relnorm(v, want) for v in muts.values()) > 2 * FFN_BOUND


def test_neither_the_routed_nor_the_shared_output_drowns():
    """The generator's code ranges keep the two outputs of one size, so a
    lost or wrong one shows (``W2_CODES``, ``SHARED_CODES``)."""
    c = ffn_case(3, **NARROW, seed=2)
    w_sgu, w_sd = c["shared"]
    routed = case_ffn(dict(c, shared=(w_sgu * 0, w_sd * 0))).float().norm(dim=-1)
    shared = case_ffn(dict(c, weights=c["weights"] * 0)).float().norm(dim=-1)
    ratio = routed / shared
    assert ((ratio > 0.25) & (ratio < 4)).all(), ratio


def test_the_routed_pad_is_exactly_zero():
    """Rows past the valid intermediate are zero, so their mids are zero whole
    MXFP4 blocks: the kernel's skipping them changes nothing."""
    c = ffn_case(2, **NARROW, seed=5)
    w13, w2 = c["routed"](0)
    i = NARROW["i_pad"]
    pad = torch.arange(i) >= NARROW["i_valid"]
    assert not w13[:i][pad].any() and not w13[i:][pad].any()
    assert not w2[:, pad].any()
    xq = qdq_mxfp4(c["x"])
    gu = w13 @ xq[0]
    assert not swiglu_oai(gu[:i], gu[i:])[pad].any()


# ------------------------------------------------- operand addressing (aiter)


def _sample_lanes():
    return (0, 5, 16, 31, 47, 63)


def test_fp4_tile_addressing_matches_aiters_shuffle():
    """``fp4_tile_dword`` (the kernel's w13 / w2 loads) against the loader's
    (16, 16) shuffle of an expert's plane: lane l holds row 16 rg + l % 16,
    k 128 st + 32 (l / 16) .. + 32."""
    pytest.importorskip("aiter")
    from aiter.ops.shuffle import shuffle_weight

    g = torch.Generator().manual_seed(0)
    for rows, k in ((2 * L.INTER_R, HIDDEN), (HIDDEN, L.INTER_R)):
        b = torch.randint(0, 256, (2, rows, k // 2), generator=g, dtype=torch.uint8)
        sw = shuffle_weight(b, layout=(16, 16)).contiguous().view(-1)
        for rg, st, lane in itertools.product(
            (0, 1, rows // 16 - 1), (0, 3), _sample_lanes()
        ):
            dw = rows * k // 8 + L.fp4_tile_dword(rg, k, st, lane)  # expert 1
            want = b[1, 16 * rg + lane % 16]
            c0 = (128 * st + 32 * (lane // 16)) // 2
            assert torch.equal(sw[4 * dw : 4 * dw + 16], want[c0 : c0 + 16])


def test_routed_scale_addressing_matches_aiters_shuffle():
    """``w13_scale_word`` / ``w2_scale_word`` and the byte the kernel takes,
    against ``shuffle_scale`` of the whole plane (every expert's rows)."""
    pytest.importorskip("aiter")
    from aiter.ops.shuffle import shuffle_scale

    g = torch.Generator().manual_seed(1)
    rows13, cols13 = 2 * L.INTER_R, HIDDEN // 32
    c13 = torch.randint(0, 256, (3 * rows13, cols13), generator=g, dtype=torch.uint8)
    s13 = shuffle_scale(c13).contiguous().view(-1)
    for e, rg, st, lane in itertools.product(
        (0, 2), (0, 1, 2 * 40 + 1, rows13 // 16 - 1), (0, 1, 54, 55), _sample_lanes()
    ):
        byte = 4 * L.w13_scale_word(e, rg, st, lane) + (st % 2) * 2 + rg % 2
        assert s13[byte] == c13[e * rows13 + 16 * rg + lane % 16, 4 * st + lane // 16]
    cols2 = L.INTER_R // 32
    c2 = torch.randint(0, 256, (2 * HIDDEN, cols2), generator=g, dtype=torch.uint8)
    s2 = shuffle_scale(c2).contiguous().view(-1)
    for e, rgd, st, lane in itertools.product(
        (0, 1), (0, 1, 7, L.DOWN_TASKS - 1), range(L.W2_STEPS), _sample_lanes()
    ):
        word = L.w2_scale_word(e, rgd // 2, st // 2, lane)
        byte = 4 * word + (st % 2) * 2 + rgd % 2
        assert s2[byte] == c2[e * HIDDEN + 16 * rgd + lane % 16, 4 * st + lane // 16]


def test_mxfp8_tile_addressing_matches_aiters_shuffle():
    """``mxfp8_tile_dword`` (the shared expert's loads) against the native
    MXFP8 loader's (16, 16) shuffle: lane group g's two 16 B pieces are k 16
    g .. + 16 and 64 + 16 g .. + 16 of its row's 128-k chunk."""
    pytest.importorskip("aiter")
    from aiter.ops.shuffle import shuffle_weight

    g = torch.Generator().manual_seed(2)
    for rows, cols in ((2 * L.INTER_S, HIDDEN), (HIDDEN, L.INTER_S)):
        f = torch.randint(0, 256, (rows, cols), generator=g, dtype=torch.uint8)
        sw = shuffle_weight(f, layout=(16, 16)).contiguous().view(-1)
        for row, kc, grp in itertools.product(
            (0, 9, 17, rows - 1), (0, cols // 128 - 1), range(4)
        ):
            dw = L.mxfp8_tile_dword(row, cols, kc, grp)
            for h in range(2):
                k0 = 128 * kc + 64 * h + 16 * grp
                got = sw[4 * (dw + 256 * h) : 4 * (dw + 256 * h) + 16]
                assert torch.equal(got, f[row, k0 : k0 + 16]), (row, kc, grp, h)
