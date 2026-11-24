# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pro mono phase 1's ``upto = "ffn"`` build, its plain-Python half: the FFN
stages' widths, regions and contract, and the torch reference the 8-GPU test
(``test_minimax_m3_pro_mono_ffn_gpu``) holds the kernel to -- its MXFP4 rule,
its router, and that the GPU test's bounds pass summation-order noise and
catch every injected bug (on a narrow case: hidden 1024, intermediate 96 of
128, 16 experts), for every routed format (A16W4, the default, W4A4 and
W4A8); that the A16W4 reference is aiter's a16w4 math up to rounding order;
and the A16W4 operand maps (a K half of x in LDS, the bf16 mids, the K-half
tasks on their CTAs).

At Pro's widths (hidden 7168, 640 of 768, 24 experts, the same generator and
seeds; CPU, about 15 s) the noise model below gives at most 0.016 per row at
2^-12 and 0.021 at 2^-9, and the weakest mutation 0.066 (an x8 scale code
+1), against ``FFN_BOUND`` 0.031 (W4A8); A16W4 at most 0.017 at 2^-12, 0.019
at 2^-9, the weakest mutation 0.068 (an x8 scale code + 1), and its reference
0.003 from the a16w4 emulation where W4A8's is 0.065 .. 0.070."""

import itertools

import pytest
import torch
from minimax_m3_pro_mono_ref import (
    E2M1,
    FFN_BOUND,
    ROUTE_SCALE,
    ROUTED_ACTS,
    TOPK,
    case_ffn,
    dequant,
    dq_mxfp4,
    ffn_case,
    ffn_failures,
    ffn_mutations,
    logit_bound,
    mxfp4,
    mxfp8,
    near_tie_router,
    qdq_mxfp4,
    qdq_mxfp8,
    route_failures,
    route_reference,
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
    RoutedPlanes,
)
from atom.mono.plan.execution import BLOCKS, LDS_BYTES, THREADS, WAVES
from atom.mono.plan.trace import Space

NARROW = {"h": 1024, "i_valid": 96, "i_pad": 128, "e": 16, "i_sh": 96}
# the routed planes' width an FFN build bakes: Pro's at pad_align 128
# (MiniMaxM3MoE), no pad
INTER_R = 640
# the loaded widths the addressing is checked at: Pro's, and the default
# pad_align 256's (the up block and w2's K stride move, the code doesn't)
LOADED_WIDTHS = (640, 768)


# ------------------------------------------------------------ the layout


def test_ffn_geometry():
    assert (L.E_RANK, L.R_GROUPS, L.R_PARTS, L.R_KLEN, L.R_TASKS) == (
        48,
        3,
        7,
        1024,
        21,
    )
    assert (L.INTER_V, L.UG_GROUPS, L.W13_HALF, L.W2_STEPS) == (640, 20, 28, 5)
    assert (L.INTER_S, L.SH_TASKS, L.SH_CPW, L.SD_STEPS, L.DOWN_TASKS) == (
        640, 80, 7, 5, 448,
    )  # fmt: skip
    # any pad is whole MXFP4 blocks: skipping it skips whole zero blocks
    assert L.INTER_V % 32 == 0
    assert all(w % 32 == 0 for w in LOADED_WIDTHS)
    # a ug task: 4 row groups x 2 K halves, one a wave (A16W4: a task a K
    # half, its waves 4 row groups x 2 K quarters)
    assert 4 * 2 == WAVES and 2 * L.W13_HALF * 128 == HIDDEN
    assert 2 * L.W13_Q == L.W13_HALF and L.W13_Q * 128 * 4 == HIDDEN
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


@pytest.mark.parametrize("act", ROUTED_ACTS)
@pytest.mark.parametrize("debug", [False, True])
def test_ffn_regions_are_disjoint_and_sized(debug, act):
    key = L.P1Build(
        tokens=MAX_ROWS, upto="ffn", inter_r=INTER_R, debug=debug, routed_act=act
    )
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
    # A16W4's K-half partials: two pairs a (pick row, mid column), the last
    # pairs (only the queue words past them), so every other pair region sits
    # where the other formats' do
    pairs = {k: v for k, v in s.items() if k != "queue"}
    if act == "bf16":
        assert (L.ugp_pair(MAX_ROWS_LAYOUT * L.TOPK - 1, L.UG_GROUPS - 1, 31) + 2) * (
            8
        ) == s["ugp"][1]
        assert max(o for o, _ in pairs.values()) == s["ugp"][0]
    else:
        assert "ugp" not in s
    other = L.scratch_layout(
        L.P1Build(
            tokens=MAX_ROWS, upto="ffn", inter_r=INTER_R, debug=debug, routed_act="fp4"
        )
    )
    assert {k: v for k, v in pairs.items() if k != "ugp"} == {
        k: v for k, v in other.items() if k != "queue"
    }
    p = L.plain_layout(key)
    if act == "fp4":
        assert p["x4"][1] == MAX_ROWS_LAYOUT * HIDDEN // 2
        assert p["inter"][1] == MAX_ROWS_LAYOUT * L.TOPK * L.INTER_V // 2
    else:
        # W4A8 reads the norm's x8, A16W4 the bf16 x: no MXFP4 copy of x
        assert "x4" not in p and "x4s" not in p
    if act == "fp8":
        # a mid is a byte a value
        assert p["inter"][1] == MAX_ROWS_LAYOUT * L.TOPK * L.INTER_V
    if act == "bf16":
        # a bf16 mid: two bytes a value, no codes
        assert p["inter"][1] == MAX_ROWS_LAYOUT * L.TOPK * L.INTER_V * 2
        assert "isc" not in p
    peer = L.peer_layout()
    assert (L.rlog_pair(L.TP - 1, t, L.E_RANK - 1) + 1) * 8 == peer["rlog"][1]
    # a norm build has none of them
    norm = L.P1Build(tokens=MAX_ROWS)
    assert L.plain_layout(norm) == {} and "route" not in L.scratch_layout(norm)


@pytest.mark.parametrize("act", ROUTED_ACTS)
def test_the_queue_words_are_their_own_and_zeroed_with_the_step(act):
    """A launch tag's queue words (one a task list: A16W4 two, the kernel's
    list h at the first + 4 h) lie in ``scratch`` (the step's zeroing clears
    them) past every tagged pair, apart for every layer the runner launches."""
    key = L.P1Build(tokens=1, upto="ffn", inter_r=INTER_R, routed_act=act)
    lay = L.scratch_layout(key)
    q0, qn = lay["queue"]
    assert all(o + n <= q0 for name, (o, n) in lay.items() if name != "queue")
    lists = L.ug_split(act)
    words = [
        L.queue_word_offset(key, layer) + 4 * h
        for layer in range(L.LAYERS)
        for h in range(lists)
    ]
    assert len(set(words)) == L.LAYERS * lists
    assert min(words) >= q0 and max(words) + 4 <= q0 + qn <= L.scratch_bytes(key)
    with pytest.raises(ValueError):
        L.queue_word_offset(key, L.LAYERS)
    assert L.MOE_TABLE == L.MOE_WEIGHTS + ("queue",)
    assert "queue" not in L.scratch_layout(L.P1Build(tokens=1))


def test_only_a_one_row_build_places_its_routed_tasks():
    """A 1-row build takes no queue claim (they cost its layer 2.5 us on GPU,
    v1tuned_1123T1524): its union's ug tasks fit the first round, so placing
    them balances nothing less, and the placed down tasks (CTA d mod BLOCKS)
    cover all of them. Every wider build takes its tasks from the queue."""
    assert not L.queued(1)
    assert all(L.queued(s) for s in range(2, MAX_ROWS + 1))
    assert L.max_u(1) * L.UG_GROUPS * L.ug_split("bf16") <= BLOCKS
    assert sorted(
        b + BLOCKS * i
        for b in range(BLOCKS)
        for i in range(-(-L.DOWN_TASKS // BLOCKS))
        if b + BLOCKS * i < L.DOWN_TASKS
    ) == list(range(L.DOWN_TASKS))


@pytest.mark.parametrize("act", ROUTED_ACTS)
def test_the_task_lists_place_one_entry_a_cta(act):
    """Each task list's first round is placed one entry a CTA (A16W4: list h,
    every CTA of it a half-h CTA, the half it stages); the lists' down tasks
    are every down task once."""
    lists, per = L.ug_split(act), L.list_ctas(act)
    assert lists * per == BLOCKS
    placed = [L.task_list(bid, act) for bid in range(BLOCKS)]
    assert sorted(placed) == [(h, k) for h in range(lists) for k in range(per)]
    for bid, (h, k) in enumerate(placed):
        # A16W4: the CTA's placed task 2 k + h (``first_ug``) is pair k's half h
        assert L.first_ug(bid) == lists * k + h
    for ug in (0, 7, 80):
        downs = [
            L.list_down(ug + j, ug, h, act)
            for h in range(lists)
            for j in range(-(-L.DOWN_TASKS // lists))
        ]
        assert sorted(d for d in downs if d < L.DOWN_TASKS) == list(range(L.DOWN_TASKS))


def _run_lists(act, nu, queued, seed):
    """A model of ``ffn.run_ug`` / ``ffn.run_down``'s schedule. A CTA's loop
    iteration is two steps: first (never blocking) it claims the list entry
    after next and publishes its previous ug task's flag; then it waits where
    the kernel waits -- an A16W4 half-0 task for its pair's half-1 partials
    -- and finishes the task. Before its first down task a CTA waits for
    every pair's flag. CTAs step in a random order. Returns the tasks run in
    order; raises on a deadlock (no CTA can step)."""
    import random

    rnd = random.Random(seed)
    lists, per = L.ug_split(act), L.list_ctas(act)
    ug = nu * L.UG_GROUPS
    counts = [0] * lists
    partial, flag, ran = set(), set(), []

    def claim(h):
        counts[h] += 1
        return per + counts[h] - 1

    ctas = []
    for bid in range(BLOCKS):
        h, k0 = L.task_list(bid, act)
        held = [k0, claim(h)] if queued else [k0, k0 + per]
        ctas.append({"h": h, "held": held, "prev": None, "mid": False, "down": 0})

    def step(c):
        """One step of CTA c, or False where it blocks."""
        h, k = c["h"], c["held"][0]
        if k >= ug and c["down"] == 0:
            # run_down's wait_ug, the ug loop's last flag out before it
            if len(flag) < ug:
                return False
            c["down"] = 1
            return True
        if not c["mid"]:
            c["held"].append(claim(h) if queued else c["held"][-1] + per)
            if c["prev"] is not None:
                flag.add(c["prev"])
                c["prev"] = None
            c["mid"] = True
            return True
        if k < ug and lists == 2 and h == 0 and k not in partial:
            return False
        c["mid"] = False
        c["held"].pop(0)
        if k >= ug:
            ran.append(("down", L.list_down(k, ug, h, act)))
            return True
        ran.append(("ug", h, k))
        if lists == 2 and h == 1:
            partial.add(k)
        else:
            c["prev"] = k
        if c["held"][0] >= ug and c["prev"] is not None:
            # the loop's end publishes the last flag
            flag.add(c["prev"])
            c["prev"] = None
        return True

    def done(c):
        k = c["held"][0]
        if k < ug or c["mid"]:
            return False
        # a placed build's down tasks are CTA d mod BLOCKS's (not modelled)
        return not queued or L.list_down(k, ug, c["h"], act) >= L.DOWN_TASKS

    while True:
        live = [c for c in ctas if not done(c)]
        if not live:
            break
        rnd.shuffle(live)
        # a random subset steps (every one of them: no short cut), else all
        stepped = [step(c) for c in live[: rnd.randint(1, len(live))]]
        if not any(stepped) and not any([step(c) for c in live]):  # noqa: C419
            raise RuntimeError(f"deadlock: {len(ran)} tasks run")
    return ran


@pytest.mark.parametrize("act", ROUTED_ACTS)
@pytest.mark.parametrize("nu", [1, 4, 13, 48])
def test_the_queued_lists_run_every_task_once_and_never_deadlock(act, nu):
    """With claims a task ahead and CTAs at any relative speed, every ug task
    (A16W4: every pair's two halves) and every down task runs once, and no
    wait blocks for good: a half-1 task waits for nothing and runs on a
    half-1 CTA, so the half-0 waits end (``layout.task_lists``)."""
    lists = L.ug_split(act)
    for seed in range(3):
        ran = _run_lists(act, nu, True, seed)
        ugs = [r for r in ran if r[0] == "ug"]
        assert sorted(ugs) == sorted(
            ("ug", h, k) for h in range(lists) for k in range(nu * L.UG_GROUPS)
        )
        assert sorted(r[1] for r in ran if r[0] == "down") == list(range(L.DOWN_TASKS))


@pytest.mark.parametrize("act", ROUTED_ACTS)
def test_a_placed_one_row_build_runs_every_ug_task_once(act):
    """The 1-row build's placed lists (no claim): every ug task once, in one
    round, without a deadlock (its down tasks are CTA d mod BLOCKS's)."""
    nu = L.max_u(1)
    lists = L.ug_split(act)
    assert nu * L.UG_GROUPS <= L.list_ctas(act)
    for seed in range(3):
        ran = _run_lists(act, nu, False, seed)
        assert sorted(r for r in ran if r[0] == "ug") == sorted(
            ("ug", h, k) for h in range(lists) for k in range(nu * L.UG_GROUPS)
        )


def test_ffn_build_key_and_contract_table():
    key = L.P1Build(tokens=1, upto="ffn", inter_r=INTER_R)
    assert key.upto == "ffn"
    with pytest.raises(ValueError):
        L.P1Build(tokens=1, upto="moe")
    # A16W4 by default; the format names the kernel symbol (``sym`` "ra")
    assert key.routed_act == L.ROUTED_ACT_DEFAULT == "bf16"
    assert set(L.ROUTED_ACTS) == {"fp4", "fp8", "bf16"}
    with pytest.raises(ValueError):
        L.P1Build(tokens=1, upto="ffn", inter_r=INTER_R, routed_act="fp16")
    from atom.mono.plan.build_key import symbol_params

    assert symbol_params(key)["ra"] == "bf16"
    # the routed planes' loaded width names the kernel too (``sym`` "ir"): an
    # FFN build needs one the kernel can address, a norm build has none
    assert symbol_params(key)["ir"] == INTER_R and key.planes.inter_r == INTER_R
    assert symbol_params(L.P1Build(tokens=1, upto="ffn", inter_r=768))["ir"] == 768
    for bad in (0, 600, 656):
        with pytest.raises(ValueError):
            L.P1Build(tokens=1, upto="ffn", inter_r=bad)
    with pytest.raises(ValueError):
        L.P1Build(tokens=1, inter_r=INTER_R)
    for act in ("fp4", "fp8"):
        k = L.P1Build(tokens=1, upto="ffn", inter_r=INTER_R, routed_act=act)
        assert symbol_params(k)["ra"] == act
        # no K-half hand-off outside A16W4
        assert {d.name for d in L.decls(k)} == set(L.REGIONS) - {"ugp"}
    got = {d.name: (d.space, d.writers, d.exchange) for d in L.decls(key)}
    assert set(got) == set(L.REGIONS)
    assert got["rlog"] == (Space.PEER, ("router",), False)
    assert got["rpart"] == (Space.SCRATCH, ("router",), True)
    assert got["ugf"][:2] == (Space.SCRATCH, ("ug",))
    # a half-0 ug task polls its half-1 task's partials, ug to ug
    assert got["ugp"] == (Space.SCRATCH, ("ug",), True)
    assert [d.name for d in L.decls(L.P1Build(tokens=1))] == list(L.NORM_REGIONS)


def test_moe_weights_table_names_the_binders_fields():
    import dataclasses

    from atom.models.minimax_m3.pro_mono.weights import ProMoeLayer

    fields = {f.name for f in dataclasses.fields(ProMoeLayer)}
    assert set(L.MOE_WEIGHTS) <= fields
    assert {"moe_w", "ffn_out", "plain"} <= set(L.ABI_NAMES)


@pytest.mark.parametrize("act", ROUTED_ACTS)
def test_ffn_lds_fits_at_the_widest_build(act):
    """The FFN build's LDS (``layer.build``'s structs, counted here) under the
    CU's: the contract run checks the binary's. W4A8 stages the x8 rows in
    ``xs`` (grown to ``X8_ROW`` words a token), not beside it; A16W4 a K half
    of the bf16 x there (``XH_ROW``: a whole row, 14 KB, would not fit 16)."""
    s = MAX_ROWS
    xs_row = {"fp4": L.O_ROW, "fp8": max(L.O_ROW, L.X8_ROW)}.get(
        act, max(L.O_ROW, L.XH_ROW)
    )
    x4l = s * L.X4_ROW if act == "fp4" else 4
    xsl = 4 if act == "bf16" else s * L.XSC_WORDS
    words = (
        s * xs_row + s * L.OSC_ROW + WAVES * 64 * 4 + L.O_ROWS * s + 2 * 11
        + x4l + xsl + s * L.SMSC_WORDS
        + s * L.TOPK * L.DOWN_ROWS + s * 2 * L.TOPK + 2 * EXPERTS
        + 2 * L.max_u(s) + 1 + s * L.TOPK + 3 * (128 + 16) + WAVES + 4
    )  # fmt: skip
    assert 4 * words <= LDS_BYTES
    if act == "bf16":
        # whole bf16 x rows would not fit: the reason for the K-half tasks
        whole = words + s * (2 * L.XH_WORDS + 4 - xs_row)
        assert 4 * whole > LDS_BYTES


def test_x8_rows_and_mids_hold_whole_mfma_steps():
    """A W4A8 B operand (k 16 kb .. + 16, 64 + 16 kb .. + 16 of a 128-k step)
    stays inside its token's x8 row and its pick's MXFP8 mid row."""
    assert L.X8_WORDS * 4 == HIDDEN and L.X8_ROW > L.X8_WORDS
    assert L.INTER8_WORDS * 4 == L.INTER_V
    assert L.inter_words("fp8") == L.INTER8_WORDS
    assert L.inter_words("fp4") == L.INTER_WORDS
    # an ug task's 32 mid columns are 8 words of the MXFP8 row (4 of MXFP4)
    assert L.UG_COLS // 4 == 8 and L.UG_GROUPS * 8 == L.INTER8_WORDS
    last = (L.W2_STEPS - 1) * 32 + 3 * 4 + 16 + 4
    assert last <= L.INTER8_WORDS


def test_a16w4_operands_follow_the_w13_and_w2_tiles():
    """A16W4's bf16 B operands: operand j of a 128-k step on lane group kb is
    k 128 st + 32 kb + 8 j .. + 8, the k of ``fp4_bf16_tile``'s operand j of
    the lane's weight tile (``fp4_tile_dword``: k 128 st + 32 kb .. + 32) --
    for x, the K half ``load_xh`` stages (``xh_word``), and for the mids, the
    words an ug task stores and the down reads (``interb_word``)."""
    for h, q, i, kb, j in itertools.product(
        range(L.UG_HALVES), range(2), range(L.W13_Q), range(4), range(4)
    ):
        st = h * L.W13_HALF + q * L.W13_Q + i
        w = L.xh_word(0, q * L.W13_Q + i, kb, j)
        # load_xh: x word h XH_WORDS + w of a token -> LDS word w of its row
        assert 2 * (h * L.XH_WORDS + w) == 128 * st + 32 * kb + 8 * j
        assert w + 4 <= L.XH_WORDS
    assert L.xh_word(1, 0, 0, 0) == L.XH_ROW and L.UG_HALVES * L.XH_WORDS * 2 == HIDDEN
    for st, kb, j in itertools.product(range(L.W2_STEPS), range(4), range(4)):
        w = L.interb_word(0, 4 * st + kb, 8 * j)
        assert 2 * w == 128 * st + 32 * kb + 8 * j and w + 4 <= L.INTERB_WORDS
    # an ug task's even column stores the word of its column and the next
    for g, col in itertools.product(range(L.UG_GROUPS), range(0, L.UG_COLS, 2)):
        assert 2 * L.interb_word(0, g, col) == L.UG_COLS * g + col
    assert L.interb_word(1, 0, 0) == L.INTERB_WORDS == L.INTER_V // 2


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


def test_mxfp4_dequantizes_to_bf16_exactly():
    """An e2m1 value times an E8M0 scale is a bf16 (codes 1 .. 250): the
    kernel's in-register dequant (``v_cvt_scalef32_pk_bf16_fp4``, as aiter's
    a16w4 kernels) loses nothing, so its bf16 MFMA multiplies the weight."""
    e2m1 = E2M1.repeat(250)
    scale = torch.exp2(torch.arange(1, 251).repeat_interleave(16).float() - 127)
    w = e2m1 * scale
    assert torch.equal(w.to(torch.bfloat16).float(), w)


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
    bound = logit_bound(c["x"], c["w_gate"])
    assert not route_failures(logits, picks, w, logits, c["bias"], bound)
    # fp32 logits summed in another order pass (the picks and weights then
    # follow from them); bf16-rounded ones (a router that rounds as the bf16
    # gate does) do not
    for name, mine, want in (
        ("fp32", c["x"].float() @ c["w_gate"].float().T, []),
        ("bf16", logits.to(torch.bfloat16).float(), ["logits"]),
    ):
        _, p, ww, _ = route_reference_of_logits(mine, c["bias"])
        assert route_failures(mine, p, ww, logits, c["bias"], bound) == want, name
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
        assert route_failures(logits, p, ww, logits, c["bias"], bound), name


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_near_tie_router_splits_fp32_from_bf16_logits(seed):
    """At Pro's widths (the GPU test's router generator): on the near-tie
    router the fp32 logits pick B and the bf16-rounded ones A, and the logit
    check flags the bf16-rounded logits."""
    g = torch.Generator().manual_seed(seed)
    x = (
        torch.randn(HIDDEN, generator=g) * (1 + 0.3 * torch.randn(HIDDEN, generator=g))
    ).to(torch.bfloat16)
    w_gate = (torch.randn(EXPERTS, HIDDEN, generator=g) * 0.02).to(torch.bfloat16)
    bias = torch.randn(EXPERTS, generator=g) * 0.02
    w, b, want32, want16 = near_tie_router(x, w_gate, bias)
    assert want32 != want16
    logits, picks, ww, _ = route_reference(x[None], w, b)
    assert sorted(picks[0].tolist()) == want32
    rounded = logits.to(torch.bfloat16).float()
    _, p16, w16, _ = route_reference_of_logits(rounded, b)
    assert sorted(p16[0].tolist()) == want16
    bound = logit_bound(x[None], w)
    assert not route_failures(logits, picks, ww, logits, b, bound)
    assert "logits" in route_failures(rounded, p16, w16, logits, b, bound)


def test_mxfp8_mid_rule():
    """The W4A8 mid: ``quantize_fp8``'s MXFP8 of the bf16 mids (the shared
    mid's rule), within half an e4m3 step."""
    g = torch.Generator().manual_seed(4)
    y = (torch.randn(8, 128, generator=g) * 3).to(torch.bfloat16)
    q, code = mxfp8(y)
    back = dequant(q, code)
    assert torch.equal(back, qdq_mxfp8(y.float()))
    scale = (code.to(torch.int32) << 23).view(torch.float32).repeat_interleave(32, 1)
    # RNE: half an e4m3 step (2^-3 relative), or half the subnormal step
    err = (back - y.float()).abs()
    assert (err <= y.float().abs() * 2.0**-4 + scale * 2.0**-10).all()
    # the block's max fits 448 x scale, half the scale would not
    amax = y.float().view(8, 4, 32).abs().amax(-1).repeat_interleave(32, 1)
    assert (amax <= 448 * scale).all() and (amax > 224 * scale).all()


@pytest.mark.parametrize("act", ROUTED_ACTS)
def test_the_routed_formats_differ_and_w4a8_reads_x8(act):
    """W4A8's routed input is the x8 the shared expert reads (an x8 scale code
    moves the routed output, x itself does not); W4A4's is MXFP4(x), A16W4's
    the bf16 x itself."""
    c = ffn_case(3, **NARROW, seed=7)
    w_sgu, w_sd = c["shared"]
    routed_only = dict(c, shared=(w_sgu * 0, w_sd * 0))
    base = case_ffn(routed_only, act=act)
    x_moved = case_ffn(dict(routed_only, x=c["x"] * 1.5), act=act)
    x8s_moved = case_ffn(dict(routed_only, x8s=c["x8s"] + 1), act=act)
    if act == "fp8":
        assert torch.equal(x_moved, base) and not torch.equal(x8s_moved, base)
    else:
        assert not torch.equal(x_moved, base) and torch.equal(x8s_moved, base)
    other = "fp4" if act == "fp8" else "fp8"
    assert row_relnorm(case_ffn(routed_only, act=other), base) > 0


@pytest.mark.parametrize("act", ROUTED_ACTS)
@pytest.mark.parametrize("s", [1, 3, 16])
def test_the_gpu_tests_ffn_bound_passes_sum_order_noise(s, act):
    """gate / up off by 2^-12 relative, at random (f32 sum order is ~2^-20):
    the bound holds. Off by 2^-20 nothing moves at all past the MX mids; the
    bf16 mid of A16W4 has no coarse grid to hide it, so a mid can move a bf16
    step there, far under the bound."""
    c = ffn_case(s, **NARROW, seed=11 + s)
    want = case_ffn(c, act=act)
    g = torch.Generator().manual_seed(s)

    def noisy(rel):
        def f(gg, uu):
            def jitter(t):
                sign = torch.randint(0, 2, t.shape, generator=g) * 2 - 1
                return t * (1 + rel * sign)

            return swiglu_oai(jitter(gg), jitter(uu))

        return f

    quiet = case_ffn(c, swiglu=noisy(2.0**-20), act=act)
    if act == "bf16":
        assert row_relnorm(quiet, want) <= FFN_BOUND / 16
    else:
        assert torch.equal(quiet, want)
    assert not ffn_failures(case_ffn(c, swiglu=noisy(2.0**-12), act=act), want)


@pytest.mark.parametrize("act", ROUTED_ACTS)
@pytest.mark.parametrize("s", [1, 3, 16])
def test_the_gpu_tests_ffn_bound_catches_injected_bugs(s, act):
    c = ffn_case(s, **NARROW, seed=11 + s)
    want = case_ffn(c, act=act)
    muts = ffn_mutations(
        c["x"], c["x8"], c["x8s"], c["picks"], c["weights"], c["routed"],
        c["shared"], NARROW["e"], act=act,
    )  # fmt: skip
    assert len(muts) >= 13
    missed = {
        k: row_relnorm(v, want) for k, v in muts.items() if not ffn_failures(v, want)
    }
    assert not missed, missed
    # and each by a margin
    assert min(row_relnorm(v, want) for v in muts.values()) > 2 * FFN_BOUND


def a16w4_emulation(c):
    """The routed output as aiter's a16w4 kernels compute it (gemm1_a16w4 at
    k_wave 4, gemm2_a16w4's atomic epilog): fp32 gate / up as four K quarters
    summed ((q0 + q1) + q2) + q3, bf16(SwiGLU), the fp32 down, bf16(w acc) a
    pick added to a bf16 output (bf16 atomics; top-k order here)."""
    x = c["x"].float()
    h = x.shape[1]
    out = []
    for t in range(x.shape[0]):
        y = torch.zeros(h, dtype=torch.bfloat16)
        for k in range(TOPK):
            w13, w2 = c["routed"](int(c["picks"][t, k]))
            qs = [
                w13[:, q * h // 4 : (q + 1) * h // 4].float()
                @ x[t, q * h // 4 : (q + 1) * h // 4]
                for q in range(4)
            ]
            gu = ((qs[0] + qs[1]) + qs[2]) + qs[3]
            i = gu.shape[0] // 2
            mid = swiglu_oai(gu[:i], gu[i:]).to(torch.bfloat16).float()
            acc = w2.float() @ mid
            pick = (acc * c["weights"][t, k].float()).to(torch.bfloat16)
            y = (y.float() + pick.float()).to(torch.bfloat16)
        out.append(y)
    return torch.stack(out)


@pytest.mark.parametrize("s", [1, 4, 16])
def test_the_a16w4_reference_is_aiters_a16w4_math(s):
    """``ffn_reference(act="bf16")``'s routed output is aiter's a16w4 kernels'
    up to rounding order (its bf16 picks summed in fp32, theirs by bf16
    atomics): 0.003 per row here, under a quarter of ``FFN_BOUND``, where the
    W4A8 / W4A4 references, which quantize the activations, are 0.06 / 0.27
    away."""
    c = ffn_case(s, **NARROW, seed=31 + s)
    w_sgu, w_sd = c["shared"]
    routed_only = dict(c, shared=(w_sgu * 0, w_sd * 0))
    want = a16w4_emulation(c)
    near = row_relnorm(case_ffn(routed_only, act="bf16"), want)
    assert near <= FFN_BOUND / 4, near
    for act in ("fp8", "fp4"):
        far = row_relnorm(case_ffn(routed_only, act=act), want)
        assert far > 8 * near, (act, far, near)


def test_neither_the_routed_nor_the_shared_output_drowns():
    """The generator's code ranges keep the two outputs of one size, so a
    lost or wrong one shows (``W2_CODES``, ``SHARED_CODES``), in either
    routed format."""
    c = ffn_case(3, **NARROW, seed=2)
    w_sgu, w_sd = c["shared"]
    for act in ROUTED_ACTS:
        routed = case_ffn(dict(c, shared=(w_sgu * 0, w_sd * 0)), act=act)
        shared = case_ffn(dict(c, weights=c["weights"] * 0), act=act)
        ratio = routed.float().norm(dim=-1) / shared.float().norm(dim=-1)
        assert ((ratio > 0.25) & (ratio < 4)).all(), (act, ratio)


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


@pytest.mark.parametrize("inter_r", LOADED_WIDTHS)
def test_fp4_tile_addressing_matches_aiters_shuffle(inter_r):
    """``fp4_tile_dword`` (the kernel's w13 / w2 loads) against the loader's
    (16, 16) shuffle of an expert's plane at the loaded width: lane l holds
    row 16 rg + l % 16, k 128 st + 32 (l / 16) .. + 32; the up block starts
    at row group ``w13_up_group`` (row ``inter_r``)."""
    pytest.importorskip("aiter")
    from aiter.ops.shuffle import shuffle_weight

    planes = RoutedPlanes(inter_r)
    assert L.w13_up_group(planes) * 16 == inter_r
    g = torch.Generator().manual_seed(0)
    for rows, k in ((planes.w13_rows, HIDDEN), (HIDDEN, inter_r)):
        b = torch.randint(0, 256, (2, rows, k // 2), generator=g, dtype=torch.uint8)
        sw = shuffle_weight(b, layout=(16, 16)).contiguous().view(-1)
        for rg, st, lane in itertools.product(
            (0, 1, rows // 16 - 1), (0, 3), _sample_lanes()
        ):
            dw = rows * k // 8 + L.fp4_tile_dword(rg, k, st, lane)  # expert 1
            want = b[1, 16 * rg + lane % 16]
            c0 = (128 * st + 32 * (lane // 16)) // 2
            assert torch.equal(sw[4 * dw : 4 * dw + 16], want[c0 : c0 + 16])


@pytest.mark.parametrize("inter_r", LOADED_WIDTHS)
def test_routed_scale_addressing_matches_aiters_shuffle(inter_r):
    """``w13_scale_word`` / ``w2_scale_word`` and the byte the kernel takes,
    against ``shuffle_scale`` of the whole plane (every expert's rows) at the
    loaded width; at 640 w2's 20 groups are shuffled as 24 (the tile pad)."""
    pytest.importorskip("aiter")
    from aiter.ops.shuffle import shuffle_scale

    planes = RoutedPlanes(inter_r)
    g = torch.Generator().manual_seed(1)
    rows13, cols13 = planes.w13_rows, HIDDEN // 32
    c13 = torch.randint(0, 256, (3 * rows13, cols13), generator=g, dtype=torch.uint8)
    s13 = shuffle_scale(c13).contiguous().view(-1)
    up = L.w13_up_group(planes)
    for e, rg, st, lane in itertools.product(
        (0, 2),
        (0, 1, up - 1, up, up + 1, rows13 // 16 - 1),
        (0, 1, 54, 55),
        _sample_lanes(),
    ):
        byte = 4 * L.w13_scale_word(planes, e, rg, st, lane) + (st % 2) * 2 + rg % 2
        assert s13[byte] == c13[e * rows13 + 16 * rg + lane % 16, 4 * st + lane // 16]
    cols2 = inter_r // 32
    c2 = torch.randint(0, 256, (3 * HIDDEN, cols2), generator=g, dtype=torch.uint8)
    s2 = shuffle_scale(c2).contiguous()
    assert s2.shape == (3 * HIDDEN, planes.w2_scale_cols)
    s2 = s2.view(-1)
    for e, rgd, st, lane in itertools.product(
        (0, 2), (0, 1, 7, L.DOWN_TASKS - 1), range(L.W2_STEPS), _sample_lanes()
    ):
        word = L.w2_scale_word(planes, e, rgd // 2, st // 2, lane)
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
