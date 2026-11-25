# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The phase-1 layer kernel's hand-off contract on real builds: traced
compile-only for gfx950 (no launch, no GPU), every mailbox access held to
``layout.decls``, no spills, LDS within the CU. The negative controls check the
same trace against tables that are wrong, so a pass means something."""

import dataclasses

import pytest

pytest.importorskip("aiter")
pytest.importorskip("flydsl")

from atom.models.minimax_m3.pro_mono import contract
from atom.models.minimax_m3.pro_mono.layout import P1Build, decls
from atom.mono.plan.check import ContractError, check
from atom.mono.plan.trace import Space

# one row, one 8-row DSpark k7 verify, and the widest build (two token passes)
ROWS = (1, 8, 16)
# the routed planes' width an FFN build bakes: Pro's at pad_align 128
INTER_R = 640


def _ir(upto):
    return INTER_R if upto == "ffn" else 0


@pytest.fixture(scope="module")
def arch():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ARCH", "gfx950")
        yield


# (upto, routed_act): the norm build, the FFN build in A16W4 (the default),
# W4A8 and W4A4
BUILDS = [("norm", "bf16"), ("ffn", "bf16"), ("ffn", "fp8"), ("ffn", "fp4")]


@pytest.mark.parametrize("upto, act", BUILDS)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("lookahead", [True, False])
def test_the_layer_kernel_holds(arch, rows, lookahead, upto, act):
    contract.check_build(
        P1Build(
            tokens=rows,
            upto=upto,
            lookahead=lookahead,
            routed_act=act,
            inter_r=_ir(upto),
        )
    )


@pytest.mark.parametrize("upto, act", BUILDS)
def test_the_debug_build_holds(arch, upto, act):
    contract.check_build(
        P1Build(tokens=16, upto=upto, debug=True, routed_act=act, inter_r=_ir(upto))
    )


# the norm builds past one token tile: the o stage's second tile and AR #1's
# third pass, at one wave's third token and at every wave's (the widest)
@pytest.mark.parametrize("rows", [17, 24])
@pytest.mark.parametrize("lookahead", [True, False])
@pytest.mark.parametrize("debug", [False, True])
def test_the_two_tile_norm_builds_hold(arch, rows, lookahead, debug):
    contract.check_build(P1Build(tokens=rows, lookahead=lookahead, debug=debug))


# the FFN builds whose norm runs two passes with a partial one (an SGPR peak:
# a value of the o stage held into the FFN spilled at 9 .. 13 rows)
@pytest.mark.parametrize("act", ["bf16", "fp8", "fp4"])
@pytest.mark.parametrize("rows", [9, 13])
def test_the_ffn_builds_past_one_pass_hold(arch, rows, act):
    contract.check_build(
        P1Build(
            tokens=rows, upto="ffn", inter_r=INTER_R, lookahead=True, routed_act=act
        )
    )


@pytest.fixture(scope="module")
def traced(arch):
    from atom.models.minimax_m3.pro_mono.kernels.layer import ABI
    from atom.mono.runtime.compile import trace

    key = P1Build(tokens=8)
    return key, trace(contract.build_layer(key), ABI)


@pytest.fixture(scope="module")
def traced_ffn(arch):
    from atom.models.minimax_m3.pro_mono.kernels.layer import ABI
    from atom.mono.runtime.compile import trace

    key = P1Build(tokens=8, upto="ffn", inter_r=INTER_R)
    return key, trace(contract.build_layer(key), ABI)


@pytest.mark.parametrize(
    "wrong, msg",
    [
        ({"agsq": {"writer": "norm"}}, "agsq polled by its own writer norm"),
        ({"rs": {"space": Space.SCRATCH}}, "rs in o as peer, declared scratch"),
        ({"ag": {"writer": "o"}}, "ag put by reduce, its writer is o"),
    ],
)
def test_a_wrong_declaration_is_caught(traced, wrong, msg):
    key, rec = traced
    check(rec, decls(key))
    bad = [
        dataclasses.replace(d, **wrong[d.name]) if d.name in wrong else d
        for d in decls(key)
    ]
    with pytest.raises(ContractError, match=msg):
        check(rec, bad)


@pytest.mark.parametrize(
    "wrong, msg",
    [
        ({"ugf": {"writer": "down"}}, "ugf put by ug, its writer is down"),
        ({"rlog": {"space": Space.SCRATCH}}, "rlog in router as peer"),
        ({"route": {"writer": "union"}}, "route put by route, its writer is union"),
        # the router polls its K slice of x as the norm put it
        ({"xr": {"writer": "router"}}, "xr put by norm, its writer is router"),
        # A16W4 (the default): the half-0 task polls its half-1 task's partials
        ({"ugp": {"exchange": False}}, "ugp polled by its own writer ug"),
        ({"ugp": {"writer": "down"}}, "ugp put by ug, its writer is down"),
    ],
)
def test_a_wrong_ffn_declaration_is_caught(traced_ffn, wrong, msg):
    key, rec = traced_ffn
    check(rec, decls(key))
    bad = [
        dataclasses.replace(d, **wrong[d.name]) if d.name in wrong else d
        for d in decls(key)
    ]
    with pytest.raises(ContractError, match=msg):
        check(rec, bad)


def test_the_ffn_stages_x_before_it_polls_the_route(traced_ffn):
    """``ffn.run_ffn``: every CTA stages the routed B operands (``load_xb``,
    behind its poll of the norm CTAs' ``xrdy``) before it polls ``route``, so
    the staging runs while the route is in flight and only the union is left
    between the route and the first ug weights."""
    _, rec = traced_ffn
    polls = [a.region for a in rec.accesses if a.stage == "union" and a.kind == "poll"]
    assert "xrdy" in polls and "route" in polls, polls
    last_x = max(i for i, r in enumerate(polls) if r == "xrdy")
    assert last_x < polls.index("route"), polls
