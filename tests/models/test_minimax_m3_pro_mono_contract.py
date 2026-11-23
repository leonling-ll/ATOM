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


@pytest.fixture(scope="module")
def arch():
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ARCH", "gfx950")
        yield


# (upto, routed_act): the norm build, the FFN build in W4A8 (the default) and
# in W4A4
BUILDS = [("norm", "fp8"), ("ffn", "fp8"), ("ffn", "fp4")]


@pytest.mark.parametrize("upto, act", BUILDS)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("lookahead", [True, False])
def test_the_layer_kernel_holds(arch, rows, lookahead, upto, act):
    contract.check_build(
        P1Build(tokens=rows, upto=upto, lookahead=lookahead, routed_act=act)
    )


@pytest.mark.parametrize("upto, act", BUILDS)
def test_the_debug_build_holds(arch, upto, act):
    contract.check_build(P1Build(tokens=16, upto=upto, debug=True, routed_act=act))


# the FFN builds whose norm runs two passes with a partial one (an SGPR peak:
# a value of the o stage held into the FFN spilled at 9 .. 13 rows)
@pytest.mark.parametrize("act", ["fp8", "fp4"])
@pytest.mark.parametrize("rows", [9, 13])
def test_the_ffn_builds_past_one_pass_hold(arch, rows, act):
    contract.check_build(
        P1Build(tokens=rows, upto="ffn", lookahead=True, routed_act=act)
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

    key = P1Build(tokens=8, upto="ffn")
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
        ({"rpart": {"exchange": False}}, "rpart polled by its own writer router"),
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
