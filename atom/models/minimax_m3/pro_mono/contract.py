# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The phase-1 layer kernel's builds checked without a GPU: each traced
compile-only (``ARCH=gfx950``, no launch) and its mailbox hand-offs held to
``layout.decls``, its binary to no scratch (a persistent kernel that spills is
refused) and to the CU's LDS.

Usage: ARCH=gfx950 python -m atom.models.minimax_m3.pro_mono.contract [norm|ffn] [fp4|fp8] [S ...]
(the routed activation format: an FFN build's, default ``layout.ROUTED_ACT_DEFAULT``)
"""

import sys

from atom.models.minimax_m3.pro_mono.config import MAX_ROWS
from atom.models.minimax_m3.pro_mono.layout import (
    ROUTED_ACT_DEFAULT,
    ROUTED_ACTS,
    P1Build,
    decls,
)
from atom.mono.plan.execution import LDS_BYTES
from atom.mono.runtime.compile import (
    check_traced,
    launcher_resources,
    scratch_bytes,
)


def build_layer(key: P1Build):
    from atom.models.minimax_m3.pro_mono.kernels.layer import build

    return build(key)


def check_build(key: P1Build) -> dict:
    """Trace ``key``'s build, check its hand-offs (``ContractError``), then its
    resources (``RuntimeError``); returns the binary's resources."""
    from atom.models.minimax_m3.pro_mono.kernels.layer import ABI

    launch = build_layer(key)
    check_traced(launch, ABI, decls(key))
    spill = scratch_bytes(launch)
    res = launcher_resources(launch)
    problems = [f"scratch {spill} B a lane"] if spill else []
    for r in res:
        problems += [f"{k} {r[k]}" for k in ("vgpr_spill", "sgpr_spill") if r.get(k)]
        if r["lds"] > LDS_BYTES:
            problems.append(f"lds {r['lds']} > {LDS_BYTES}")
    if problems:
        raise RuntimeError(f"{key}: " + ", ".join(problems))
    return res[0]


def builds(rows, upto="norm", routed_act=ROUTED_ACT_DEFAULT) -> list[P1Build]:
    """Every build a runner makes at these row counts: lookahead on (layers
    4 .. 65) and off (66, 67), plus the debug build."""
    return [
        P1Build(tokens=s, upto=upto, lookahead=la, debug=dbg, routed_act=routed_act)
        for s in rows
        for la in (True, False)
        for dbg in (False, True)
    ]


if __name__ == "__main__":
    args = sys.argv[1:]
    upto = args.pop(0) if args and args[0] in ("norm", "ffn") else "norm"
    act = args.pop(0) if args and args[0] in ROUTED_ACTS else ROUTED_ACT_DEFAULT
    sizes = [int(a) for a in args] or list(range(1, MAX_ROWS + 1))
    for key in builds(sizes, upto, act):
        r = check_build(key)
        print(
            f"S={key.tokens} upto={key.upto} routed_act={key.routed_act}"
            f" lookahead={key.lookahead}"
            f" debug={key.debug}: contract holds; vgpr {r['vgpr']}"
            f" sgpr {r['sgpr']} lds {r['lds']}",
            flush=True,
        )
