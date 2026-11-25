# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The phase-1 layer kernel's builds checked without a GPU: each traced
compile-only (``ARCH=gfx950``, no launch) and its mailbox hand-offs held to
``layout.decls``, its binary to no scratch (a persistent kernel that spills is
refused) and to the CU's LDS.

Usage: ARCH=gfx950 python -m atom.models.minimax_m3.pro_mono.contract [norm|ffn] [fp4|fp8|bf16] [ir=N] [S ...]
(the routed activation format: an FFN build's, default ``layout.ROUTED_ACT_DEFAULT``;
``ir``: the routed planes' loaded intermediate, default the unpadded 640 Pro's
``pad_align`` 128 loads; the sizes default to every width ``upto`` builds,
``layout.max_tokens``)
"""

import sys

from atom.models.minimax_m3.pro_mono.config import ProDims
from atom.models.minimax_m3.pro_mono.layout import (
    ROUTED_ACT_DEFAULT,
    ROUTED_ACTS,
    P1Build,
    decls,
    max_tokens,
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


def builds(
    rows, upto="norm", routed_act=ROUTED_ACT_DEFAULT, inter_r=None
) -> list[P1Build]:
    """Every build a runner makes at these row counts: lookahead on (layers
    4 .. 65) and off (66, 67), plus the debug build. ``inter_r``: an FFN
    build's routed width (default ``ProDims.inter_valid``, unpadded)."""
    if upto == "ffn" and inter_r is None:
        inter_r = ProDims(8).inter_valid
    ir = inter_r if upto == "ffn" else 0
    return [
        P1Build(
            tokens=s,
            upto=upto,
            lookahead=la,
            debug=dbg,
            routed_act=routed_act,
            inter_r=ir,
        )
        for s in rows
        for la in (True, False)
        for dbg in (False, True)
    ]


if __name__ == "__main__":
    args = sys.argv[1:]
    upto = args.pop(0) if args and args[0] in ("norm", "ffn") else "norm"
    act = args.pop(0) if args and args[0] in ROUTED_ACTS else ROUTED_ACT_DEFAULT
    ir = int(args.pop(0)[3:]) if args and args[0].startswith("ir=") else None
    sizes = [int(a) for a in args] or list(range(1, max_tokens(upto, act) + 1))
    for key in builds(sizes, upto, act, ir):
        r = check_build(key)
        print(
            f"S={key.tokens} upto={key.upto} routed_act={key.routed_act}"
            f" inter_r={key.inter_r}"
            f" lookahead={key.lookahead}"
            f" debug={key.debug}: contract holds; vgpr {r['vgpr']}"
            f" sgpr {r['sgpr']} lds {r['lds']}",
            flush=True,
        )
