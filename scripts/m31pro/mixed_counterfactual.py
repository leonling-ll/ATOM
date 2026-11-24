#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Counterfactual sizing of mixed prefill+decode forwards for MiniMax-M3.1-Pro.

ATOM runs a prefill forward alone, then the decode step of the sequences it stalled.
This replays one rank's profiled step sequence (``kernel_roofline.read_forwards``, every
forward kept) with each prefill forward that a decode step directly follows absorbing
that step: the mixed forward costs the prefill's wall plus the decode rows' own GPU
work ``g`` (taken from the absorbed step's kernels) plus ``--host-ms`` of host work,
and the decode step's wall is saved. A run of prefill forwards absorbs one decode step,
at its last forward; a forward cut off by GPU idle (a request gap) absorbs none.

What ``g`` counts is the question a design answers, so four sets are reported:

  attn       the decode rows' attention prep, indexer score and select, sparse
             attention, index-cache writes, the DSpark draft and the post-forward
             sampling; every other kernel rides the prefill's launches
  format     ``attn``, plus the routed experts and their router and sort when the
             prefill runs them on MXFP4 activations (``a4w4``): decode rows keep
             their bf16-activation (a16w4) experts, so that mixed step calls the
             experts twice; a prefill already on a16w4 shares its call
  attn+moe   ``attn`` plus the routed experts, router and sort on every mixed step
  full       the decode step's whole GPU busy time: decode rows keep every kernel
             they run today, so only the step's launch and idle gaps are saved

Reported per set: forward time today and mixed, the tok/s gain (work is fixed, so
gain = time today / time mixed - 1) and the decode step interval p90 / p99 (time
between a decode sequence's consecutive steps, weighted by the step's sequences).
With ``--metrics-start`` / ``--metrics-end`` (two ``/metrics`` scrapes) the per-prefill
saving is also scaled to the run's prefill and decode counts and wall; with
``--summary`` (the proxy run's ``summary.json``) the run's tok/s/GPU is scaled by it.

usage:
  mixed_counterfactual.py TRACES... [--rank 0] [--host-ms 0] [--idle-cap-us 5000]
                          [--metrics-start P0 --metrics-end END] [--summary JSON]
                          [--json OUT]
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "m31pro_kernel_roofline", Path(__file__).with_name("kernel_roofline.py")
)
kr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(kr)

ATTN_ROLES = {"prep", "index_score", "index_topk", "sparse_attn"}
MOE_ROLES = {"routed_gemm1", "routed_gemm2", "router", "router_topk", "sort"}
INDEX_CACHE = re.compile(r"_compress_pool|_ring_store|compress_insert|index_cache")
SETS = ("attn", "format", "attn+moe", "full")
FP4_ACT = re.compile(r"a4w4")


def decode_increment(fw, prefill=None) -> dict:
    """GPU microseconds the decode rows of ``fw`` (a decode step) would add to a mixed
    forward with ``prefill``, per set."""
    attn = moe = 0.0
    for _, dur, role, name, part, _ in fw["kernels"]:
        if part in ("draft", "post") or role in ATTN_ROLES or INDEX_CACHE.search(name):
            attn += dur
        elif role in MOE_ROLES:
            moe += dur
    busy = kr.union_len([(k[0], k[0] + k[1]) for k in fw["kernels"]])
    fp4 = prefill is not None and any(
        k[2] in MOE_ROLES and FP4_ACT.search(k[3]) for k in prefill["kernels"]
    )
    return {
        "attn": attn,
        "format": attn + (moe if fp4 else 0.0),
        "attn+moe": attn + moe,
        "full": busy,
    }


def replay(fwds, which: str, host_us: float):
    """``(steps today, steps mixed, absorbed pairs)``; a step is ``(wall us, decode
    sequences)``, a ``None`` step a break in the sequence (no interval spans it)."""

    def joined(i):
        return i + 1 < len(fwds) and (
            abs(fwds[i]["start"] + fwds[i]["wall"] - fwds[i + 1]["start"]) < 1.0
        )

    today, mixed, pairs = [], [], []
    i = 0
    while i < len(fwds):
        f = fwds[i]
        nxt = fwds[i + 1] if joined(i) else None
        if f["prefill"] and nxt is not None and nxt["decode"]:
            g = decode_increment(nxt, f)[which]
            today += [(f["wall"], 0), (nxt["wall"], nxt["seqs"])]
            mixed.append((f["wall"] + g + host_us, nxt["seqs"]))
            pairs.append(
                {
                    "rows": f["rows"],
                    "prefill_us": f["wall"],
                    "decode_us": nxt["wall"],
                    "g_us": g,
                    "saved_us": nxt["wall"] - g - host_us,
                }
            )
            i += 2
        else:
            step = (f["wall"], 0 if f["prefill"] else f["seqs"])
            today.append(step)
            mixed.append(step)
            i += 1
        if not joined(i - 1):
            today.append(None)
            mixed.append(None)
    return today, mixed, pairs


def intervals(steps):
    """``[(us between a decode sequence's consecutive steps, weight)]``."""
    out, t, last = [], 0.0, None
    for st in steps:
        if st is None:
            t, last = 0.0, None
            continue
        t += st[0]
        if st[1]:
            if last is not None:
                out.append((t - last, st[1]))
            last = t
    return out


def wpct(samples, q: float) -> float:
    """Weighted percentile (the smallest value covering ``q`` of the weight)."""
    samples = sorted(samples)
    total = sum(w for _, w in samples)
    acc = 0.0
    for v, w in samples:
        acc += w
        if acc >= q * total:
            return v
    return float("nan")


def read_metrics(path) -> dict:
    """Prefill and decode target forwards and the refresh timestamp of one scrape."""
    out = {"prefill": 0.0, "decode": 0.0, "ts": None}
    for line in Path(path).read_text().splitlines():
        if line.startswith("atom:forward_steps_total{") and 'model="target"' in line:
            phase = re.search(r'phase="(\w+)"', line).group(1)
            if phase in ("prefill", "decode"):
                out[phase] += float(line.rsplit(" ", 1)[1])
        elif line.startswith("atom:metrics_last_refresh_timestamp_seconds"):
            out["ts"] = float(line.rsplit(" ", 1)[1])
    return out


def size(fwds, host_us: float, run=None, base_tok_s=None) -> dict:
    res = {}
    n_prefill = sum(1 for f in fwds if f["prefill"])
    for which in SETS:
        today, mixed, pairs = replay(fwds, which, host_us)
        t0 = sum(s[0] for s in today if s)
        t1 = sum(s[0] for s in mixed if s)
        it0, it1 = intervals(today), intervals(mixed)
        r = {
            "prefills": n_prefill,
            "absorbed": len(pairs),
            "decode_steps": sum(1 for f in fwds if f["decode"]),
            "time_today_ms": t0 / 1e3,
            "time_mixed_ms": t1 / 1e3,
            "gain_pct": (t0 / t1 - 1) * 100 if t1 else 0.0,
            "g_ms_mean": (
                sum(p["g_us"] for p in pairs) / len(pairs) / 1e3 if pairs else 0.0
            ),
            "saved_ms_per_prefill": sum(p["saved_us"] for p in pairs)
            / max(n_prefill, 1)
            / 1e3,
            "itl_p90_ms": (wpct(it0, 0.9) / 1e3, wpct(it1, 0.9) / 1e3),
            "itl_p99_ms": (wpct(it0, 0.99) / 1e3, wpct(it1, 0.99) / 1e3),
        }
        if run:
            saved_s = r["saved_ms_per_prefill"] / 1e3 * run["prefill"]
            r["run_saved_s"] = saved_s
            r["run_gain_pct"] = (run["seconds"] / (run["seconds"] - saved_s) - 1) * 100
            r["run_ms_per_decode_step"] = saved_s * 1e3 / max(run["decode"], 1)
            if base_tok_s:
                r["tok_s_gpu"] = (
                    base_tok_s,
                    base_tok_s * (1 + r["run_gain_pct"] / 100),
                )
        res[which] = r
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--host-ms", type=float, default=0.0)
    ap.add_argument("--idle-cap-us", type=float, default=5000.0)
    ap.add_argument("--metrics-start")
    ap.add_argument("--metrics-end")
    ap.add_argument("--summary")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    files = kr.find_traces(a.paths, a.rank)
    if not files:
        sys.exit(f"no traces under {a.paths}")
    # windows are concatenated: a window's last forward never reaches the next
    # window's first, so `replay` breaks the sequence there
    fwds = []
    for p in files:
        fwds += kr.read_forwards(p, a.idle_cap_us, decode_only=False)
    run = None
    if a.metrics_start and a.metrics_end:
        m0, m1 = read_metrics(a.metrics_start), read_metrics(a.metrics_end)
        run = {k: m1[k] - m0[k] for k in ("prefill", "decode", "ts")}
        run["seconds"] = run.pop("ts")
    base = None
    if a.summary:
        base = json.loads(Path(a.summary).read_text()).get("tok_s_per_gpu")
    res = size(fwds, a.host_ms * 1e3, run, base)
    if run:
        print(
            f"run: {run['prefill']:.0f} prefill and {run['decode']:.0f} decode "
            f"forwards over {run['seconds']:.0f} s"
        )
    for which, r in res.items():
        line = (
            f"{which:9s} absorbed {r['absorbed']}/{r['prefills']} prefills; "
            f"g {r['g_ms_mean']:.2f} ms; traces {r['time_today_ms']:.0f} -> "
            f"{r['time_mixed_ms']:.0f} ms ({r['gain_pct']:+.2f}%); step interval p90 "
            f"{r['itl_p90_ms'][0]:.1f} -> {r['itl_p90_ms'][1]:.1f} ms, p99 "
            f"{r['itl_p99_ms'][0]:.1f} -> {r['itl_p99_ms'][1]:.1f} ms"
        )
        if run:
            line += (
                f"; run {r['run_saved_s']:.1f} s saved, "
                f"{r['run_ms_per_decode_step']:.2f} ms a decode step, "
                f"{r['run_gain_pct']:+.2f}%"
            )
        if "tok_s_gpu" in r:
            line += f"; tok/s/GPU {r['tok_s_gpu'][0]:.0f} -> {r['tok_s_gpu'][1]:.0f}"
        print(line)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
