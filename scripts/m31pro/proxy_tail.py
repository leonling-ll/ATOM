#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Per-request tail of an agentic proxy run, and what it costs the p90 metric.

For each ``profile_export.jsonl`` (one proxy point) this reports:

* the final-chunk excess: the median final inter-chunk gap minus the median
  of each request's median gap (a request's last chunk arriving late);
* decode stalls: inter-chunk gaps above ``--stall-ms``, their summed time,
  and how many of them overlap another request's prefill (its start to its
  first token) -- the split between prefill interference and anything else;
* p90 E2E-normalized interactivity and three what-ifs recomputed from the
  same records: the final-chunk excess removed, every stall cut to its
  request's median gap, and TTFT shortened by ``--ttft-shave-ms``.

With ``--contamination`` it reports instead what makes a point's number
untrustworthy, per point:

* requests whose TTFT exceeds ``--outlier-ttft-s`` (a first-use compile, a
  stall), and the p90 with each of those TTFTs replaced by the median TTFT of
  the point's other requests in the same new-token band (uncached input
  tokens, powers of two; the nearest band holding any when its own has none),
  labelled ``corrected (outliers)``;
* the bootstrap 90% band of the p90 (``BOOTSTRAP_RESAMPLES`` resamples of the
  requests, seed ``BOOTSTRAP_SEED``);
* the Triton cache entries and aiter JIT modules that appeared during the
  replay, from the ``compile_counts.txt`` that ``proxy_bench.sh`` writes
  beside ``aiperf/``; a WARNING when any did;
* a FATAL line when the point's ``server.log`` (beside ``aiperf/`` or one
  directory up) holds ``no tuned FlyDSL config``: that aiter lacks Pro's tuned
  MoE rows, so the point measures a slower MoE. The script then exits 1.

The metric and the record filter come from ``interactivity.py`` (the
dashboard's implementation), so the baseline column equals its
``agentic_interactivity`` value. aiperf's ITL is (latency - TTFT) / (OSL - 1),
so a chunk that arrives ``d`` earlier lowers ITL by ``d / (OSL - 1)``.
What-ifs are arithmetic on measured records, not measurements.

    python3 scripts/m31pro/proxy_tail.py <results dir or profile_export.jsonl> ...
    python3 scripts/m31pro/proxy_tail.py --contamination <results dir> ...
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github/scripts/atomesh"))
from interactivity import (
    PROFILING_PHASE,
    RECORDS_FILENAME,
    _metric,
    agentic_throughput,
    iter_records,
    percentile_linear,
)

GAPS_KEY = "inter_chunk_latency"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
COMPILE_COUNTS = "compile_counts.txt"
UNTUNED_FLYDSL = "no tuned FlyDSL config"


def load(path: Path) -> tuple[list[dict], list[tuple[float, float, str]]]:
    """Requests the metric counts, and every request's prefill span (warmup too)."""
    requests, prefills = [], []
    for record, _raw in iter_records(path):
        if record is None or record.get("error"):
            continue
        meta = record.get("metadata") or {}
        ttft_ms = _metric(record, "time_to_first_token")
        start_ns = meta.get("request_start_ns")
        if start_ns and ttft_ms is not None:
            start = start_ns / 1e9
            prefills.append((start, start + ttft_ms / 1e3, meta.get("x_request_id")))
        if meta.get("benchmark_phase") != PROFILING_PHASE:
            continue
        itl_ms = _metric(record, "inter_token_latency")
        osl = _metric(record, "output_sequence_length")
        if itl_ms is None or ttft_ms is None or not osl or osl <= 0:
            continue
        gaps = (record.get("metrics") or {}).get(GAPS_KEY) or {}
        gaps = gaps.get("value") if isinstance(gaps, dict) else gaps
        isl = int(_metric(record, "input_sequence_length") or 0)
        cached = int(_metric(record, "usage_prompt_cache_read_tokens") or 0)
        requests.append(
            {
                "id": meta.get("x_request_id"),
                # position among the counted records of the file
                "index": len(requests),
                "isl": isl,
                "new": max(isl - cached, 0),
                "first_token": (start_ns or 0) / 1e9 + ttft_ms / 1e3,
                "itl": itl_ms / 1e3,
                "ttft": ttft_ms / 1e3,
                "osl": osl,
                "gaps": [g / 1e3 for g in gaps or []],
            }
        )
    return requests, prefills


def p90_metric(requests: list[dict], save=lambda r: 0.0, ttft_shave: float = 0.0):
    """1 / p90(eff), with ``save(r)`` seconds taken off the decode span."""
    effs = []
    for r in requests:
        itl = r["itl"] - (save(r) / (r["osl"] - 1) if r["osl"] > 1 else 0.0)
        effs.append(itl + max(r["ttft"] - ttft_shave, 0.0) / r["osl"])
    return 1.0 / percentile_linear(effs, 90.0)


def final_excess(r: dict) -> float:
    """How much later than the request's median gap its last chunk came."""
    if len(r["gaps"]) < 4:
        return 0.0
    return max(r["gaps"][-1] - statistics.median(r["gaps"]), 0.0)


def analyse(path: Path, stall_s: float, ttft_shave_s: float, gpus: int) -> dict:
    requests, prefills = load(path)
    if not requests:
        raise ValueError(f"no profiling requests in {path}")
    long = [r for r in requests if len(r["gaps"]) >= 4]
    stalls, beside_prefill = [], []
    for r in requests:
        t = r["first_token"]
        for g in r["gaps"]:
            if g > stall_s:
                stalls.append(g)
                if any(
                    p0 < t + g and p1 > t for p0, p1, rid in prefills if rid != r["id"]
                ):
                    beside_prefill.append(g)
            t += g

    def stall_save(r):
        if not r["gaps"]:
            return 0.0
        med = statistics.median(r["gaps"])
        return sum(g - med for g in r["gaps"] if g > stall_s)

    ms = 1e3
    return {
        # <arm>/<cN> of <arm>/<cN>/aiperf/profile_export.jsonl, whatever the cwd
        "point": "/".join(path.resolve().parents[1].parts[-2:]),
        "requests": len(requests),
        "final_gap_ms": (
            statistics.median(r["gaps"][-1] for r in long) * ms if long else None
        ),
        "median_gap_ms": (
            statistics.median(statistics.median(r["gaps"]) for r in long) * ms
            if long
            else None
        ),
        "stalls": len(stalls),
        "stall_s": sum(stalls),
        "stalls_beside_prefill": len(beside_prefill),
        "stall_beside_prefill_s": sum(beside_prefill),
        "chunks": sum(len(r["gaps"]) for r in requests),
        "ttft_p50_ms": percentile_linear([r["ttft"] for r in requests], 50.0) * ms,
        "ttft_p90_ms": percentile_linear([r["ttft"] for r in requests], 90.0) * ms,
        "itl_p50_ms": percentile_linear([r["itl"] for r in requests], 50.0) * ms,
        "p90_e2e_norm": p90_metric(requests),
        "whatif_no_final_excess": p90_metric(requests, final_excess),
        "whatif_no_stalls": p90_metric(requests, stall_save),
        "whatif_ttft_shaved": p90_metric(requests, ttft_shave=ttft_shave_s),
        "tok_s_per_gpu": agentic_throughput(path)["total_tput_tps"] / gpus,
    }


def ttft_corrected(requests: list[dict], limit_s: float) -> list[dict]:
    """The requests with each TTFT above ``limit_s`` replaced by its band's median."""

    def band(r):
        return r["new"].bit_length()

    normal = {}
    for r in requests:
        if r["ttft"] <= limit_s:
            normal.setdefault(band(r), []).append(r["ttft"])
    if not normal:
        return requests
    out = []
    for r in requests:
        if r["ttft"] > limit_s:
            near = min(normal, key=lambda b: (abs(b - band(r)), b))
            r = dict(r, ttft=statistics.median(normal[near]))
        out.append(r)
    return out


def p90_band(requests: list[dict]) -> tuple[float, float]:
    """Bootstrap 90% band of the p90 metric, resampling requests."""
    effs = [r["itl"] + r["ttft"] / r["osl"] for r in requests]
    rnd = random.Random(BOOTSTRAP_SEED)
    values = sorted(
        1.0 / percentile_linear(rnd.choices(effs, k=len(effs)), 90.0)
        for _ in range(BOOTSTRAP_RESAMPLES)
    )
    return (
        values[int(0.05 * BOOTSTRAP_RESAMPLES)],
        values[int(0.95 * BOOTSTRAP_RESAMPLES) - 1],
    )


def compile_delta(point_dir: Path) -> dict | None:
    """New Triton cache entries and aiter JIT modules over the replay."""
    path = point_dir / COMPILE_COUNTS
    if not path.is_file():
        return None
    counts = {}
    for line in path.read_text().splitlines():
        when, triton, jit = line.split()
        counts[when] = (int(triton), int(jit))
    if not {"before", "after"} <= counts.keys():
        return None
    return {
        "triton_cache_entries": counts["after"][0] - counts["before"][0],
        "aiter_jit_modules": counts["after"][1] - counts["before"][1],
    }


def untuned_flydsl(point_dir: Path) -> tuple[Path, int] | None:
    """The point's server log and its untuned FlyDSL lines, or None without a log."""
    for log in (point_dir / "server.log", point_dir.parent / "server.log"):
        if log.is_file():
            with log.open(errors="replace") as handle:
                return log, sum(UNTUNED_FLYDSL in line for line in handle)
    return None


def contamination(path: Path, outlier_ttft_s: float) -> dict:
    requests, _ = load(path)
    if not requests:
        raise ValueError(f"no profiling requests in {path}")
    point_dir = path.resolve().parents[1]
    flydsl = untuned_flydsl(point_dir)
    return {
        "point": "/".join(point_dir.parts[-2:]),
        "requests": len(requests),
        "p90_e2e_norm": p90_metric(requests),
        "p90_band": p90_band(requests),
        "ttft_outliers": [
            {k: r[k] for k in ("index", "isl", "new", "osl", "ttft")}
            for r in requests
            if r["ttft"] > outlier_ttft_s
        ],
        "p90_corrected": p90_metric(ttft_corrected(requests, outlier_ttft_s)),
        "compiles": compile_delta(point_dir),
        "server_log": str(flydsl[0]) if flydsl else None,
        "untuned_flydsl_lines": flydsl[1] if flydsl else None,
    }


def print_contamination(row: dict, outlier_ttft_s: float) -> None:
    lo, hi = row["p90_band"]
    print(
        f"{row['point']}: p90 E2E-norm {row['p90_e2e_norm']:.2f} "
        f"(bootstrap 90% [{lo:.1f}, {hi:.1f}]), "
        f"corrected (outliers) {row['p90_corrected']:.2f}, n {row['requests']}"
    )
    print(f"  TTFT > {outlier_ttft_s:g} s: {len(row['ttft_outliers'])}")
    for r in row["ttft_outliers"]:
        print(
            f"    #{r['index']}: ISL {r['isl']}, new {r['new']}, "
            f"OSL {r['osl']:.0f}, TTFT {r['ttft']:.2f} s"
        )
    compiles = row["compiles"]
    if compiles is None:
        print(f"  compiles during the replay: not recorded ({COMPILE_COUNTS})")
    else:
        print(
            f"  compiles during the replay: {compiles['triton_cache_entries']} Triton "
            f"cache entries, {compiles['aiter_jit_modules']} aiter JIT modules"
        )
        if any(compiles.values()):
            print(
                f"WARNING: {row['point']}: kernels compiled during the replay "
                "(aiperf warmup included); one in the profiling phase shows as a "
                "TTFT outlier"
            )
    if row["server_log"] is None:
        print("  server log: not found")
    else:
        print(
            f"  '{UNTUNED_FLYDSL}' lines in the server log: {row['untuned_flydsl_lines']}"
        )
    if row["untuned_flydsl_lines"]:
        print(
            f"FATAL: {row['point']}: {row['server_log']} has "
            f"{row['untuned_flydsl_lines']} '{UNTUNED_FLYDSL}' lines: this aiter "
            "lacks Pro's tuned MoE rows, so the point is not comparable"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--stall-ms", type=float, default=100.0)
    parser.add_argument("--ttft-shave-ms", type=float, default=30.0)
    parser.add_argument("--gpus", type=int, default=8)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--contamination", action="store_true")
    parser.add_argument("--outlier-ttft-s", type=float, default=5.0)
    args = parser.parse_args()
    files = []
    for p in args.paths:
        files += [p] if p.is_file() else sorted(p.rglob(RECORDS_FILENAME))
    if not files:
        sys.exit(f"no {RECORDS_FILENAME} under {args.paths}")
    if args.contamination:
        rows = [contamination(f, args.outlier_ttft_s) for f in files]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                print_contamination(row, args.outlier_ttft_s)
        sys.exit(1 if any(row["untuned_flydsl_lines"] for row in rows) else 0)
    rows = [
        analyse(f, args.stall_ms / 1e3, args.ttft_shave_ms / 1e3, args.gpus)
        for f in files
    ]
    if args.json:
        print(json.dumps(rows, indent=2))
        return
    print(
        "| point | n | final / median gap ms | stalls (beside prefill) | stall s (beside prefill) "
        f"| TTFT p50 / p90 ms | ITL p50 ms | p90 E2E-norm | no final excess | no stalls "
        f"| TTFT -{args.ttft_shave_ms:g} ms | tok/s/GPU |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        gaps = (
            "-"
            if r["final_gap_ms"] is None
            else f"{r['final_gap_ms']:.1f} / {r['median_gap_ms']:.1f}"
        )
        print(
            f"| {r['point']} | {r['requests']} | {gaps} | {r['stalls']} ({r['stalls_beside_prefill']}) "
            f"| {r['stall_s']:.2f} ({r['stall_beside_prefill_s']:.2f}) "
            f"| {r['ttft_p50_ms']:.0f} / {r['ttft_p90_ms']:.0f} | {r['itl_p50_ms']:.2f} "
            f"| {r['p90_e2e_norm']:.2f} | {r['whatif_no_final_excess']:.2f} | {r['whatif_no_stalls']:.2f} "
            f"| {r['whatif_ttft_shaved']:.2f} | {r['tok_s_per_gpu']:.1f} |"
        )


if __name__ == "__main__":
    main()
