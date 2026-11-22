#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fixed-seed session subset of the agentic-coding trace corpus.

Reads ``traces.jsonl`` of ``semianalysisai/cc-traces-weka-062126`` (one Weka
trace per line), drops every session whose peak requested context exceeds
``--max-context``, shuffles the rest with ``--seed`` and keeps the first
``--num-sessions``. Each kept session is written verbatim as its own
``<rank>_<id>.json`` in ``--out``, the directory form aiperf's ``weka_trace``
loader reads (``--custom-dataset-type weka_trace --input-file <dir>``); the
file names sort in the seeded order. ``subset.json`` beside them records the
selection.

A session's peak context is computed the way aiperf filters
``--max-context-length``: over parent and subagent requests, the input length
plus the output length, with an output of 0 counted as 1. So passing the same
cap to aiperf keeps every session written here.

Stdlib only, apart from ``huggingface_hub`` when ``--traces`` is not given.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

DATASET = "semianalysisai/cc-traces-weka-062126"
# Pinned, so that a seed names the same sessions in every run.
DATASET_REVISION = "23f152f6f0f9399a85901b89a6458def0ef16729"
TRACES_FILE = "traces.jsonl"
MANIFEST = "subset.json"

DEFAULT_SEED = 42
DEFAULT_MAX_CONTEXT = 128 * 1024
DEFAULT_NUM_SESSIONS = 64


def _request_context(request: dict[str, Any]) -> int:
    return int(request["in"]) + max(int(request["out"]), 1)


def trace_peak_context(trace: dict[str, Any]) -> int:
    """Largest input + output of any request in the session, subagents included."""
    peak = 0
    for request in trace["requests"]:
        if request["type"] == "subagent":
            for inner in request["requests"]:
                peak = max(peak, _request_context(inner))
        else:
            peak = max(peak, _request_context(request))
    return peak


def eligible_sessions(traces: Path, max_context: int) -> list[tuple[str, int]]:
    """``(session id, line number)`` of every session that fits ``max_context``."""
    eligible = []
    seen = set()
    with traces.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if not line.strip():
                continue
            trace = json.loads(line)
            session = trace["id"]
            if session in seen:
                raise ValueError(f"{traces}: duplicate session id {session!r}")
            seen.add(session)
            if trace_peak_context(trace) <= max_context:
                eligible.append((session, line_no))
    return eligible


def select_sessions(
    traces: Path, seed: int, max_context: int, num_sessions: int
) -> list[tuple[str, int]]:
    """The seeded subset, in replay order.

    The eligible sessions are sorted by id before the shuffle, so the result
    depends on the corpus content, the seed and the two limits only.
    """
    eligible = sorted(eligible_sessions(traces, max_context))
    if not eligible:
        raise ValueError(f"{traces}: no session fits a {max_context}-token context")
    random.Random(seed).shuffle(eligible)
    return eligible[:num_sessions]


def write_subset(
    traces: Path,
    out: Path,
    seed: int,
    max_context: int,
    num_sessions: int,
) -> list[str]:
    """Write the subset to ``out`` and return its session ids in replay order."""
    chosen = select_sessions(traces, seed, max_context, num_sessions)
    rank_of = {line_no: rank for rank, (_session, line_no) in enumerate(chosen)}
    out.mkdir(parents=True, exist_ok=True)
    # A stale file from an earlier subset would be replayed too.
    for stale in out.glob("*.json"):
        stale.unlink()
    with traces.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            rank = rank_of.get(line_no)
            if rank is None:
                continue
            session = chosen[rank][0]
            (out / f"{rank:04d}_{session}.json").write_text(
                line.rstrip("\n"), encoding="utf-8"
            )
    sessions = [session for session, _line_no in chosen]
    manifest = {
        "source": str(traces),
        "seed": seed,
        "max_context": max_context,
        "num_sessions": len(sessions),
        "sessions": sessions,
    }
    # Not a *.json the loader would pick up as a trace: it lives next to them.
    (out.parent / f"{out.name}.{MANIFEST}").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return sessions


def default_traces() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=DATASET,
            repo_type="dataset",
            filename=TRACES_FILE,
            revision=DATASET_REVISION,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--traces",
        type=Path,
        help=f"{TRACES_FILE} of {DATASET}; downloaded at {DATASET_REVISION[:9]} "
        "through the Hugging Face cache when omitted",
    )
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-context",
        type=int,
        default=DEFAULT_MAX_CONTEXT,
        help="drop sessions whose peak input + output exceeds this many tokens",
    )
    parser.add_argument("--num-sessions", type=int, default=DEFAULT_NUM_SESSIONS)
    args = parser.parse_args(argv)
    if args.num_sessions < 1:
        parser.error("--num-sessions must be at least 1")

    traces = args.traces or default_traces()
    sessions = write_subset(
        traces, args.out, args.seed, args.max_context, args.num_sessions
    )
    print(
        f"{len(sessions)} sessions (seed {args.seed}, max context "
        f"{args.max_context}) -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
