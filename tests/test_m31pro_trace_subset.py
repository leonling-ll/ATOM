# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the MiniMax-M3.1-Pro proxy benchmark's trace subset."""

import importlib.util
import json
import random
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/m31pro/trace_subset.py"
SPEC = importlib.util.spec_from_file_location("m31pro_trace_subset", SCRIPT)
trace_subset = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(trace_subset)

CAP = 1000


def _request(t, isl, osl, kind="s"):
    return {
        "t": t,
        "type": kind,
        "model": "claude",
        "in": isl,
        "out": osl,
        "hash_ids": list(range(max(isl // 64, 1))),
    }


def _subagent(t, requests):
    return {
        "t": t,
        "type": "subagent",
        "agent_id": "agent_001",
        "subagent_type": "Explore",
        "status": "completed",
        "requests": requests,
        "models": ["claude"],
    }


def _trace(session, requests):
    return {
        "id": session,
        "models": ["claude"],
        "block_size": 64,
        "hash_id_scope": "local",
        "requests": requests,
    }


def _corpus():
    """Sessions around CAP, keyed by whether they fit it."""
    rng = random.Random(0)
    fits, too_long = [], []
    for i in range(40):
        isl = rng.randrange(100, 900)
        fits.append(
            _trace(f"fit{i:02d}", [_request(0.0, isl, 50), _request(1.0, isl + 40, 9)])
        )
    # Exactly at the cap, with an output of 0 counted as 1.
    fits.append(_trace("edge", [_request(0.0, CAP - 1, 0, kind="n")]))
    too_long += [
        _trace("over_parent", [_request(0.0, CAP, 1)]),
        _trace("over_zero_out", [_request(0.0, CAP, 0)]),
        # Only a subagent's inner request exceeds the cap.
        _trace(
            "over_subagent",
            [_request(0.0, 200, 10), _subagent(1.0, [_request(1.5, CAP - 5, 6)])],
        ),
    ]
    return fits, too_long


@pytest.fixture
def traces(tmp_path):
    fits, too_long = _corpus()
    rows = fits + too_long
    random.Random(1).shuffle(rows)
    path = tmp_path / "traces.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path, {t["id"] for t in fits}, {t["id"] for t in too_long}


def _replay(out):
    """The request list a replay of ``out`` sends, in file order."""
    return [
        (path.name, json.loads(path.read_text())["requests"])
        for path in sorted(out.glob("*.json"))
    ]


def test_same_seed_same_request_list(traces, tmp_path):
    path, _fits, _ = traces
    a = trace_subset.write_subset(path, tmp_path / "a", 7, CAP, 16)
    b = trace_subset.write_subset(path, tmp_path / "b", 7, CAP, 16)
    assert a == b
    assert len(a) == 16
    assert _replay(tmp_path / "a") == _replay(tmp_path / "b")


def test_seed_changes_the_subset(traces, tmp_path):
    path, _fits, _ = traces
    assert trace_subset.write_subset(
        path, tmp_path / "a", 7, CAP, 16
    ) != trace_subset.write_subset(path, tmp_path / "b", 8, CAP, 16)


def test_subset_independent_of_corpus_order(traces, tmp_path):
    path, _fits, _ = traces
    lines = path.read_text().splitlines(keepends=True)
    reordered = tmp_path / "reordered.jsonl"
    reordered.write_text("".join(reversed(lines)))
    assert trace_subset.select_sessions(path, 3, CAP, 10) == [
        (session, len(lines) - 1 - line_no)
        for session, line_no in trace_subset.select_sessions(reordered, 3, CAP, 10)
    ]


def test_cap_drops_every_session_that_does_not_fit(traces, tmp_path):
    path, fits, too_long = traces
    kept = trace_subset.write_subset(path, tmp_path / "out", 0, CAP, 10_000)
    assert set(kept) == fits
    assert not set(kept) & too_long
    manifest = json.loads((tmp_path / "out.subset.json").read_text())
    assert manifest["sessions"] == kept
    assert manifest["max_context"] == CAP


def test_rewrite_removes_stale_sessions(traces, tmp_path):
    path, _fits, _ = traces
    out = tmp_path / "out"
    trace_subset.write_subset(path, out, 0, CAP, 20)
    kept = trace_subset.write_subset(path, out, 1, CAP, 5)
    assert [p.name.split("_", 1)[1][:-5] for p in sorted(out.glob("*.json"))] == kept


def test_written_sessions_are_verbatim(traces, tmp_path):
    path, _fits, _ = traces
    source = {json.loads(line)["id"]: line.rstrip("\n") for line in path.open()}
    out = tmp_path / "out"
    for session in trace_subset.write_subset(path, out, 5, CAP, 8):
        (written,) = out.glob(f"*_{session}.json")
        assert written.read_text() == source[session]


# aiperf (agentx-harness) is what replays the subset. These run where it is
# importable, as in the ATOM image's agentx venv.


def test_peak_context_matches_aiperf():
    weka_trace = pytest.importorskip("aiperf.dataset.loader.weka_trace")
    from aiperf.dataset.loader.weka_trace_models import WekaTrace

    fits, too_long = _corpus()
    for row in fits + too_long:
        assert trace_subset.trace_peak_context(
            row
        ) == weka_trace._trace_peak_context_length(WekaTrace.model_validate(row))


def test_aiperf_loads_the_subset_and_keeps_it_whole(traces, tmp_path):
    weka_trace = pytest.importorskip("aiperf.dataset.loader.weka_trace")
    from aiperf.dataset.loader.selection import filter_then_cap
    from aiperf.dataset.loader.weka_trace_models import WekaTrace

    path, _fits, _ = traces
    out = tmp_path / "out"
    kept = trace_subset.write_subset(path, out, 2, CAP, 12)
    assert weka_trace.WekaTraceLoader.can_load(filename=out)
    loaded = [
        WekaTrace.model_validate_json(p.read_bytes())
        for p in sorted(out.glob("*.json"))
    ]
    assert [t.id for t in loaded] == kept
    # aiperf's own --max-context-length filter at the same cap drops nothing.
    pairs, _stats = filter_then_cap(
        ((t.id, weka_trace._trace_peak_context_length(t)) for t in loaded),
        num_dataset_entries=None,
        max_context_length=CAP,
    )
    assert pairs == kept
