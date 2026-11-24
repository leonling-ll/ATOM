# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the MiniMax-M3.1-Pro proxy tail analysis."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/m31pro/proxy_tail.py"
SPEC = importlib.util.spec_from_file_location("m31pro_proxy_tail", SCRIPT)
proxy_tail = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proxy_tail)
from interactivity import agentic_interactivity  # on the path the script set


def _record(rid, start_s, ttft_ms, gaps_ms, phase="profiling", tokens_per_chunk=2):
    osl = 1 + tokens_per_chunk * len(gaps_ms)
    decode_ms = sum(gaps_ms)
    metrics = {
        "time_to_first_token": {"value": ttft_ms},
        "output_sequence_length": {"value": osl},
        "input_sequence_length": {"value": 1000},
        "request_latency": {"value": ttft_ms + decode_ms},
        "inter_chunk_latency": {"value": gaps_ms},
    }
    if osl > 1:
        metrics["inter_token_latency"] = {"value": decode_ms / (osl - 1)}
    end_s = start_s + (ttft_ms + decode_ms) / 1e3
    return {
        "metadata": {
            "x_request_id": rid,
            "benchmark_phase": phase,
            "request_start_ns": int(start_s * 1e9),
            "request_end_ns": int(end_s * 1e9),
        },
        "metrics": metrics,
    }


def _write(tmp_path, records):
    path = tmp_path / "arm" / "c2" / "aiperf" / "profile_export.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def _run(path):
    return proxy_tail.analyse(path, stall_s=0.1, ttft_shave_s=0.03, gpus=8)


@pytest.fixture
def records():
    return [
        # warmup: never counted, but its prefill (100.30 .. 100.40 s) is real load
        _record("w", 100.3, 100.0, [10.0] * 5, phase="warmup"),
        # first token at 100.20 s; a 300 ms stall at 100.22 .. 100.52 s, which
        # the warmup request's prefill starts and ends inside
        _record("a", 100.0, 200.0, [10.0, 10.0, 300.0, 10.0, 10.0, 10.0]),
        # a 250 ms stall at 200.18 .. 200.43 s with nothing else in flight
        _record("b", 200.0, 150.0, [10.0, 10.0, 10.0, 230.0 + 20.0, 10.0]),
        # last chunk 8 ms late
        _record("c", 300.0, 600.0, [10.0, 10.0, 10.0, 10.0, 18.0]),
        _record("d", 400.0, 120.0, [12.0, 12.0, 12.0, 12.0, 20.0]),
    ]


def test_baseline_matches_dashboard_metric(tmp_path, records):
    path = _write(tmp_path, records)
    row = _run(path)
    assert row["point"] == "arm/c2"
    assert row["requests"] == 4
    assert row["p90_e2e_norm"] == pytest.approx(agentic_interactivity(path)["value"])


def test_stalls_and_prefill_overlap(tmp_path, records):
    row = _run(_write(tmp_path, records))
    assert row["stalls"] == 2
    assert row["stall_s"] == pytest.approx(0.55)
    assert row["stalls_beside_prefill"] == 1
    assert row["stall_beside_prefill_s"] == pytest.approx(0.3)
    assert row["chunks"] == 6 + 5 + 5 + 5


def test_final_chunk_excess(tmp_path, records):
    row = _run(_write(tmp_path, records))
    # final gaps 10, 10, 18, 20 -> 14; per-request medians 10, 10, 10, 12 -> 10
    assert row["final_gap_ms"] == pytest.approx(14.0)
    assert row["median_gap_ms"] == pytest.approx(10.0)
    assert proxy_tail.final_excess(
        {"gaps": [0.010, 0.010, 0.010, 0.010, 0.018]}
    ) == pytest.approx(0.008)
    assert proxy_tail.final_excess({"gaps": [0.010, 0.020, 0.010, 0.005]}) == 0.0


def test_whatifs_move_only_what_they_remove(tmp_path, records):
    path = _write(tmp_path, records)
    reqs, _ = proxy_tail.load(path)
    base = proxy_tail.p90_metric(reqs)
    # no shortening is the baseline
    assert proxy_tail.p90_metric(reqs, lambda r: 0.0, 0.0) == pytest.approx(base)
    row = _run(path)
    assert row["whatif_no_final_excess"] > base
    assert row["whatif_no_stalls"] > base
    assert row["whatif_ttft_shaved"] > base
    # request "a" without its stall: 290 ms off a 350 ms decode span of 12 tokens
    a = next(r for r in reqs if r["id"] == "a")
    stall_free = dict(a, itl=a["itl"] - 0.29 / (a["osl"] - 1))
    assert proxy_tail.p90_metric([a], lambda r: 0.29) == pytest.approx(
        proxy_tail.p90_metric([stall_free])
    )


def test_cli_table(tmp_path, records, capsys, monkeypatch):
    path = _write(tmp_path, records)
    monkeypatch.setattr("sys.argv", ["proxy_tail.py", str(tmp_path)])
    proxy_tail.main()
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("| point |")
    assert out[2].startswith("| arm/c2 | 4 | 14.0 / 10.0 | 2 (1) | 0.55 (0.30) |")
    assert path.is_file()


@pytest.mark.parametrize(
    "cwd_parts, arg",
    [(("aiperf",), "profile_export.jsonl"), ((), "aiperf/profile_export.jsonl")],
)
def test_cli_relative_file_path(tmp_path, records, capsys, monkeypatch, cwd_parts, arg):
    point_dir = _write(tmp_path, records).parents[1]
    monkeypatch.chdir(point_dir.joinpath(*cwd_parts))
    monkeypatch.setattr("sys.argv", ["proxy_tail.py", arg])
    proxy_tail.main()
    assert capsys.readouterr().out.splitlines()[2].startswith("| arm/c2 | 4 |")
