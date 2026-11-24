# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Pro accuracy gate's verdict on lm-eval's gsm8k results file."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parents[1] / "scripts" / "m31pro" / "accuracy_gate.sh"


def _verdict(tmp_path, flexible):
    run = tmp_path / "gsm8k" / "__data__models__m3-pro"
    run.mkdir(parents=True)
    metrics = {"flexible-extract": flexible, "strict-match": flexible - 0.002}
    results = {f"exact_match,{k}": v for k, v in metrics.items()}
    results.update({f"exact_match_stderr,{k}": 0.0065 for k in metrics})
    doc = {"results": {"gsm8k": results}, "n-samples": {"gsm8k": {"effective": 1319}}}
    (run / "results_2026-11-22T00-00-00.json").write_text(json.dumps(doc))
    return subprocess.run(
        ["bash", str(GATE), "--check", str(tmp_path / "gsm8k")],
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
# stderr is 0.0065: a score passes when within one stderr of the 0.94 bar (0.9335).
@pytest.mark.parametrize(
    "flexible, code", [(0.9409, 0), (0.9393, 0), (0.9336, 0), (0.9334, 1)]
)
def test_gate_judges_flexible_extract_plus_stderr(tmp_path, flexible, code):
    done = _verdict(tmp_path, flexible)
    assert done.returncode == code, done.stdout + done.stderr
    assert ("PASS" if code == 0 else "FAIL") in done.stdout


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_gate_fails_without_results(tmp_path):
    done = subprocess.run(
        ["bash", str(GATE), "--check", str(tmp_path)], check=False, text=True
    )
    assert done.returncode != 0


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_gate_passes_its_arguments_to_the_server(tmp_path):
    # setsid stands in for the server launch: it records its arguments and stays up
    # until the stubbed stop_server kills it; curl reports the server ready at once.
    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "setsid").write_text('#!/bin/bash\necho "launch: $*"\nexec sleep 30\n')
    (stubs / "curl").write_text("#!/bin/bash\nexit 0\n")
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    script = (
        f'source "{GATE}"\n'
        "gpus_free() { :; }\n"
        "run_gsm8k() { :; }\n"
        "check_gsm8k() { :; }\n"
        'stop_server() { [ -z "$SERVER_PID" ] || kill "$SERVER_PID"; SERVER_PID=""; }\n'
        "main --max-num-seqs 32 --extra-flag\n"
    )
    env = dict(
        os.environ,
        PATH=f"{stubs}{os.pathsep}{os.environ['PATH']}",
        HOME=str(tmp_path),
        M31PRO_OUT=str(tmp_path / "out"),
    )
    done = subprocess.run(
        ["bash", "-c", script], env=env, check=False, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr
    launch = (tmp_path / "out" / "gate" / "server.log").read_text()
    assert launch.rstrip().endswith("--max-num-seqs 32 --extra-flag"), launch
