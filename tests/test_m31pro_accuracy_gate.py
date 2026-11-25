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


def _stop_server_stub(tmp_path):
    # Stops the stub launch once it has written its output, for at most 10 s: the
    # stubbed curl reports ready at once, so an immediate kill could beat the write.
    log = tmp_path / "out" / "gate" / "server.log"
    return (
        "stop_server() {\n"
        f'  for _ in $(seq 200); do [ -s "{log}" ] && break; sleep 0.05; done\n'
        '  kill "$SERVER_PID"; SERVER_PID=""\n'
        f'  [ -s "{log}" ] || {{ echo "stub launch wrote no output" >&2; exit 1; }}\n'
        "}\n"
    )


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
        f"{_stop_server_stub(tmp_path)}"
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


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize(
    "given, variants, lamport",
    [
        ({}, '{"kernels": true, "attn": "aiter", "topk": "fused"}', "1"),
        (
            {"GATE_VARIANTS": '{"index": "fused"}', "GATE_DISABLE_LAMPORT": "0"},
            '{"index": "fused"}',
            "0",
        ),
    ],
)
def test_gate_takes_variants_and_lamport_from_the_caller(
    tmp_path, given, variants, lamport
):
    # The stub launch prints the environment the server would start with.
    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "setsid").write_text(
        '#!/bin/bash\necho "variants=$ATOM_M3PRO_VARIANTS"\n'
        'echo "lamport=$AITER_CUSTOM_AR_DISABLE_LAMPORT"\nexec sleep 30\n'
    )
    (stubs / "curl").write_text("#!/bin/bash\nexit 0\n")
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    script = (
        f'source "{GATE}"\n'
        "gpus_free() { :; }\n"
        "run_gsm8k() { :; }\n"
        "check_gsm8k() { :; }\n"
        f"{_stop_server_stub(tmp_path)}"
        "main\n"
    )
    env = dict(
        os.environ,
        PATH=f"{stubs}{os.pathsep}{os.environ['PATH']}",
        HOME=str(tmp_path),
        M31PRO_OUT=str(tmp_path / "out"),
        **given,
    )
    done = subprocess.run(
        ["bash", "-c", script], env=env, check=False, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr
    launch = (tmp_path / "out" / "gate" / "server.log").read_text()
    assert f"variants={variants}\n" in launch, launch
    assert f"lamport={lamport}\n" in launch, launch
    assert f"ATOM_M3PRO_VARIANTS={variants} " in done.stdout, done.stdout
