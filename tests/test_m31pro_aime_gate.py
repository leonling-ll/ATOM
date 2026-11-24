# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Pro AIME gate: its server launch, its client against a stub server, its verdict."""

import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import ClassVar

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "m31pro"
GATE = SCRIPTS / "aime_gate.sh"
CLIENT = SCRIPTS / "aime_client.py"
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _write_run(path, matches, expected=None, samples=4):
    """A run of len(matches) problems, matches[i] of its `samples` replies right."""
    path.mkdir(parents=True)
    recs = [
        {
            "id": str(i),
            "sample": k,
            "target": "70",
            "pred": "70" if k < right else "7",
            "match": int(k < right),
            "finish": "stop",
            "thinking_closed": True,
            "completion_tokens": 5000 + k,
        }
        for i, right in enumerate(matches)
        for k in range(samples)
    ]
    (path / "replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    meta = {
        "dataset": "math-ai/aime25",
        "problems": len(matches),
        "samples": samples,
        "expected": expected or len(recs),
        "max_tokens": 98304,
        "concurrency": 32,
        "wall_s": 3000.0,
    }
    (path / "run.json").write_text(json.dumps(meta))


def _check(path, *extra):
    return subprocess.run(
        ["bash", str(GATE), "--check", str(path), *extra],
        check=False,
        capture_output=True,
        text=True,
    )


@needs_bash
# 30 problems x 4 samples. 27 all right and 3 at 2 of 4: avg 0.95, stderr 0.0167.
# 24 all right and 6 at 2 of 4: avg 0.90, stderr 0.0236, passes the 0.90 bar.
# 21 all right, 6 at 2 of 4, 3 none: avg 0.80, stderr 0.0236, fails.
@pytest.mark.parametrize(
    "matches, code",
    [
        ([4] * 27 + [2] * 3, 0),
        ([4] * 24 + [2] * 6, 0),
        ([4] * 21 + [2] * 6 + [0] * 3, 1),
    ],
)
def test_gate_judges_avg_plus_stderr(tmp_path, matches, code):
    _write_run(tmp_path / "aime", matches)
    done = _check(tmp_path / "aime")
    assert done.returncode == code, done.stdout + done.stderr
    assert ("PASS" if code == 0 else "FAIL") in done.stdout
    assert "Output tokens: mean 5002" in done.stdout, done.stdout
    assert "Wall time: 3000 s" in done.stdout, done.stdout


@needs_bash
def test_gate_takes_the_bar_from_the_caller(tmp_path):
    _write_run(tmp_path / "aime", [4] * 27 + [2] * 3)
    env = dict(os.environ, AIME_THRESHOLD="0.97")
    done = subprocess.run(
        ["bash", str(GATE), "--check", str(tmp_path / "aime")],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1, done.stdout + done.stderr
    assert "vs bar 0.97" in done.stdout


@needs_bash
def test_gate_fails_on_missing_replies(tmp_path):
    _write_run(tmp_path / "aime", [4] * 30, expected=121)
    done = _check(tmp_path / "aime")
    assert done.returncode == 1
    assert "1 replies missing" in done.stdout


@needs_bash
def test_gate_fails_without_a_run(tmp_path):
    assert _check(tmp_path).returncode != 0


@needs_bash
def test_gate_launches_the_aime_server(tmp_path):
    # setsid stands in for the server launch: it records its arguments and environment
    # and stays up until the stubbed stop_server kills it; curl reports it ready at once.
    stubs = tmp_path / "bin"
    stubs.mkdir()
    (stubs / "setsid").write_text(
        '#!/bin/bash\necho "variants=$ATOM_M3PRO_VARIANTS"\n'
        'printf "launch:"; printf " %s" "$@"; echo\nexec sleep 30\n'
    )
    (stubs / "curl").write_text("#!/bin/bash\nexit 0\n")
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    log = tmp_path / "out" / "aime" / "server.log"
    script = (
        f'source "{GATE}"\n'
        "gpus_free() { :; }\n"
        "run_aime() { :; }\n"
        "check_aime() { :; }\n"
        "stop_server() {\n"
        f'  for _ in $(seq 200); do [ -s "{log}" ] && break; sleep 0.05; done\n'
        '  [ -z "$SERVER_PID" ] || kill "$SERVER_PID"; SERVER_PID=""\n'
        "}\n"
        "aime_main --extra-flag\n"
    )
    env = dict(
        os.environ,
        PATH=f"{stubs}{os.pathsep}{os.environ['PATH']}",
        HOME=str(tmp_path),
        M31PRO_OUT=str(tmp_path / "out"),
        GATE_VARIANTS='{"index": "fused"}',
        AIME_CHAT_TEMPLATE="/t/chat_template.jinja",
    )
    done = subprocess.run(
        ["bash", "-c", script], env=env, check=False, capture_output=True, text=True
    )
    assert done.returncode == 0, done.stdout + done.stderr
    launch = log.read_text()
    assert 'variants={"index": "fused"}\n' in launch, launch
    tail = (
        " --max-model-len 131072 --max-num-seqs 32 --chat-template /t/chat_template.jinja"
        ' --default-chat-template-kwargs {"thinking_mode": "enabled"} --extra-flag'
    )
    assert launch.rstrip().endswith(tail), launch


_STOP = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
_LENGTH = {"choices": [{"delta": {}, "finish_reason": "length"}]}


def _delta(**delta):
    return {"choices": [{"delta": delta}]}


class _StubChat(http.server.BaseHTTPRequestHandler):
    """Streams the reply `replies` holds for problem "Problem <key>."; an int reply is
    an HTTP error status."""

    requests: ClassVar[list] = []
    replies: ClassVar[dict] = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).requests.append(body)
        prompt = body["messages"][0]["content"]
        reply = next(v for k, v in self.replies.items() if f"Problem {k}.\n" in prompt)
        if isinstance(reply, int):
            self.send_error(reply)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for c in [*reply, {"choices": [], "usage": {"completion_tokens": 1234}}]:
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


def _run_client(tmp_path, replies, samples=1):
    """Runs the client on one problem a key of `replies` (answer 70) against the stub."""
    problems = tmp_path / "aime.jsonl"
    rows = [{"id": k, "problem": f"Problem {k}.", "answer": "70"} for k in replies]
    problems.write_text("".join(json.dumps(r) + "\n" for r in rows))
    _StubChat.requests, _StubChat.replies = [], replies
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StubChat)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cmd = [sys.executable, str(CLIENT), "run", str(tmp_path / "run")]
        cmd += ["--port", str(server.server_address[1]), "--model", "m"]
        cmd += ["--dataset", str(problems), "--samples", str(samples)]
        done = subprocess.run(cmd, check=False, capture_output=True, text=True)
    finally:
        server.shutdown()
    lines = (tmp_path / "run" / "replies.jsonl").read_text().splitlines()
    recs = {r["id"]: r for r in map(json.loads, lines)}
    judged = subprocess.run(
        [sys.executable, str(CLIENT), "check", str(tmp_path / "run")]
        + ["--threshold", "0.5"],
        check=False,
        capture_output=True,
        text=True,
    )
    return done, recs, judged


def test_client_scores_the_content_after_thinking(tmp_path):
    thinking = _delta(reasoning_content="Let me think. \\boxed{70}")
    replies = {
        "right": [thinking, _delta(content="So \\boxed{70}."), _STOP],
        "wrong": [thinking, _delta(content="So \\boxed{1}."), _STOP],
    }
    done, recs, judged = _run_client(tmp_path, replies, samples=2)
    assert done.returncode == 0, done.stdout + done.stderr
    sent = next(r for r in _StubChat.requests if "right" in r["messages"][0]["content"])
    assert sent["temperature"] == 1.0 and sent["top_p"] == 0.95, sent
    assert sent["max_tokens"] == 98304, sent
    assert sent["messages"][0]["content"].endswith("Problem: Problem right.\nAnswer:")
    assert (recs["right"]["pred"], recs["right"]["match"]) == ("70", 1)
    assert (recs["wrong"]["pred"], recs["wrong"]["match"]) == ("1", 0)
    assert recs["right"]["completion_tokens"] == 1234
    assert recs["right"]["thinking_closed"]
    assert judged.returncode == 0, judged.stdout + judged.stderr
    assert "avg@2 0.5000 +- 0.0000" in judged.stdout, judged.stdout


@pytest.mark.parametrize(
    "answer",
    ["\\boxed{ 70 }", "\\boxed{070}", "\\boxed{70.}", "\\boxed 70$", "\\fbox{70}"],
)
def test_client_normalizes_the_answer(tmp_path, answer):
    replies = {"p": [_delta(reasoning_content="..."), _delta(content=answer), _STOP]}
    _, recs, _ = _run_client(tmp_path, replies)
    assert (recs["p"]["pred"], recs["p"]["match"]) == ("70", 1), recs["p"]


def test_client_scores_unfinished_replies_zero(tmp_path):
    # The right box in each, but the first hit the cap, the second's stream ended
    # without a finish, and the third (no reasoning split) was cut inside its thinking.
    replies = {
        "capped": [
            _delta(reasoning_content="..."),
            _delta(content="\\boxed{70}"),
            _LENGTH,
        ],
        "cut": [_delta(reasoning_content="..."), _delta(content="\\boxed{70}")],
        "think": [_delta(content="Thinking: \\boxed{70}, let me check"), _LENGTH],
    }
    done, recs, judged = _run_client(tmp_path, replies)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [recs[k]["match"] for k in replies] == [0, 0, 0], recs
    assert not recs["think"]["thinking_closed"]
    assert "2 at the 98304 cap, 1 without a finish" in judged.stdout, judged.stdout
    assert "FAIL" in judged.stdout and judged.returncode == 1


def test_client_reports_a_failed_request_as_missing(tmp_path):
    replies = {
        "ok": [_delta(reasoning_content="..."), _delta(content="\\boxed{70}"), _STOP],
        "lost": 500,
    }
    done, recs, judged = _run_client(tmp_path, replies)
    assert done.returncode == 1
    assert "request failed" in done.stderr, done.stderr
    assert list(recs) == ["ok"]
    assert "1/2 replies" in judged.stdout, judged.stdout
    assert "FAIL: 1 replies missing" in judged.stdout and judged.returncode == 1
