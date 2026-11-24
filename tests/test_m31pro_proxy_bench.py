# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Pro proxy benchmark's acceptance length and launch, from its dry run."""

import itertools
import json
import os
import shlex
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "scripts" / "m31pro" / "proxy_bench.sh"
AGENTIC = ROOT / "recipes" / "Agentic-MiniMax-M3.1-Pro.md"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


_CLEARED = (
    "ACCEPT_LENGTH",
    "SPEC_TOKENS",
    "THINKING",
    "RECIPE",
    "GPU_MEM_UTIL",
    "MAX_NUM_SEQS",
    "SERVER_EXTRA",
    "PYTHONHASHSEED",
    "CHAT_TEMPLATE",
    "MODEL",
    "PREWARM",
    "PREWARM_PROMPT_TOKENS",
    "PREWARM_BATCHES",
    "SERVER_LOG",
    "PORT",
)


def _clean_env():
    return {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ATOM_", "AITER_", "GPTOSS_", "LMCACHE_"))
        and k not in _CLEARED
    }


def _dry_run(tmp_path, **env):
    clean = _clean_env()
    clean.update(env, PROXY_DRY_RUN="1", HOME=str(tmp_path))
    done = subprocess.run(
        ["bash", str(BENCH), "32", str(tmp_path / "out")],
        env=clean,
        check=False,
        capture_output=True,
        text=True,
    )
    lines = done.stdout.splitlines()
    values = dict(line.split("=", 1) for line in lines if "=" in line[:40])
    args = next((line for line in lines if line.startswith("server_args:")), "")
    return done, values, args


# Pro's thinking_on golden curve for the bundled DSpark draft.
@pytest.mark.parametrize(
    "k, al",
    [("1", "1.75"), ("2", "2.23"), ("3", "2.53"), ("4", "2.71")]
    + [("5", "2.89"), ("6", "2.94"), ("7", "2.94")],
)
def test_acceptance_is_the_golden_al_of_the_draft_length(tmp_path, k, al):
    done, values, args = _dry_run(tmp_path, SPEC_TOKENS=k)
    assert done.returncode == 0, done.stdout + done.stderr
    assert values["acceptance_length"] == al
    assert f"--num-speculative-tokens {k} --spec-decode-acceptance-length {al}" in args


def test_default_is_k7_at_golden_294(tmp_path):
    done, _, args = _dry_run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert "--num-speculative-tokens 7 --spec-decode-acceptance-length 2.94" in args


@pytest.mark.parametrize(
    "env", [{"SPEC_TOKENS": "8"}, {"THINKING": "thinking_off"}, {"RECIPE": "vllm"}]
)
def test_refuses_what_it_has_no_value_for(tmp_path, env):
    done, _, args = _dry_run(tmp_path, **env)
    assert done.returncode != 0
    assert not args
    assert not (tmp_path / "out").exists()


def test_acceptance_length_override(tmp_path):
    done, _, args = _dry_run(
        tmp_path, SPEC_TOKENS="8", THINKING="thinking_off", ACCEPT_LENGTH="4.0"
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "--num-speculative-tokens 8 --spec-decode-acceptance-length 4.0" in args


def test_no_draft_serves_without_spec_flags(tmp_path):
    done, values, args = _dry_run(tmp_path, SPEC_TOKENS="0", THINKING="thinking_off")
    assert done.returncode == 0, done.stdout + done.stderr
    assert values["acceptance_length"] == "none"
    assert "--method" not in args


def test_agentic_recipe_launch(tmp_path):
    done, values, _ = _dry_run(tmp_path, RECIPE="agentic")
    assert done.returncode == 0, done.stdout + done.stderr
    assert values["GPU_MEM_UTIL"] == "0.95"
    assert values["ATOM_KV_OFFLOAD"] == "lmcache"
    assert values["LMCACHE_MAX_LOCAL_CPU_SIZE"] == "184"
    assert values["ATOM_ONLINE_QUANT_STREAMING"] == "1"
    assert values["ATOM_CACHE_AWARE_ADMISSION"] == "1"
    assert '"ar_lamport_rows": 16' in values["ATOM_M3PRO_VARIANTS"]
    assert "AITER_CUSTOM_AR_DISABLE_LAMPORT" not in values
    assert values["AITER_QUICK_REDUCE_QUANTIZATION"] == "NONE"
    assert values["ATOM_WARMUP_TOKENS"] == "auto"
    assert values["LMCACHE_CHUNK_SIZE"] == "256"
    assert values["LMCACHE_LOCAL_CPU"] == "True"
    assert values["PYTHONHASHSEED"] == "0"
    assert values["ATOM_CUSTOM_AR_POOL_BYTES"] == "134217728"


def test_pro_recipe_launch(tmp_path):
    done, values, _ = _dry_run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert values["GPU_MEM_UTIL"] == "0.94"
    assert values["AITER_CUSTOM_AR_DISABLE_LAMPORT"] == "1"
    assert values["AITER_QUICK_REDUCE_QUANTIZATION"] == "NONE"
    assert "ATOM_KV_OFFLOAD" not in values
    assert values["ATOM_CUSTOM_AR_POOL_BYTES"] == "134217728"


AGENTIC_LADDER = (
    "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,"
    "48,64,128,256,512]"
)


def _ladders(args):
    """Every --cudagraph-capture-sizes value in the dry run's server flags."""
    words = args.replace("\\", "").split()
    pairs = itertools.pairwise(words)
    return [b for a, b in pairs if a == "--cudagraph-capture-sizes"]


def test_agentic_launch_captures_graphs_between_8_16_and_32(tmp_path):
    done, _, args = _dry_run(tmp_path, RECIPE="agentic")
    assert done.returncode == 0, done.stdout + done.stderr
    assert _ladders(args) == [AGENTIC_LADDER]


def test_pro_launch_keeps_the_default_ladder(tmp_path):
    done, _, args = _dry_run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert _ladders(args) == []


def test_server_extra_ladder_comes_after_the_recipe_ladder(tmp_path):
    # the server's argument parser keeps the last value, so SERVER_EXTRA wins
    done, _, args = _dry_run(
        tmp_path, RECIPE="agentic", SERVER_EXTRA="--cudagraph-capture-sizes [1,2]"
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert _ladders(args) == [AGENTIC_LADDER, "[1,2]"]


# Flags proxy_bench sets from its own arguments: the checkpoint, TP, port and
# MAX_NUM_SEQS (2 x CONC in both).
RUN_FLAGS = ("--model", "--tensor-parallel-size", "--server-port", "--max-num-seqs")


def _flags(words):
    """{flag: value or None} of a server command line; the last value wins."""
    flags = {}
    for i, word in enumerate(words):
        if word.startswith("--"):
            nxt = words[i + 1] if i + 1 < len(words) else "--"
            flags[word] = None if nxt.startswith("--") else nxt
    return flags


def _recipe_launch():
    """Exports, unsets and server flags of the Agentic recipe's server launch."""
    text = AGENTIC.read_text(encoding="utf-8")
    section = text.split("\n## 1. Start the ATOM server\n", 1)[1]
    block = section.split("```bash\n", 1)[1].split("\n```", 1)[0]
    env, unset = {}, set()
    for line in block.splitlines():
        if not line.startswith(("export ", "unset ")):
            continue
        words = shlex.split(line, comments=True)
        if words[0] == "unset":
            unset.update(words[1:])
        elif "${" not in line:
            name, value = words[1].split("=", 1)
            env[name] = value
    command = block.split("python3 -m atom.entrypoints.openai_server", 1)[1]
    command = command.split("2>&1", 1)[0].replace("\\\n", " ")
    command = command.replace(
        '${CHAT_TEMPLATE:+--chat-template "${CHAT_TEMPLATE}"}',
        '--chat-template "${CHAT_TEMPLATE}"',
    )
    return env, unset, _flags(shlex.split(command))


def test_agentic_launch_is_the_recipe_launch(tmp_path):
    template = tmp_path / "chat_template.jinja"
    template.write_text("{% if thinking_mode %}{% endif %}")
    done, values, args = _dry_run(
        tmp_path, RECIPE="agentic", CHAT_TEMPLATE=str(template)
    )
    assert done.returncode == 0, done.stdout + done.stderr
    env, unset, flags = _recipe_launch()

    variants = env.pop("ATOM_M3PRO_VARIANTS")
    served = json.loads(values["ATOM_M3PRO_VARIANTS"])
    for key, value in json.loads(variants).items():
        assert served.get(key) == value, key
    for name, value in env.items():
        assert values.get(name) == value, name
    for name in unset:
        assert name not in values, name

    # start_atom_server.sh turns these three variables into flags.
    served = _flags(shlex.split(args.removeprefix("server_args:")))
    served["--kv_cache_dtype"] = values["KV_CACHE_DTYPE"]
    served["--gpu-memory-utilization"] = values["GPU_MEM_UTIL"]
    for flag, value in flags.items():
        if flag in RUN_FLAGS:
            continue
        if value == "${CHAT_TEMPLATE}":
            value = str(template)
        assert flag in served, flag
        assert served[flag] == value, flag


def test_agentic_variants_add_onto_the_recipe(tmp_path):
    extra = {"index": "fused", "ar_fused_rows": 128, "index_compress": False}
    done, values, _ = _dry_run(
        tmp_path, RECIPE="agentic", ATOM_M3PRO_VARIANTS=json.dumps(extra)
    )
    assert done.returncode == 0, done.stdout + done.stderr
    served = json.loads(values["ATOM_M3PRO_VARIANTS"])
    assert served == {**json.loads(_recipe_launch()[0]["ATOM_M3PRO_VARIANTS"]), **extra}


def test_pro_variants_replace(tmp_path):
    done, values, _ = _dry_run(tmp_path, ATOM_M3PRO_VARIANTS='{"kernels": false}')
    assert done.returncode == 0, done.stdout + done.stderr
    assert values["ATOM_M3PRO_VARIANTS"] == '{"kernels": false}'


@pytest.mark.parametrize(
    "template, ok",
    [
        ("{{ messages[-1].content }}", False),
        ("{% if thinking_mode %}{% endif %}", True),
    ],
)
def test_agentic_refuses_the_stub_template(tmp_path, template, ok):
    (tmp_path / "chat_template.jinja").write_text(template)
    done, _, args = _dry_run(tmp_path, RECIPE="agentic", MODEL=str(tmp_path))
    assert (done.returncode == 0) == ok, done.stdout + done.stderr
    assert "--chat-template" not in args
    if not ok:
        assert "CHAT_TEMPLATE" in done.stderr


class _Reply(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        with self.server.lock, self.server.events.open("a") as events:
            events.write(f"request max_tokens={body['max_tokens']}\n")
        if self.server.fail:
            self.send_error(500)
            return
        payload = json.dumps({"usage": {"completion_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def _full_run(tmp_path, fail=False, **env):
    """Run proxy_bench.sh with stubbed server scripts, trace subset and client,
    against a recording HTTP server; return the run and its ordered events."""
    tmp_path.mkdir(exist_ok=True)
    tree = tmp_path / "tree"
    (tree / "scripts" / "m31pro").mkdir(parents=True)
    shutil.copy(BENCH, tree / "scripts" / "m31pro" / "proxy_bench.sh")
    events = tmp_path / "events"
    events.touch()
    for name in ("start_atom_server", "stop_atom_server", "wait_server_ready"):
        (tree / "scripts" / f"{name}.sh").write_text(f"echo {name} >> {events}\n")
    (tree / "scripts" / "m31pro" / "trace_subset.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "out = Path(sys.argv[sys.argv.index('--out') + 1])\n"
        "out.mkdir(parents=True)\n"
        "(out / 't.json').write_text('{}')\n"
    )
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "aiperf").write_text(f"#!/bin/bash\necho aiperf >> {events}\n")
    real_python = shutil.which("python3") or shutil.which("python")
    (stubs / "python3").write_text(
        '#!/bin/bash\ncase "$*" in *"import atom"*) exit 0 ;; esac\n'
        f'exec "{real_python}" "$@"\n'
    )
    for stub in stubs.iterdir():
        stub.chmod(0o755)

    server = ThreadingHTTPServer(("127.0.0.1", 0), _Reply)
    server.lock, server.events, server.fail = threading.Lock(), events, fail
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        clean = _clean_env()
        clean.update(
            env,
            HOME=str(tmp_path),
            PATH=f"{stubs}{os.pathsep}{os.environ['PATH']}",
            PORT=str(server.server_port),
            SERVER_LOG=str(tmp_path / "server.log"),
            MODEL=str(tmp_path),
        )
        done = subprocess.run(
            ["bash", str(tree / "scripts" / "m31pro" / "proxy_bench.sh"), "32"],
            env=clean,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
    finally:
        server.shutdown()
    return done, events.read_text().splitlines()


PREWARM_PLAN = {"PREWARM_PROMPT_TOKENS": "8,16", "PREWARM_BATCHES": "1,2"}
NO_PREWARM = ["start_atom_server", "wait_server_ready", "aiperf", "stop_atom_server"]


def test_prewarm_runs_after_the_server_is_ready_and_before_the_client(tmp_path):
    done, events = _full_run(tmp_path, RECIPE="agentic", **PREWARM_PLAN)
    assert "prewarm done" in done.stdout, done.stdout + done.stderr
    # two prefill prompts (1 token), then 1 + 2 decode prompts (64 tokens)
    prewarm = ["request max_tokens=1"] * 2 + ["request max_tokens=64"] * 3
    assert events[:2] == ["start_atom_server", "wait_server_ready"]
    assert events[2:7] == prewarm
    assert events[7] == "aiperf"


def test_prewarm_is_skipped_with_prewarm_0(tmp_path):
    done, events = _full_run(tmp_path, RECIPE="agentic", PREWARM="0", **PREWARM_PLAN)
    assert "prewarm done" not in done.stdout
    assert events == NO_PREWARM


def test_prewarm_is_off_for_the_pro_recipe_unless_asked(tmp_path):
    _, events = _full_run(tmp_path / "pro", **PREWARM_PLAN)
    assert events == NO_PREWARM
    _, events = _full_run(tmp_path / "on", PREWARM="1", **PREWARM_PLAN)
    assert events.count("request max_tokens=64") == 3


def test_prewarm_plan_in_the_dry_run(tmp_path):
    done, values, _ = _dry_run(tmp_path, RECIPE="agentic")
    assert done.returncode == 0, done.stdout + done.stderr
    assert values["prewarm"] == "1"
    prompts = [int(n) for n in values["prewarm_prompt_tokens"].split(",")]
    assert max(prompts) > 16384  # beyond one prefill chunk
    batches = [int(n) for n in values["prewarm_batches"].split(",")]
    # every graph a CONC 32 run can replay (MAX_NUM_SEQS 64)
    assert batches == [n for n in json.loads(AGENTIC_LADDER) if n <= 64]
    done, values, _ = _dry_run(tmp_path, RECIPE="agentic", PREWARM="0")
    assert values["prewarm"] == "0"
    assert "prewarm_batches" not in values


def test_a_failed_prewarm_reply_stops_the_run(tmp_path):
    done, events = _full_run(tmp_path, fail=True, RECIPE="agentic", **PREWARM_PLAN)
    assert done.returncode != 0
    assert "aiperf" not in events
    assert events[-1] == "stop_atom_server"
