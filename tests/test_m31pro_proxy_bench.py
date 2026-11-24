# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Pro proxy benchmark's acceptance length and launch, from its dry run."""

import itertools
import os
import shutil
import subprocess
from pathlib import Path

import pytest

BENCH = Path(__file__).resolve().parents[1] / "scripts" / "m31pro" / "proxy_bench.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _dry_run(tmp_path, **env):
    clean = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("ATOM_", "AITER_", "LMCACHE_"))
        and k
        not in (
            "ACCEPT_LENGTH",
            "SPEC_TOKENS",
            "THINKING",
            "RECIPE",
            "GPU_MEM_UTIL",
            "MAX_NUM_SEQS",
            "SERVER_EXTRA",
            "PYTHONHASHSEED",
        )
    }
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
    # the recipe leaves the all-reduce pool at aiter's default
    assert "ATOM_CUSTOM_AR_POOL_BYTES" not in values


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
