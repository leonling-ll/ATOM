# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The TP8 + DCP8 check script's server launches (dry run: nothing starts)."""

import ast
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "m31pro" / "dcp8_check.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _dry(step, tmp_path, **env):
    done = subprocess.run(
        ["bash", str(SCRIPT), step],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ, DRY_RUN="1", M31PRO_OUT=str(tmp_path / "out"), **env),
    )
    return done


def _launches(out):
    """Each printed launch as (its environment, its argv)."""
    launches, env = [], {}
    for line in out.splitlines():
        if line.startswith("export "):
            name, _, value = shlex.split(line[len("export ") :])[0].partition("=")
            env[name] = value
        elif line.startswith("python -m atom.entrypoints.openai_server"):
            launches.append((env, shlex.split(line)))
            env = {}
    return launches


def _flag(argv, name):
    return argv[argv.index(name) + 1]


def _variant_fields():
    tree = ast.parse((ROOT / "atom/model_ops/minimax_m3/pro_attn.py").read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ProVariants"
    )
    return {n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)}


def test_serve_is_the_conc64_launch_at_dcp8_with_dspark_and_lmcache_mp(tmp_path):
    done = _dry("serve", tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    ((env, argv),) = _launches(done.stdout)
    assert "unset AITER_CUSTOM_AR_DISABLE_LAMPORT" in done.stdout
    assert env["ATOM_KV_OFFLOAD"] == "lmcache_mp"
    assert env["LMCACHE_CHUNK_SIZE"] == "256"
    for flag, value in {
        "--tensor-parallel-size": "8",
        "--decode-context-parallel-size": "8",
        "--max-model-len": "1048576",
        "--max-num-seqs": "128",
        "--method": "dspark",
        "--num-speculative-tokens": "5",
        "--spec-decode-acceptance-length": "2.89",
        "--block-size": "16",
    }.items():
        assert _flag(argv, flag) == value, flag
    variants = json.loads(env["ATOM_M3PRO_VARIANTS"])
    assert variants["dcp_topk_merge"] is True and variants["index_compress"] is True
    assert set(variants) <= _variant_fields()


def test_in_process_lmcache_takes_the_c64_cpu_tier(tmp_path):
    done = _dry("serve", tmp_path, OFFLOAD="lmcache")
    ((env, _),) = _launches(done.stdout)
    assert env["ATOM_KV_OFFLOAD"] == "lmcache"
    assert env["LMCACHE_LOCAL_CPU"] == "True"
    assert env["LMCACHE_MAX_LOCAL_CPU_SIZE"] == "280"


def test_restore_without_mp_turns_hbm_prefix_caching_off(tmp_path):
    done = _dry("restore", tmp_path, OFFLOAD="lmcache")
    ((_, argv),) = _launches(done.stdout)
    assert "--no-enable_prefix_caching" in argv
    assert "--spec-decode-acceptance-length" not in argv  # real acceptance
    assert not (tmp_path / "out").exists()


def test_ab_launches_differ_only_in_dcp_topk_merge(tmp_path):
    done = _dry("ab", tmp_path)
    (env_a, argv_a), (env_b, argv_b) = _launches(done.stdout)
    assert argv_a == argv_b
    a = json.loads(env_a.pop("ATOM_M3PRO_VARIANTS"))
    b = json.loads(env_b.pop("ATOM_M3PRO_VARIANTS"))
    assert env_a == env_b
    assert (a.pop("dcp_topk_merge"), b.pop("dcp_topk_merge")) == (True, False)
    assert a == b


def test_ab_runs_dcp_variants_against_ab_variants(tmp_path):
    ((env, _),) = _launches(_dry("serve", tmp_path).stdout)
    c64 = json.loads(env["ATOM_M3PRO_VARIANTS"])
    fold = json.dumps(dict(c64, dcp_fold=True))
    done = _dry("ab", tmp_path, DCP_VARIANTS=fold, AB_VARIANTS=json.dumps(c64))
    assert done.returncode == 0, done.stdout + done.stderr
    (env_a, _), (env_b, _) = _launches(done.stdout)
    a = json.loads(env_a["ATOM_M3PRO_VARIANTS"])
    b = json.loads(env_b["ATOM_M3PRO_VARIANTS"])
    assert a.pop("dcp_fold") is True and a == b


def test_ab_refuses_arms_of_the_same_variants(tmp_path):
    done = _dry("ab", tmp_path, DCP_VARIANTS='{"kernels": true}')
    assert done.returncode == 1 and "set AB_VARIANTS" in done.stderr


def test_an_unknown_offload_mode_is_refused(tmp_path):
    done = _dry("serve", tmp_path, OFFLOAD="bogus")
    assert done.returncode == 1
    assert "OFFLOAD must be" in done.stderr
