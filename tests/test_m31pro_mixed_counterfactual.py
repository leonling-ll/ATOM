# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the mixed prefill+decode counterfactual
(``scripts/m31pro/mixed_counterfactual.py``): a torch-profiler trace of decode and
prefill forwards, labelled the way ATOM labels them, read through
``kernel_roofline.read_forwards`` and replayed with each prefill absorbing the decode
step after it."""

import gzip
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "scripts/m31pro"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mc = _load("mixed_counterfactual")
kr = _load("kernel_roofline")

SEQS = 4
MXPSH = "flydsl_mxpsh_16x32x512_F8_F8_B16_w2_x0_sk1_slrr_pf5_nt"
# a decode step: 90 us of kernels then 10 us of launch gap, so a 100 us wall
DECODE_TARGET = (
    ("_prep_kernel", 10.0),
    ("_score_tile_kernel", 10.0),
    ("_attn_kernel", 10.0),
    ("gemm1_a16w4_port_a16w4_h7168_i640_ne384_bm16_tn64_swiglu", 20.0),
    (MXPSH, 30.0),
)
DECODE_DRAFT = (("_markov_stage1", 5.0),)
DECODE_POST = (("topk_per_row_argmax_fold1", 5.0),)
G = {"attn": 40.0, "attn+moe": 60.0, "full": 90.0}
# P2 and P5 run their routed experts on MXFP4 activations, P8 on bf16 ones (as a
# decode step does), so only P8 shares its expert call with the decode rows
G_PAIRS = {k: [v] * 3 for k, v in G.items()} | {"format": [60.0, 60.0, 40.0]}
PREFILL_FP4 = "gemm1_a4w4_port_fp4_h7168_i640_bm128"
DECODE_WALL, PREFILL_WALL = 100.0, 510.0


class _Trace:
    """CPU annotation ranges, launches and their GPU kernels, back to back."""

    def __init__(self):
        self.ev, self.cpu, self.gpu, self.corr = [], 0.0, 0.0, 0

    def span(self, name, t0, t1):
        self.ev.append(
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": name,
                "ts": t0,
                "dur": t1 - t0,
                "pid": 1,
                "tid": 1,
            }
        )

    def launch(self, name, dur):
        self.corr += 1
        self.cpu += 1.0
        for cat, ts, pid, tid, d in (
            ("cuda_runtime", self.cpu, 1, 1, 0.5),
            ("kernel", self.gpu, 0, 7, dur),
        ):
            self.ev.append(
                {
                    "ph": "X",
                    "cat": cat,
                    "name": "hipLaunchKernel" if cat == "cuda_runtime" else name,
                    "ts": ts,
                    "dur": d,
                    "pid": pid,
                    "tid": tid,
                    "args": {"correlation": self.corr},
                }
            )
        self.gpu += dur

    def forward(self, prefill, idle_after=10.0, fp4=True):
        rows = 1000 if prefill else SEQS * 6
        f0 = self.cpu
        self.cpu += 1
        r0 = self.cpu
        routed = PREFILL_FP4 if fp4 else DECODE_TARGET[3][0]
        prefill_kernels = ((MXPSH, 400.0), (routed, 100.0))
        for name, d in prefill_kernels if prefill else DECODE_TARGET:
            self.launch(name, d)
        run = f"prefill[bs=1 tok={rows}]" if prefill else f"decode[bs={SEQS} spec=5]"
        self.span(run, r0, self.cpu + 0.5)
        self.span("ATOM::run_model", r0 - 0.5, self.cpu + 1)
        if not prefill:
            self.cpu += 2
            d0 = self.cpu
            for name, d in DECODE_DRAFT:
                self.launch(name, d)
            self.span(f"propose_dspark[bs={SEQS} graph T=5]", d0, self.cpu + 0.5)
            self.cpu += 2
            for name, d in DECODE_POST:
                self.launch(name, d)
        self.span(
            f"ATOM::forward tokens={rows} prefill={int(prefill)} "
            f"decode={0 if prefill else SEQS}",
            f0,
            self.cpu + 1,
        )
        self.cpu += 5
        self.gpu += idle_after

    def write(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as f:
            json.dump({"traceEvents": self.ev}, f)


# D0 D1 P2 D3 P4 P5 D6 D7 (idle after) P8 D9 D10: the edges and the stalled forward are dropped,
# which leaves D1 P2 D3 P4 P5 D6 | P8 D9 with a break where D7 was
ORDER = "DDPDPPDDPDD"


@pytest.fixture
def trace(tmp_path):
    tr = _Trace()
    for i, kind in enumerate(ORDER):
        tr.forward(kind == "P", idle_after=8000.0 if i == 7 else 10.0, fp4=i != 8)
    path = tmp_path / "prof" / "rank_0" / "x.pt.trace.json.gz"
    tr.write(path)
    return path


def test_reader_keeps_prefill_only_on_request(trace):
    decode = kr.read_forwards(str(trace))
    every = kr.read_forwards(str(trace), decode_only=False)
    assert [f["prefill"] for f in decode] == [0] * 4
    assert ["P" if f["prefill"] else "D" for f in every] == list("DPDPPDPD")
    assert [f["wall"] for f in every[:6]] == [
        DECODE_WALL,
        PREFILL_WALL,
        DECODE_WALL,
        PREFILL_WALL,
        PREFILL_WALL,
        DECODE_WALL,
    ]


def test_decode_increment_sets(trace):
    dec = next(f for f in kr.read_forwards(str(trace)))
    assert mc.decode_increment(dec) == G | {"format": G["attn"]}


@pytest.mark.parametrize("which", mc.SETS)
def test_replay_absorbs_the_decode_after_a_prefill(trace, which):
    fwds = kr.read_forwards(str(trace), decode_only=False)
    today, mixed, pairs = mc.replay(fwds, which, host_us=0.0)
    # P2-D3, P5-D6 (P4 is followed by a prefill), P8-D9; D6 to P8 is a break
    assert [p["saved_us"] for p in pairs] == [DECODE_WALL - g for g in G_PAIRS[which]]
    assert sum(s[0] for s in today if s) == 4 * DECODE_WALL + 4 * PREFILL_WALL
    saved = sum(s[0] for s in today if s) - sum(s[0] for s in mixed if s)
    assert saved == pytest.approx(sum(DECODE_WALL - g for g in G_PAIRS[which]))
    # step intervals: D1->D3 and D3->D6 today; the first decode after a break has none
    assert [v for v, _ in mc.intervals(today)] == [
        PREFILL_WALL + DECODE_WALL,
        2 * PREFILL_WALL + DECODE_WALL,
    ]
    assert [v for v, _ in mc.intervals(mixed)] == [
        PREFILL_WALL + G_PAIRS[which][0],
        2 * PREFILL_WALL + G_PAIRS[which][1],
    ]


def test_host_cost_is_charged_to_each_mixed_forward(trace):
    fwds = kr.read_forwards(str(trace), decode_only=False)
    _, _, pairs = mc.replay(fwds, "attn", host_us=25.0)
    assert [p["saved_us"] for p in pairs] == [DECODE_WALL - G["attn"] - 25.0] * 3


def test_main_scales_to_the_run(trace, tmp_path):
    def scrape(name, prefill, decode, ts):
        p = tmp_path / name
        p.write_text(
            "\n".join(
                f'atom:forward_steps_total{{cudagraph="{cg}",model="{m}",phase="{ph}"}} '
                f"{n if m == 'target' and cg == 'true' else 7.0}"
                for cg in ("true", "false")
                for m in ("target", "draft")
                for ph, n in (("prefill", prefill), ("decode", decode))
            )
            + f"\natom:metrics_last_refresh_timestamp_seconds {ts}\n"
        )
        return str(p)

    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"tok_s_per_gpu": 1000.0}))
    out = tmp_path / "res.json"
    argv = [str(tmp_path / "prof"), "--json", str(out), "--summary", str(summary)]
    argv += ["--metrics-start", scrape("m0", 0, 0, 1000.0)]
    argv += ["--metrics-end", scrape("m1", 100, 1000, 1100.0)]
    assert mc.main(argv) == 0
    res = json.loads(out.read_text())
    r = res["attn"]
    assert (r["prefills"], r["absorbed"], r["decode_steps"]) == (4, 3, 4)
    per_prefill_ms = 3 * (DECODE_WALL - G["attn"]) / 4 / 1e3
    assert r["saved_ms_per_prefill"] == pytest.approx(per_prefill_ms)
    # the draft rows' 7 forwards a scrape cancel out: 100 prefills over 100 s
    saved_s = per_prefill_ms / 1e3 * 100
    assert r["run_saved_s"] == pytest.approx(saved_s)
    assert r["run_gain_pct"] == pytest.approx((100 / (100 - saved_s) - 1) * 100)
    assert r["tok_s_gpu"][1] == pytest.approx(1000 * (1 + r["run_gain_pct"] / 100))
    assert r["itl_p99_ms"] == pytest.approx(
        [(2 * PREFILL_WALL + DECODE_WALL) / 1e3, (2 * PREFILL_WALL + G["attn"]) / 1e3]
    )
    assert res["full"]["run_gain_pct"] < res["attn+moe"]["run_gain_pct"]
