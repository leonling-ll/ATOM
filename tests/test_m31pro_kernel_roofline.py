# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the MiniMax-M3.1-Pro decode kernel roofline
(``scripts/m31pro/kernel_roofline.py``): the bytes / FLOPs models and the trace
parser, on a small synthetic trace labelled the way ATOM labels its forwards."""

import csv
import gzip
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pytest

from atom.model_engine.run_labels import build_run_label

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/m31pro/kernel_roofline.py"
SPEC = importlib.util.spec_from_file_location("m31pro_kernel_roofline", SCRIPT)
kr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(kr)

LAYERS, DENSE = 6, 2
# Pro's text config, cut to six layers (two dense, four MoE)
CONFIG = {
    "text_config": {
        "hidden_size": 7168,
        "intermediate_size": 5120,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": 128,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "vocab_size": 200064,
        "rotary_dim": 64,
        "dense_intermediate_size": 20480,
        "shared_intermediate_size": 5120,
        "num_local_experts": 384,
        "num_experts_per_tok": 4,
        "moe_layer_freq": [0] * DENSE + [1] * (LAYERS - DENSE),
        "dflash_config": {"num_hidden_layers": 5},
        "dspark_ffn_hidden_size": 20480,
        "dspark_target_layer_ids": [1, 2, 3, 4, 5],
        "sliding_window": 2048,
        "sparse_attention_config": {
            "sparse_index_dim": 128,
            "sparse_num_index_heads": 16,
            "sparse_topk_blocks": 128,
            "sparse_block_size": 16,
            "value_head_dim": 192,
        },
    }
}
SHAPES = kr.Shapes.from_config(CONFIG, tp=8)
AR_US = {"1": 4.0, "16": 6.0, "32": 8.0, "64": 10.0}
PEAKS = {
    "hbm_tbs": 5.0,
    "bf16_tflops": 2000.0,
    "fp8_tflops": 4000.0,
    "fp4_tflops": 8000.0,
}

MXPSH = "flydsl_mxpsh_16x32x512_F8_F8_B16_w2_x0_sk1_slrr_pf5_nt"
LAMPORT = (
    "void aiter::lamportAllReduceWaveSplit<std::bfloat16_t, 8>(std::bfloat16_t const*)"
)
NORM = "void aiter::add_rmsnorm_quant_kernel<std::bfloat16_t, 256>(std::bfloat16_t*)"
GEMM1 = "gemm1_a16w4_port_a16w4_h7168_i768_ne384_bm16_tn32_swiglu"
UNKNOWN = "mystery_kernel"


def test_shapes_from_config():
    s = SHAPES
    assert (s.heads, s.value_dim, s.layers, s.dense_layers) == (16, 192, LAYERS, DENSE)
    assert s.gemms()["qk_proj"][:2] == (16 * 256 + 256, 7168)
    assert s.gemms()["index_qk_proj"][:2] == (17 * 128, 7168)
    assert s.gemms()["o_proj"][:2] == (7168, 16 * 192)
    assert s.gemms()["dense_gate_up"][:2] == (2 * 2560, 7168)
    assert s.gemms()["shared_down"][:2] == (7168, 640)
    assert s.gemms()["lm_head"][:2] == (25008, 7168)
    assert (s.moe_inter, s.draft_inter, s.draft_inputs, s.draft_window) == (
        640,
        2560,
        5,
        2048,
    )


def test_gemm_and_routed_models():
    b, f, peak, _ = kr.kernel_cost(SHAPES, "o_proj", MXPSH, {"rows": 16, "seqs": 2}, 0)
    assert peak == "fp8"
    assert b == pytest.approx(7168 * 3072 * kr.MX8 + 16 * 3072 * kr.MX8 + 16 * 7168 * 2)
    assert f == 2 * 16 * 7168 * 3072
    assert kr.expected_unique(384, 4, 1) == pytest.approx(4)
    assert kr.expected_unique(384, 4, 10_000) == pytest.approx(384)
    u = kr.expected_unique(384, 4, 16)
    b, f, peak, note = kr.kernel_cost(
        SHAPES, "routed_gemm1", GEMM1, {"rows": 16, "seqs": 2}, 0
    )
    # the kernel's name says it runs the 640 intermediate padded to 768
    w = u * 2 * 768 * 7168 * kr.MX4
    assert b == pytest.approx(w + 16 * 7168 * 2 + 16 * 4 * 768 * 2)
    assert f == 2 * 16 * 4 * 2 * 768 * 7168 and "uniform" in note
    # attention reads each sequence's selected pages once: at most 128 x 16 tokens
    b, f, _, _ = kr.kernel_cost(
        SHAPES, "sparse_attn", "_attn_kernel", {"rows": 16, "seqs": 2}, 2 * 50_000
    )
    assert b == pytest.approx(2 * 2048 * 260 + 16 * 16 * (256 + 192) * 2)
    b, _, _, note = kr.kernel_cost(SHAPES, "other", UNKNOWN, {"rows": 1, "seqs": 1}, 0)
    assert b is None and note


def test_floor_bounds_and_ar_table():
    assert kr.floor_us(PEAKS, 3.0, 1e3, 0, "fp8") == (3.0, "latency")
    fl, bound = kr.floor_us(PEAKS, 3.0, 50e6, 1e9, "fp8")  # 50 MB at 5 TB/s = 10 us
    assert (fl, bound) == (pytest.approx(10.0), "memory")
    fl, bound = kr.floor_us(
        PEAKS, 3.0, 1e6, 4e11, "fp8"
    )  # 400 GFLOP at 4 PFLOPS = 100 us
    assert (fl, bound) == (pytest.approx(100.0), "compute")
    peaks = {"ar_us": AR_US}
    assert kr.ar_latency(peaks, 16) == 6.0
    assert kr.ar_latency(peaks, 8) == pytest.approx(4.0 + 2.0 * 7 / 15)
    assert kr.ar_latency(peaks, 128) is None and kr.ar_latency({}, 4) is None


def _k(role, name=""):
    return (0.0, 1.0, role, name, "target", "")


def test_label_gemms_by_order():
    # layer 0's q+k | dense layer 0 | MoE layer 1 | mono layer 2 (o_proj and FFN inside) | last
    seq = [
        _k(None),
        _k(None),
        _k("prep"),
        _k(None),
        _k(None),
        _k(None),
        _k(None),
        _k(None),
        _k("prep"),
        _k(None),
        _k(None),
        _k(None),
        _k(None),
        _k(None),
        _k("prep"),
        _k("mono_layer"),
        _k(None),
        _k(None),
        _k("prep"),
        _k(None),
    ]
    lab = kr.label_gemms(seq, dense_layers=1)
    assert [lab[i] for i in (0, 1)] == ["qk_proj", "index_qk_proj"]
    assert [lab[i] for i in (3, 4, 5, 6, 7)] == [
        "o_proj",
        "dense_gate_up",
        "dense_down",
        "qk_proj",
        "index_qk_proj",
    ]
    assert [lab[i] for i in (9, 10, 11, 12, 13)] == [
        "o_proj",
        "shared_gate_up",
        "shared_down",
        "qk_proj",
        "index_qk_proj",
    ]
    assert [lab[i] for i in (16, 17)] == ["qk_proj", "index_qk_proj"]
    assert lab[19] == "o_proj"


@dataclass
class _Batch:  # the fields build_run_label reads
    total_tokens_num: int
    total_seqs_num_prefill: int
    total_seqs_num_decode: int
    num_spec_step: int
    context_lens: object = None


class _Trace:
    """A torch-profiler trace of ATOM forwards: CPU ranges, launches, GPU kernels."""

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
        self.ev.append(
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "hipLaunchKernel",
                "ts": self.cpu,
                "dur": 0.5,
                "pid": 1,
                "tid": 1,
                "args": {"correlation": self.corr},
            }
        )
        self.ev.append(
            {
                "ph": "X",
                "cat": "kernel",
                "name": name,
                "ts": self.gpu,
                "dur": dur,
                "pid": 0,
                "tid": 7,
                "args": {"correlation": self.corr},
            }
        )
        self.gpu += dur + 0.5  # 0.5 us launch gap

    def forward(self, seqs, k, ctx, idle_after=10.0):
        rows = seqs * (k + 1)
        batch = _Batch(rows, 0, seqs, k)
        label = build_run_label(
            is_prefill=False,
            use_cudagraph=True,
            is_dummy=False,
            tbo_on=False,
            scheduled_bs=seqs,
            batch=batch,
            running_bs=seqs,
            detailed_suffix=f" sk={ctx * seqs}",
        )
        f0 = self.cpu
        self.cpu += 1
        r0 = self.cpu
        self.launch("Memcpy DtoD (Device -> Device)", 3.0)
        for layer in range(LAYERS):
            if layer:
                self.launch(LAMPORT, 6.0)
            self.launch(NORM, 4.0)
            self.launch(MXPSH, 8.0)  # qk_proj
            self.launch("bmm_a8w8_mxscale_t16x32x256", 7.0)  # index_qk_proj
            for name, d in (
                ("_prep_kernel", 8.0),
                ("_score_tile_kernel", 5.0),
                ("_select_split_kernel", 4.0),
                ("_attn_kernel", 10.0),
                ("_combine_kernel", 4.0),
                ("quantize_fp8_kernel", 4.0),
                (MXPSH, 5.0),
                (LAMPORT, 6.0),
                (NORM, 4.0),
            ):
                self.launch(name, d)
            if layer < DENSE:
                ffn = ((MXPSH, 9.0), ("_swiglu_oai_mxfp8_kernel", 4.0), (MXPSH, 5.0))
            else:
                ffn = (
                    ("hgemm_bf16_t16x32x64", 5.0),
                    ("aiter::topk_gating_kernel_opt", 6.0),
                    ("aiter::opus_moe_sorting_entry<P0>", 4.0),
                    (GEMM1, 40.0),
                    ("gemm2_a16w4_port_ne384_h7168_i768", 20.0),
                    (MXPSH, 6.0),
                    ("_swiglu_oai_mxfp8_kernel", 4.0),
                    (MXPSH, 4.0),
                    ("at::native::vectorized_elementwise_kernel<CUDAFunctor_add>", 4.0),
                )
            for name, d in ffn:
                self.launch(name, d)
        self.launch(LAMPORT, 6.0)
        self.launch("Cijk_Alik_Bljk_BBS_BH_MT256x16x128", 60.0)  # lm_head
        self.launch(UNKNOWN, 2.0)
        self.span(label, r0, self.cpu + 0.5)
        self.span("ATOM::run_model", r0 - 0.5, self.cpu + 1)
        self.cpu += 2
        d0 = self.cpu
        for name, d in (
            ("bmm_a8w8_mxscale_t64x64x256", 9.0),
            ("_markov_stage1", 5.0),
            ("void aiter::cross_device_reduce_2stage<bf16>()", 10.0),
        ):
            self.launch(name, d)
        self.span(f"propose_dspark[bs={seqs} graph T={k}]", d0, self.cpu + 0.5)
        self.cpu += 2
        self.launch("topk_per_row_argmax_fold1", 5.0)
        self.span(
            f"ATOM::forward tokens={rows} prefill=0 decode={seqs}", f0, self.cpu + 1
        )
        self.cpu += 5
        self.gpu += idle_after

    def write(self, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as f:
            json.dump({"traceEvents": self.ev}, f)


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


@pytest.fixture
def run(tmp_path):
    tr = _Trace()
    for i in range(7):  # forward 3 is followed by 8 ms of idle: no work ready, left out
        tr.forward(seqs=4, k=7, ctx=40_000, idle_after=8000.0 if i == 3 else 10.0)
    trace = tmp_path / "c4" / "w1" / "rank_0" / "x.pt.trace.json.gz"
    tr.write(trace)
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CONFIG))
    peaks = tmp_path / "peaks.json"
    peaks.write_text(json.dumps({**PEAKS, "ar_us": AR_US, "kernel_min_us": 3.0}))
    out = tmp_path / "out"
    assert (
        kr.main(
            [
                str(tmp_path / "c4"),
                "--config",
                str(cfg),
                "--peaks",
                str(peaks),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    return out


def test_trace_end_to_end(run):
    (step,) = _rows(run / "steps.csv")
    # 7 forwards, minus the two edges and the stalled one
    assert (step["conc"], step["rows"], step["forwards"]) == ("4", "32", "4")
    assert float(step["covered_pct"]) >= 90
    k = {(r["phase"], r["role"], r["kernel"]): r for r in _rows(run / "kernels.csv")}
    o = k[("target", "o_proj", MXPSH)]
    assert float(o["calls_per_step"]) == LAYERS
    b, _ = kr.gemm_cost(32, 7168, 3072, kr.MX8)
    assert float(o["floor_us"]) == pytest.approx(LAYERS * max(3.0, b / 5e6), rel=1e-4)
    assert o["bound"] == "memory"
    assert (
        float(k[("target", "shared_gate_up", MXPSH)]["calls_per_step"])
        == LAYERS - DENSE
    )
    assert float(k[("target", "dense_down", MXPSH)]["calls_per_step"]) == DENSE
    # the all-reduces: (2 a layer - 1) + the final one, each at its measured latency
    ar = k[("target", "allreduce", "aiter::lamportAllReduceWaveSplit")]
    ars = 2 * LAYERS
    assert float(ar["calls_per_step"]) == ars
    assert float(ar["floor_us"]) == pytest.approx(ars * 8.0) and ar["bound"] == "comm"
    # the indexer reads each sequence's 40k index-K rows once
    sc = k[("target", "index_score", "_score_tile_kernel")]
    assert float(sc["bytes_per_step"]) / LAYERS == pytest.approx(
        4 * 40_000 * 132 + 32 * 16 * 128 * 2 + 32 * math.ceil(40_000 / 16) * 4, rel=1e-6
    )
    # the draft's GEMMs are one group against the draft weights
    dg = k[("propose_dspark", "gemm", "bmm_a8w8_mxscale_t64x64x256")]
    assert float(dg["bytes_per_step"]) == pytest.approx(
        SHAPES.draft_weight_bytes("propose_dspark"), rel=1e-6
    )
    assert k[("post", "sampling", "topk_per_row_argmax_fold1")]["bound"] == "latency"
    unk = k[("target", "other", UNKNOWN)]
    assert unk["floor_us"] == "" and unk["note"]
    assert "Gap list" in (run / "summary.md").read_text()


def test_needs_context_without_sk_labels(tmp_path):
    tr = _Trace()
    for _ in range(5):
        tr.forward(seqs=1, k=7, ctx=1000)
    for e in tr.ev:
        e["name"] = e["name"].replace(" sk=1000", "")
    tr.write(tmp_path / "t" / "x.pt.trace.json.gz")
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(CONFIG))
    args = [str(tmp_path / "t"), "--config", str(cfg), "--out", str(tmp_path / "o")]
    with pytest.raises(SystemExit):
        kr.main(args)
    assert kr.main(args + ["--ctx", "1000"]) == 0
    (step,) = _rows(tmp_path / "o" / "steps.csv")
    assert step["ctx_per_seq"] == "1000"
