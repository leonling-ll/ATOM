#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Decode kernel roofline of MiniMax-M3.1-Pro from ATOM torch-profiler traces.

Reads one rank's traces (``*.pt.trace.json[.gz]``, as ATOM's ``--torch-profiler-dir``
writes them), keeps the pure-decode forwards (``ATOM::forward ... prefill=0``), and puts
every GPU kernel of a forward against a floor computed from Pro's shapes (the model's
``config.json`` and the forward's rows, sequences and context):

    floor = max(t_min, bytes / BW, flops / F)        latency / memory / compute bound
    floor = AR(rows)                                 all-reduce (measured latency table)

``bytes`` is the compulsory HBM traffic: every weight once, activations in and out once,
each sequence's KV and index pages once. ``t_min`` is the shortest kernel the GPU runs
(measured, or the shortest kernel of the trace). ``BW``, ``F`` and ``AR`` come from
``--peaks`` (a JSON of measured peaks); spec-sheet values stand in when it has none, and
the output says which. Percent of roofline = floor / measured time.

Attribution needs no kernel-order heuristics for the phase: a kernel's ``correlation``
leads to its launch on the CPU thread, whose enclosing ``record_function`` ranges say
which forward (``ATOM::forward``) and which part (``ATOM::run_model`` target, or the
DSpark draft ``draft_kv`` / ``propose_dspark``) launched it. The MXFP8 GEMMs share
kernel names across roles, so they are labelled by their order between two indexer
``_prep_kernel`` launches: the first after a prep is ``o_proj``, the last two before the
next prep ``qk_proj`` and ``index_qk_proj`` of the next layer, the rest the shared expert
(the dense MLP in the dense layers). The draft's GEMMs are one group, against the
draft's weight bytes.

Outputs (``--out DIR``): ``kernels.csv`` (one row per bucket and kernel: calls, us, bytes,
FLOPs, TB/s, TFLOPS, floor, percent of roofline, bound, note), ``steps.csv`` (one row per
bucket: step time, coverage, share within 80% of roofline, gaps) and ``summary.md``
(per bucket: the step split and the gap list ranked by time - floor). A bucket is a
concurrency (the ``c<N>`` directory a trace sits in, else 0) and a verify width.

usage:
  kernel_roofline.py TRACES... --config CONFIG.json --out DIR [--peaks PEAKS.json]
                     [--rank 0] [--tp 8] [--ctx N] [--min-forwards 3]
  kernel_roofline.py --mono-floor 1,8,16 --config CONFIG.json [--peaks PEAKS.json]
"""

import argparse
import collections
import csv
import gzip
import itertools
import json
import math
import os
import re
import sys
from dataclasses import dataclass

MX8 = 1 + 1 / 32  # bytes an element: e4m3 + one E8M0 scale a 32-block
MX4 = 0.5 + 1 / 32  # e2m1 + one E8M0 scale a 32-block
BF16 = 2

# MI355X spec sheet, one GPU, dense (stand-ins when --peaks has no measurement)
SPEC_PEAKS = {
    "hbm_tbs": 8.0,
    "bf16_tflops": 2516.6,
    "fp8_tflops": 5033.2,
    "fp4_tflops": 10066.3,
}
PEAK_KEYS = ("hbm_tbs", "bf16_tflops", "fp8_tflops", "fp4_tflops")


# ---------------------------------------------------------------- model shapes
@dataclass(frozen=True)
class Shapes:
    """Pro's sizes a rank sees at tensor parallel ``tp`` (from the model config)."""

    hidden: int
    layers: int
    dense_layers: int
    heads: int  # query heads a rank
    head_dim: int
    value_dim: int
    idx_heads: int
    idx_dim: int
    topk_blocks: int
    page: int
    experts: int
    topk: int
    moe_inter: int  # routed intermediate a rank
    shared_inter: int  # shared-expert intermediate a rank
    dense_inter: int  # dense-MLP intermediate a rank
    vocab: int  # lm_head rows a rank
    draft_layers: int
    draft_inter: int  # draft dense-FFN intermediate a rank
    draft_inputs: int  # target layers the draft's fc reads
    draft_window: int  # the draft attention's sliding window (0: whole context)
    tp: int

    @classmethod
    def from_config(cls, cfg: dict, tp: int) -> "Shapes":
        t = cfg.get("text_config", cfg)
        sp = t.get("sparse_attention_config", {})
        freq = t.get("moe_layer_freq") or [1] * t["num_hidden_layers"]
        hd = t["head_dim"]
        rot = int(t.get("rotary_dim") or hd * t.get("partial_rotary_factor", 1.0))
        dflash = t.get("dflash_config", {})
        return cls(
            hidden=t["hidden_size"],
            layers=t["num_hidden_layers"],
            dense_layers=sum(1 for f in freq if not f),
            heads=t["num_attention_heads"] // tp,
            head_dim=hd,
            value_dim=sp.get("value_head_dim") or hd - rot,
            idx_heads=sp.get("sparse_num_index_heads", 0),
            idx_dim=sp.get("sparse_index_dim", 0),
            topk_blocks=sp.get("sparse_topk_blocks", 0),
            page=sp.get("sparse_block_size", 16),
            experts=t["num_local_experts"],
            topk=t["num_experts_per_tok"],
            moe_inter=t["intermediate_size"] // tp,
            shared_inter=t.get("shared_intermediate_size", 0) // tp,
            dense_inter=t.get("dense_intermediate_size", 0) // tp,
            vocab=math.ceil(t["vocab_size"] / tp),
            draft_layers=dflash.get("num_hidden_layers", 0),
            draft_inter=t.get("dspark_ffn_hidden_size", 0) // tp,
            draft_inputs=len(t.get("dspark_target_layer_ids") or []),
            draft_window=int(t.get("sliding_window") or 0),
            tp=tp,
        )

    def gemms(self) -> dict:
        """Target GEMM role -> (N, K, weight bytes an element, peak)."""
        h = self.hidden
        return {
            "qk_proj": (self.heads * self.head_dim + self.head_dim, h, MX8, "fp8"),
            "index_qk_proj": ((self.idx_heads + 1) * self.idx_dim, h, MX8, "fp8"),
            "o_proj": (h, self.heads * self.value_dim, MX8, "fp8"),
            "shared_gate_up": (2 * self.shared_inter, h, MX8, "fp8"),
            "shared_down": (h, self.shared_inter, MX8, "fp8"),
            "dense_gate_up": (2 * self.dense_inter, h, MX8, "fp8"),
            "dense_down": (h, self.dense_inter, MX8, "fp8"),
            "router": (self.experts, h, BF16, "bf16"),
            "lm_head": (self.vocab, h, BF16, "bf16"),
        }

    def draft_weight_bytes(self, part: str) -> float:
        """The DSpark draft's GEMM weights read once a draft call (weights only)."""
        h, d = self.hidden, self.head_dim
        if part == "draft_kv":  # fc (bf16, replicated) + each layer's context K / V
            return (
                self.draft_inputs * h * h * BF16 + self.draft_layers * 2 * d * h * MX8
            )
        qkv = (self.heads * d + 2 * d) * h
        o = h * self.heads * d
        ffn = 3 * self.draft_inter * h
        return self.draft_layers * (qkv + o + ffn) * MX8 + self.vocab * h * BF16


def expected_unique(experts: int, topk: int, rows: int) -> float:
    """Expected distinct experts ``rows`` tokens pick under uniform routing."""
    return experts * (1 - (1 - topk / experts) ** max(rows, 0))


def gemm_cost(m, n, k, w_b, a_b=MX8, o_b=BF16):
    return n * k * w_b + m * k * a_b + m * n * o_b, 2.0 * m * n * k


def inter_from_name(name: str, default: int) -> int:
    """The routed intermediate a kernel was built for (``_i768_``: padded)."""
    m = re.search(r"_i(\d+)_", name)
    return int(m.group(1)) if m else default


# ---------------------------------------------------------------- classification
# (kernel-name regex, role); first match wins. A role of None: an MXFP8 GEMM, whose role
# comes from its order (label_gemms).
RULES = [
    (r"minimax_m3_pro_mono_layer", "mono_layer"),
    (r"mono_step_begin", "misc"),
    (r"topk_per_row_argmax|argmax|rejection|_sample|multinomial", "sampling"),
    (r"allgather|all_gather", "allgather"),
    (r"local_device_load_rmsnorm", "allreduce_tail"),
    (
        r"lamport|allreduce|all_reduce|reduce_scatter|cross_device_reduce|[nr]cclDevKernel",
        "allreduce",
    ),
    (r"_prep_kernel|fused_qkv?_norm_rope_cache", "prep"),
    (r"_score_|index_score", "index_score"),
    (r"_select_|top_k_per_row|_topk_index|radix", "index_topk"),
    (
        r"_attn_kernel|paged_attention|pa_decode|_combine_kernel|fmha|mha_",
        "sparse_attn",
    ),
    (r"gemm1_a(16|4|8)w4", "routed_gemm1"),
    (r"gemm2_a(16|4|8)w4", "routed_gemm2"),
    (r"topk_gating|topk_softmax|grouped_topk", "router_topk"),
    (r"moe_sorting|moe_align", "sort"),
    (r"hgemm_bf16", "router"),
    (r"mxpsh|bmm_a8w8_mxscale", None),
    (r"Cijk_", "lm_head"),
    (r"swiglu|act_and_mul", "act"),
    (r"rmsnorm|RMSNorm|layernorm", "norm"),
    (r"quantize_fp8|_quant_kernel|per_token_quant|mxfp8_quant", "quant"),
    (r"CUDAFunctor_add", "add"),
    (
        (
            r"Memcpy|Memset|copyBuffer|scatter_packed|_fill_deferred|kv_indices|embedding"
            r"|arange|searchsorted|clamp|FillFunctor|copy_kernel|CatArray|elementwise_kernel"
            r"|index_put|_markov|sigmoid|gather|indexSelect|distribution_|_stage_anchors"
        ),
        "misc",
    ),
]


# role -> module of the report
MODULES = {
    "prep": "attn.prep",
    "index_score": "attn.index_score",
    "index_topk": "attn.index_topk",
    "sparse_attn": "attn.sparse_attn",
    "qk_proj": "attn.qk_proj",
    "index_qk_proj": "attn.qk_proj",
    "o_proj": "attn.o_proj",
    "router": "moe.router",
    "router_topk": "moe.router",
    "sort": "moe.router",
    "routed_gemm1": "moe.routed",
    "routed_gemm2": "moe.routed",
    "shared_gate_up": "moe.shared",
    "shared_down": "moe.shared",
    "add": "moe.shared",
    "dense_gate_up": "dense_mlp",
    "dense_down": "dense_mlp",
    "allreduce": "allreduce",
    "allreduce_tail": "allreduce",
    "allgather": "lm_head",
    "lm_head": "lm_head",
    "sampling": "sampling",
    "norm": "norm_quant",
    "quant": "norm_quant",
    "mono_layer": "mono",
    "misc": "misc",
}


def classify(name: str):
    """The kernel's role (None: an MXFP8 GEMM, labelled by order), or 'other'."""
    for pattern, role in RULES:
        if re.search(pattern, name):
            return role
    return "other"


def short_name(name: str) -> str:
    """A kernel name without template arguments, for grouping."""
    n = re.sub(r"^void |\(anonymous namespace\)::", "", name)
    n = n.split("(")[0].split("<")[0]
    m = re.match(r"_ZN\d+(\w+?)\d+(\w+?)I", n)  # a mangled aiter template
    if m:
        n = m.group(1) + "::" + m.group(2)
    return n[:96]


def label_gemms(kernels, dense_layers: int) -> dict:
    """Index -> role of the target's MXFP8 GEMMs, by their order between preps."""
    preps = [i for i, k in enumerate(kernels) if k[2] == "prep"]
    bounds = [-1] + preps + [len(kernels)]
    lab = {}
    for seg in range(len(bounds) - 1):
        g = [
            i for i in range(bounds[seg] + 1, bounds[seg + 1]) if kernels[i][2] is None
        ]
        if not g:
            continue
        if seg == 0:  # before layer 0's prep: its q+k and index q+k
            for i, role in zip(g[-2:], ("qk_proj", "index_qk_proj")):
                lab[i] = role
            continue
        layer = seg - 1
        last = seg == len(bounds) - 2
        mono = any(
            kernels[i][2] == "mono_layer"
            for i in range(bounds[seg] + 1, bounds[seg + 1])
        )
        tail = [] if last else g[-2:]
        for i, role in zip(tail, ("qk_proj", "index_qk_proj")):
            lab[i] = role
        if mono:  # o_proj and the FFN run inside the mono kernel
            continue
        lab[g[0]] = "o_proj"
        mid = [i for i in g[1:] if i not in tail]
        kind = "dense" if layer < dense_layers else "shared"
        for i, role in zip(mid, (f"{kind}_gate_up", f"{kind}_down")):
            lab[i] = role
    return lab


# ---------------------------------------------------------------- trace parsing
def parse_kv(label: str) -> dict:
    return dict(m.groups() for m in re.finditer(r"(\w+)=([^\s\]]+)", label))


def first_int(v, default=None):
    m = re.match(r"-?\d+", str(v)) if v is not None else None
    return int(m.group()) if m else default


def open_trace(path):
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        tr = json.load(f)
    return tr["traceEvents"] if isinstance(tr, dict) else tr


DRAFT = ("draft_kv[", "propose_dspark[", "dspark_sched[", "propose_eagle[")


def read_forwards(path, idle_cap_us=5000.0):
    """Pure-decode forwards of one trace: dicts with the forward's labels, its kernels
    [(ts, dur, role, name, part, draft part)] in GPU order and its ``wall`` (to the next
    forward's first kernel). Dropped: the first and last forward (the window edges cut
    them) and a forward followed by more than ``idle_cap_us`` of GPU idle (no work was
    ready: request gaps, not decode)."""
    anns, launches, kers = [], [], []
    for e in open_trace(path):
        if e.get("ph") != "X":
            continue
        c = e.get("cat")
        if c == "user_annotation":
            anns.append(e)
        elif c in ("cuda_runtime", "cuda_driver") and (e.get("args") or {}).get(
            "correlation"
        ):
            launches.append(e)
        elif c in ("kernel", "gpu_memcpy", "gpu_memset"):
            kers.append(e)
    by_tid = collections.defaultdict(list)
    for a in anns:
        by_tid[(a.get("pid"), a.get("tid"))].append(a)
    owner = {}  # correlation -> (forward annotation id, part, draft label)
    labels = collections.defaultdict(dict)
    for tid, al in by_tid.items():
        al.sort(key=lambda a: (a["ts"], -a.get("dur", 0)))
        ls = sorted(
            (r for r in launches if (r.get("pid"), r.get("tid")) == tid),
            key=lambda r: r["ts"],
        )
        stack, j = [], 0
        for r in ls:
            t = r["ts"]
            while j < len(al) and al[j]["ts"] <= t:
                stack.append(al[j])
                j += 1
            stack = [a for a in stack if a["ts"] + a.get("dur", 0) > t]
            fwd = next(
                (a for a in stack if a["name"].startswith("ATOM::forward")), None
            )
            if fwd is None:
                continue
            names = [a["name"] for a in stack]
            draft = next((n for n in names if n.startswith(DRAFT)), None)
            run = next(
                (n for n in names if n.startswith(("decode[", "prefill["))), None
            )
            part = (
                "draft"
                if draft
                else (
                    "target"
                    if any(n.startswith("ATOM::run_model") for n in names)
                    else "post"
                )
            )
            fid = id(fwd)
            labels[fid].setdefault("forward", fwd["name"])
            if run:
                labels[fid].setdefault("run", run)
            owner[r["args"]["correlation"]] = (fid, part, draft)
    per = collections.defaultdict(list)
    for k in kers:
        o = owner.get((k.get("args") or {}).get("correlation"))
        if o is None:
            continue
        name = k["name"]
        cls = "misc" if k.get("cat") != "kernel" else classify(name)
        per[o[0]].append(
            (
                float(k["ts"]),
                float(k.get("dur", 0.0)),
                cls,
                name,
                o[1],
                (o[2] or "").split("[")[0],
            )
        )
    fwds = []
    for fid, kl in per.items():
        kl.sort()
        lab = labels[fid]
        fkv, rkv = parse_kv(lab["forward"]), parse_kv(lab.get("run", ""))
        fwds.append(
            {
                "decode": not first_int(fkv.get("prefill"), 0)
                and first_int(fkv.get("decode"), 0) > 0,
                "rows": first_int(fkv.get("tokens"), 0),
                "seqs": first_int(fkv.get("decode"), 0),
                "spec": first_int(rkv.get("spec"), 0),
                "sk": first_int(rkv.get("sk")),
                "start": kl[0][0],
                "end": max(t + d for t, d, *_ in kl),
                "kernels": kl,
            }
        )
    fwds.sort(key=lambda f: f["start"])
    for a, b in itertools.pairwise(fwds):
        a["wall"] = (
            b["start"] - a["start"] if b["start"] - a["end"] <= idle_cap_us else None
        )
    return [f for f in fwds[1:-1] if f["decode"] and f.get("wall")]


# ---------------------------------------------------------------- the models
def ar_latency(peaks: dict, rows: int):
    """All-reduce latency (us) at ``rows`` from the measured table (linear between
    measured rows; None outside it)."""
    tab = sorted((int(k), v) for k, v in (peaks.get("ar_us") or {}).items())
    if not tab or rows < tab[0][0] or rows > tab[-1][0]:
        return None
    for (r0, v0), (r1, v1) in zip(tab, tab[1:] + tab[-1:]):
        if r0 <= rows <= r1:
            return v0 if r1 == r0 else v0 + (v1 - v0) * (rows - r0) / (r1 - r0)
    return None


COMM = {"allreduce": "ar", "allreduce_tail": "ar", "allgather": "ag"}


def kernel_cost(s: Shapes, role: str, name: str, fw: dict, ctx_tok: float):
    """(bytes, flops, peak, note) of one launch of a target or draft kernel; bytes None:
    no reliable model."""
    m, h, seqs = fw["rows"], s.hidden, max(fw["seqs"], 1)
    if role in s.gemms():
        n, k, wb, peak = s.gemms()[role]
        b, f = gemm_cost(m, n, k, wb, a_b=BF16 if wb == BF16 else MX8)
        return b, f, peak, ""
    if role in ("routed_gemm1", "routed_gemm2"):
        inter = inter_from_name(name, s.moe_inter)
        u = expected_unique(s.experts, s.topk, m)
        w = (2 if role == "routed_gemm1" else 1) * inter * h * MX4
        act = m * h * BF16 + m * s.topk * inter * BF16
        flops = 2.0 * m * s.topk * (2 if role == "routed_gemm1" else 1) * inter * h
        return (
            u * w + act,
            flops,
            "bf16",
            f"U={u:.1f} experts (uniform routing, an upper bound)",
        )
    if role == "mono_layer":
        b, f = mono_cost(s, m)
        return (
            b,
            f,
            "fp4",
            f"U={expected_unique(s.experts, s.topk, m):.1f} (uniform routing)",
        )
    qkv_out = (s.heads * s.head_dim + s.head_dim + (s.idx_heads + 1) * s.idx_dim) * BF16
    kv_tok = s.head_dim + 4  # fp8 K row + its scale
    idx_tok = s.idx_dim + 4  # fp8 index-K row + its scale
    if role == "prep":
        return m * qkv_out * 2 + m * (kv_tok + idx_tok), 0.0, "bf16", ""
    if role == "index_score":
        flops = 2.0 * m * s.idx_heads * s.idx_dim * ctx_tok / seqs
        out = m * math.ceil(ctx_tok / seqs / s.page) * 4
        return (
            ctx_tok * idx_tok + m * s.idx_heads * s.idx_dim * BF16 + out,
            flops,
            "fp8",
            "index pages read once a sequence",
        )
    if role == "index_topk":
        return (
            m * math.ceil(ctx_tok / seqs / s.page) * 4 + m * s.topk_blocks * 4,
            0.0,
            "bf16",
            "",
        )
    if role == "sparse_attn" and fw.get("draft"):  # K and V a token, sliding window
        sel = min(ctx_tok / seqs, s.draft_window or ctx_tok / seqs)
        flops = 2.0 * m * s.heads * sel * 2 * s.head_dim
        if "reduce" in name:
            return m * s.heads * s.head_dim * 4 * 2, 0.0, "bf16", "split-K partials"
        return (
            seqs * sel * 2 * kv_tok + 2 * m * s.heads * s.head_dim * BF16,
            flops,
            "fp8",
            "window pages read once a sequence",
        )
    if role == "sparse_attn":
        sel = min(ctx_tok / seqs, s.topk_blocks * s.page)
        flops = 2.0 * m * s.heads * sel * (s.head_dim + s.value_dim)
        if "_combine" in name or "reduce" in name:
            return m * s.heads * s.value_dim * 4 * 2, 0.0, "bf16", "split-K partials"
        return (
            seqs * sel * kv_tok + m * s.heads * (s.head_dim + s.value_dim) * BF16,
            flops,
            "fp8",
            "selected pages read once a sequence",
        )
    if role == "norm":
        return m * h * (3 * BF16 + MX8), 0.0, "bf16", ""
    if role == "quant":
        return m * s.heads * s.value_dim * (BF16 + MX8), 0.0, "bf16", ""
    if role == "act":
        inter = s.dense_inter if fw.get("dense") else s.shared_inter
        return m * inter * (2 * BF16 + MX8), 0.0, "bf16", ""
    if role == "add":
        return 3 * m * h * BF16, 0.0, "bf16", ""
    if role in ("router_topk", "sort", "misc", "sampling"):
        return m * s.experts * 4 if role == "router_topk" else 0.0, 0.0, "bf16", ""
    return None, None, None, "no model for this kernel"


def mono_cost(s: Shapes, m: int):
    """Bytes and FLOPs of the mono v1 MoE-layer kernel (o_proj, AR #1 + norms, router
    split over the ranks, routed W4A4 without the intermediate pad, MXFP8 shared)."""
    stages = mono_stages(s, m)
    return (
        sum(v[0] for v in stages.values() if v[0]),
        sum(v[1] for v in stages.values() if v[1]),
    )


def mono_stages(s: Shapes, m: int) -> dict:
    """Stage -> (bytes, flops, peak) of mono v1 at ``m`` rows; AR stages carry None."""
    h, u = s.hidden, expected_unique(s.experts, s.topk, m)
    o_b, o_f = gemm_cost(m, h, s.heads * s.value_dim, MX8)
    r_b, r_f = gemm_cost(m, s.experts // s.tp, h, BF16, a_b=BF16)
    sh_b = 3 * s.shared_inter * h * MX8 + m * h * MX8 + m * h * BF16
    return {
        "o_proj": (o_b, o_f, "fp8"),
        "ar1_norm_quant": (None, None, "comm"),
        "router": (r_b, r_f, "bf16"),
        "routed_w4a4": (
            u * 3 * s.moe_inter * h * MX4 + m * h * MX4,
            2.0 * m * s.topk * 3 * s.moe_inter * h,
            "fp4",
        ),
        "shared_mxfp8": (sh_b, 2.0 * m * 3 * s.shared_inter * h, "fp8"),
        "ar2": (None, None, "comm"),
    }


def floor_us(peaks: dict, t_min: float, nbytes: float, flops: float, peak: str):
    """(floor us, bound) of one launch: the highest of the latency, memory and compute
    ceilings."""
    t_mem = nbytes / (peaks["hbm_tbs"] * 1e12) * 1e6
    t_cmp = flops / (peaks[f"{peak}_tflops"] * 1e12) * 1e6 if flops else 0.0
    bound = "memory" if t_mem >= t_cmp else "compute"
    if t_min >= max(t_mem, t_cmp):
        bound = "latency"
    return max(t_min, t_mem, t_cmp), bound


# ---------------------------------------------------------------- per-forward rows
def blank(key, dur, note):
    """A kernel row with no metric."""
    return key, dur, None, None, None, "", note


def forward_rows(fw: dict, s: Shapes, peaks: dict, t_min: float, ctx: int):
    """One row a kernel launch: (key, dur, bytes, flops, floor, bound, note); key =
    (phase, module, role, short name). Floors of grouped kernels (all-reduce, the
    draft's GEMMs) are spread over the group by time."""
    ctx_tok = fw["sk"] if fw["sk"] is not None else ctx * fw["seqs"]
    tgt = [k for k in fw["kernels"] if k[4] == "target"]
    lab = label_gemms(tgt, s.dense_layers)
    rows, groups = [], collections.defaultdict(list)
    layer = -1
    for i, (ts, dur, role, name, part, dpart) in enumerate(tgt):
        if role == "prep":
            layer += 1
        dense = 0 <= layer < s.dense_layers
        if role is None:
            role = lab.get(i, "other")
        module = MODULES.get(role, "other")
        if role == "act":
            module = "dense_mlp" if dense else "moe.shared"
        key = ("target", module, role, short_name(name))
        if role in COMM:
            groups[(COMM[role], "target", fw["rows"])].append((key, dur, role))
            continue
        b, f, peak, note = kernel_cost(s, role, name, {**fw, "dense": dense}, ctx_tok)
        if b is None:
            rows.append(blank(key, dur, note))
            continue
        fl, bound = floor_us(peaks, t_min, b, f, peak)
        if role == "mono_layer":  # AR #1 runs inside the kernel
            lat = ar_latency(peaks, fw["rows"])
            if lat is None:
                note = f"no AR latency at {fw['rows']} rows for the in-kernel AR #1"
                rows.append(blank(key, dur, note))
                continue
            fl, note = fl + lat, note + f"; + AR #1 {lat:.2f} us"
        rows.append((key, dur, b, f, fl, bound, note))
    for ts, dur, role, name, part, dpart in fw["kernels"]:
        if part == "target":
            continue
        phase = dpart if part == "draft" else "post"
        gemm = role in (None, "lm_head", "router")
        role = role or "gemm"
        key = (
            phase,
            "draft" if part == "draft" else MODULES.get(role, "other"),
            role,
            short_name(name),
        )
        drows = fw["seqs"] * max(fw["spec"], 1) if part == "draft" else fw["rows"]
        if role in COMM:
            groups[(COMM[role], phase, drows)].append((key, dur, role))
        elif part == "draft" and gemm:
            groups[("draft_gemm", dpart, drows)].append((key, dur, role))
        else:
            b, f, peak, note = kernel_cost(
                s, role, name, {**fw, "rows": drows, "draft": part == "draft"}, ctx_tok
            )
            if b is None:
                rows.append(blank(key, dur, note))
                continue
            fl, bound = floor_us(peaks, t_min, b, f, peak)
            rows.append((key, dur, b, f, fl, bound, note))
    for (kind, tag, m), items in groups.items():
        total = sum(d for _, d, _ in items) or 1.0
        if kind == "ag":
            rows += [blank(k, d, "no measured all-gather latency") for k, d, _ in items]
            continue
        if kind == "ar":
            lat = ar_latency(peaks, m)
            ops = sum(1 for *_, r in items if r == "allreduce")
            if lat is None or not ops:
                rows += [
                    blank(k, d, f"no measured AR latency at {m} rows")
                    for k, d, _ in items
                ]
                continue
            gfloor, gbytes, bound, note = (
                ops * lat,
                ops * m * s.hidden * BF16,
                "comm",
                f"{ops} all-reduces x {lat:.2f} us measured",
            )
            gflops = 0.0
        else:
            gbytes = s.draft_weight_bytes(tag)
            gflops = 2.0 * m * gbytes  # ~2 FLOPs a weight byte a row (MXFP8 ~1 B)
            gfloor, bound = floor_us(peaks, t_min * len(items), gbytes, gflops, "fp8")
            note = f"draft {tag} GEMMs as one group (weights only)"
        for key, d, r in items:
            w = d / total
            rows.append((key, d, gbytes * w, gflops * w, gfloor * w, bound, note))
    return rows


# ---------------------------------------------------------------- aggregation
def conc_of(path: str) -> int:
    for part in reversed(path.split(os.sep)):
        m = re.fullmatch(r"c(\d+)", part)
        if m:
            return int(m.group(1))
    return 0


def union_len(iv) -> float:
    tot, cs, ce = 0.0, None, None
    for a, b in sorted(iv):
        if ce is None or a > ce:
            tot += (ce - cs) if ce is not None else 0.0
            cs, ce = a, b
        else:
            ce = max(ce, b)
    return tot + ((ce - cs) if ce is not None else 0.0)


def analyse(fwds_by_bucket: dict, s: Shapes, peaks: dict, t_min: float, ctx: int):
    """bucket -> (step dict, {key: [calls, us, bytes, flops, floor, bound, note]})."""
    out = {}
    for bk, fwds in sorted(fwds_by_bucket.items()):
        n = len(fwds)
        agg = {}
        step = collections.Counter()
        for fw in fwds:
            iv = [(t, t + d) for t, d, *_ in fw["kernels"]]
            busy, span = union_len(iv), fw["end"] - fw["start"]
            step["wall"] += fw["wall"]
            step["kernels"] += sum(d for _, d, *_ in fw["kernels"])
            step["gaps in forward"] += span - busy
            step["idle between forwards"] += fw["wall"] - span
            step["ctx"] += fw["sk"] if fw["sk"] is not None else ctx * fw["seqs"]
            step["seqs"] += fw["seqs"]
            for key, d, b, f, fl, bound, note in forward_rows(fw, s, peaks, t_min, ctx):
                a = agg.setdefault(
                    key, [0, 0.0, 0.0, 0.0, 0.0, bound, note, b is not None]
                )
                a[0] += 1
                a[1] += d
                if b is not None:
                    a[2] += b
                    a[3] += f
                    a[4] += fl
        out[bk] = (
            {k: v / n for k, v in step.items()} | {"forwards": n},
            {
                k: [v[0] / n, v[1] / n, v[2] / n, v[3] / n, v[4] / n, *v[5:]]
                for k, v in agg.items()
            },
        )
    return out


def gap_list(step: dict, agg: dict):
    """(gap us, what, us, floor us, pct) ranked by time - floor."""
    items = [
        (us - fl, "/".join(k[:3]) + f" `{k[3]}`", us, fl, fl / us if us else 0.0)
        for k, (_, us, _, _, fl, _, _, ok) in agg.items()
        if ok
    ]
    for what in ("gaps in forward", "idle between forwards"):
        items.append((step[what], f"GPU {what}", step[what], 0.0, 0.0))
    return sorted(items, reverse=True)


def fmt(x, nd=1):
    return (
        "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"
    )


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_outputs(res: dict, out: str, peaks: dict, t_min: float, src: dict):
    os.makedirs(out, exist_ok=True)
    L = [
        "# Decode kernel roofline",
        "",
        (
            f"Peaks: {', '.join(f'{k} {peaks[k]:g} ({src[k]})' for k in PEAK_KEYS)}; "
            f"t_min {t_min:.2f} us ({src['kernel_min_us']}); AR table: {src['ar_us']}."
        ),
        "",
    ]
    steps, kernels = [], []
    for (conc, rows), (step, agg) in res.items():
        wall = step["wall"]
        cov = sum(v[1] for v in agg.values() if v[7])
        w80 = sum(v[1] for v in agg.values() if v[7] and v[4] >= 0.8 * v[1])
        fl = sum(v[4] for v in agg.values() if v[7])
        lat = [v for v in agg.values() if v[7] and v[5] == "latency"]
        lat_us, lat_n = sum(v[1] for v in lat), sum(v[0] for v in lat)
        gaps = gap_list(step, agg)
        ctxs = step["ctx"] / step["seqs"] if step["seqs"] else 0
        steps.append(
            {
                "conc": conc,
                "rows": rows,
                "forwards": step["forwards"],
                "step_us": round(wall, 1),
                "kernel_us": round(step["kernels"], 1),
                "covered_pct": round(100 * cov / step["kernels"], 1),
                "within80_pct": round(100 * w80 / wall, 1),
                "floor_us": round(fl, 1),
                "latency_bound_us": round(lat_us, 1),
                "latency_bound_launches": round(lat_n, 1),
                "gaps_in_forward_us": round(step["gaps in forward"], 1),
                "idle_between_us": round(step["idle between forwards"], 1),
                "ctx_per_seq": round(ctxs),
                "top3_gaps": "; ".join(f"{g[1]} {g[0]:.0f} us" for g in gaps[:3]),
            }
        )
        for k, (calls, us, b, f, flo, bound, note, ok) in sorted(
            agg.items(), key=lambda kv: -kv[1][1]
        ):
            kernels.append(
                {
                    "conc": conc,
                    "rows": rows,
                    **dict(zip(("phase", "module", "role", "kernel"), k)),
                    "calls_per_step": round(calls, 2),
                    "us_per_step": round(us, 2),
                    "share_of_step": round(us / wall, 4),
                    "bytes_per_step": round(b) if ok else "",
                    "flops_per_step": round(f) if ok else "",
                    "tb_s": round(b / us / 1e6, 3) if ok and us else "",
                    "tflops": round(f / us / 1e6, 2) if ok and us else "",
                    "floor_us": round(flo, 2) if ok else "",
                    "pct_roofline": round(100 * flo / us, 1) if ok and us else "",
                    "bound": bound if ok else "",
                    "note": note,
                }
            )
        L += [
            (
                f"## c{conc}, {rows} rows a step ({step['forwards']} forwards, "
                f"{ctxs:.0f} context tokens a sequence)"
            ),
            "",
            (
                f"Step {wall:.0f} us: kernels {step['kernels']:.0f} us (with a model: "
                f"{100 * cov / step['kernels']:.1f}%), launch gaps "
                f"{step['gaps in forward']:.0f} us, idle between forwards "
                f"{step['idle between forwards']:.0f} us. Floor of the modelled "
                f"kernels {fl:.0f} us. Within 80% of roofline: "
                f"{100 * w80 / wall:.1f}% of the step. Latency-bound kernels (floor = "
                f"t_min): {lat_n:.0f} launches, {lat_us:.0f} us."
            ),
            "",
            "| module | us | share | floor us | % roofline |",
            "|---|---:|---:|---:|---:|",
        ]
        mods = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
        for k, v in agg.items():
            m = mods[f"{k[0]}/{k[1]}"]
            m[0] += v[1]
            m[1] += v[4] if v[7] else 0.0
            m[2] += v[1] if v[7] else 0.0
        for name, (us, flo, cus) in sorted(mods.items(), key=lambda kv: -kv[1][0]):
            L.append(
                f"| {name} | {us:.0f} | {100 * us / wall:.1f}% | {flo:.0f} | "
                f"{fmt(100 * flo / cus if cus else None)} |"
            )
        L += [
            "",
            "Gap list (time - floor, us a step):",
            "",
            "| # | item | us | floor | % roofline | gap |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for i, (g, what, us, flo, p) in enumerate(gaps[:15], 1):
            L.append(
                f"| {i} | {what} | {us:.1f} | {flo:.1f} | {100 * p:.0f}% | "
                f"{g:.1f} |"
            )
        L.append("")
    write_csv(os.path.join(out, "steps.csv"), steps)
    write_csv(os.path.join(out, "kernels.csv"), kernels)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def load_peaks(path):
    got = {}
    if path:
        with open(path) as f:
            got = json.load(f)
    peaks, src = {}, {}
    for k in PEAK_KEYS:
        if got.get(k):
            peaks[k], src[k] = float(got[k]), "measured"
        else:
            peaks[k], src[k] = SPEC_PEAKS[k], "spec"
    peaks["ar_us"] = got.get("ar_us") or {}
    src["ar_us"] = got.get("ar_source", "measured") if peaks["ar_us"] else "none"
    peaks["kernel_min_us"] = got.get("kernel_min_us")
    src["kernel_min_us"] = (
        "measured" if peaks["kernel_min_us"] else "shortest kernel in the trace"
    )
    return peaks, src


def find_traces(paths, rank):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for root, _, fs in os.walk(p):
                if re.search(r"rank_\d+$", root) and not root.endswith(f"rank_{rank}"):
                    continue
                files += [
                    os.path.join(root, f)
                    for f in fs
                    if f.endswith(
                        (".pt.trace.json.gz", ".pt.trace.json", ".trace.json.gz")
                    )
                ]
        else:
            files.append(p)
    return sorted(files)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--config", required=True, help="the model's config.json")
    ap.add_argument("--out")
    ap.add_argument(
        "--peaks",
        help="JSON of measured peaks (hbm_tbs, *_tflops, ar_us, " "kernel_min_us)",
    )
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument(
        "--ctx",
        type=int,
        default=0,
        help="context tokens a sequence, for traces without sk labels",
    )
    ap.add_argument("--min-forwards", type=int, default=3)
    ap.add_argument(
        "--idle-cap-us",
        type=float,
        default=5000.0,
        help="leave out forwards followed by more GPU idle than this",
    )
    ap.add_argument("--mono-floor", help="print the mono v1 stage floors at these rows")
    a = ap.parse_args(argv)
    with open(a.config) as f:
        s = Shapes.from_config(json.load(f), a.tp)
    peaks, src = load_peaks(a.peaks)
    if a.mono_floor:
        print("rows,stage,bytes,flops,floor_us,bound")
        for m in (int(x) for x in a.mono_floor.split(",")):
            for st, (b, f, pk) in mono_stages(s, m).items():
                if b is None:
                    lat = ar_latency(peaks, m)
                    print(f"{m},{st},,,{fmt(lat, 2)},comm")
                else:
                    fl, bound = floor_us(peaks, 0.0, b, f, pk)
                    print(f"{m},{st},{b:.0f},{f:.0f},{fl:.2f},{bound}")
        return 0
    if not a.out or not a.paths:
        ap.error("TRACES and --out are required")
    files = find_traces(a.paths, a.rank)
    if not files:
        sys.exit(f"no traces under {a.paths}")
    buckets = collections.defaultdict(list)
    for p in files:
        for fw in read_forwards(p, a.idle_cap_us):
            buckets[(conc_of(p), fw["rows"])].append(fw)
    buckets = {k: v for k, v in buckets.items() if len(v) >= a.min_forwards}
    if not buckets:
        sys.exit("no pure-decode forwards in the traces")
    t_min = peaks["kernel_min_us"]
    if not t_min:
        durs = sorted(
            d
            for fs in buckets.values()
            for fw in fs
            for _, d, *_ in fw["kernels"]
            if d > 0
        )
        t_min = durs[len(durs) // 1000]
    if any(fw["sk"] is None for fs in buckets.values() for fw in fs) and not a.ctx:
        sys.exit("traces without sk labels need --ctx (context tokens a sequence)")
    res = analyse(buckets, s, peaks, t_min, a.ctx)
    write_outputs(res, a.out, peaks, t_min, src)
    print(
        f"wrote {a.out}/kernels.csv, steps.csv, summary.md ({len(res)} buckets, "
        f"{len(files)} traces)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
