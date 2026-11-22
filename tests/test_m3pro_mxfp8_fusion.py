# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro ``ProVariants.fuse_mxfp8_quant``: MXFP8 activation quant fused into the
ops that write the activations, checked bit for bit against the unfused pair (the op, then
``quantize_fp8`` of its bf16 output): the E4M3 bytes, the E8M0 scale bytes, and the bf16 /
residual outputs the fused op still writes.

Sites (``pro.py``):
  1 / 4  attention input / FFN input norm -- TP 1: ``gemma_rms_norm_mxfp8`` (aiter
         fused_qk_rmsnorm_group_quant's MX mode); TP > 1 decode: aiter custom all-reduce
         MXFP8 epilogue; TP > 1 prefill (RCCL): all_reduce + ``aiter.add_rmsnorm_mxfp8``
  2      lookahead indexer input (``ProLookaheadState.put``)
  5      down_proj input (``swiglu_oai_split_mxfp8``)

    pytest -q tests/test_m3pro_mxfp8_fusion.py                  # single GPU
    python tests/test_m3pro_mxfp8_fusion.py --ar --tp 8          # TP 8 all-reduce routes
    python tests/test_m3pro_mxfp8_fusion.py --bench [--tp 8]     # decode microbench
"""

from __future__ import annotations

import argparse
import sys
from functools import partial
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
pytest.importorskip("triton")
aiter = pytest.importorskip("aiter")

from atom.model_ops.blockscale import quantize_fp8
from atom.model_ops.layernorm import GemmaRMSNorm, gemma_rms_norm_mxfp8
from atom.model_ops.swiglu_oai import (
    swiglu_oai_split,
    swiglu_oai_split_mxfp8,
)

DEV = "cuda"
DT = torch.bfloat16
EPS = 1e-6
HIDDEN = 7168
DECODE_M = (1, 4, 16, 32)
PREFILL_M = (100, 2048, 4097)


def _rows(m, n, seed, dev=DEV):
    """bf16 rows over six decades of row scale, with outlier channels, an all-zero row and
    a denormal-scale row (the amax floor)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, n, generator=g)
    x *= 10.0 ** (torch.rand(m, 1, generator=g) * 6 - 3)
    hot = torch.randint(0, n, (max(1, n // 512),), generator=g)
    x[:, hot] *= 64.0
    if m > 2:
        x[1] = 0.0
        x[2] *= 1e-30
    return x.to(DT).to(dev)


def _norm(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    norm = GemmaRMSNorm(n, eps=EPS)
    norm.weight.data = (torch.randn(n, generator=g) * 0.3).to(DT).to(DEV)
    return norm


def _diff(a, b) -> int:
    """Differing bytes (a byte view, so -0 / +0 and NaN payloads count)."""
    if a is None or b is None:
        return 0 if a is b else -1
    a = a.contiguous().view(torch.uint8)
    b = b.contiguous().view(torch.uint8)
    if a.shape != b.shape:
        return -1
    return int((a != b).sum())


# --- sites 1 / 4 at TP 1, site 2's kernel: fused_qk_rmsnorm_group_quant MX mode ---------


@pytest.mark.parametrize("m", DECODE_M + PREFILL_M)
@pytest.mark.parametrize("n", (HIDDEN, 4096, 2048, 1024, 512, 256))
@pytest.mark.parametrize("residual", (True, False))
@pytest.mark.parametrize("write_bf16", (True, False))
def test_gemma_norm_mxfp8(m, n, residual, write_bf16):
    norm = _norm(n)
    x = _rows(m, n, seed=m * 7 + n)
    res = _rows(m, n, seed=m * 7 + n + 1) if residual else None
    ref = norm.forward_cuda(x, res)
    ref_out, ref_res = ref if residual else (ref, None)
    ref_q, ref_s = quantize_fp8(ref_out)

    out, q, s, res_out = gemma_rms_norm_mxfp8(x, res, norm, write_bf16=write_bf16)
    assert _diff(q, ref_q) == 0
    assert _diff(s, ref_s) == 0
    assert _diff(out, ref_out if write_bf16 else None) == 0
    assert _diff(res_out, ref_res) == 0


# --- sites 1 / 4, TP > 1 prefill (RCCL route): add_rmsnorm + MX side output ------------


@pytest.mark.parametrize("m", (1, 3, 64) + PREFILL_M)
@pytest.mark.parametrize("n", (HIDDEN, 8192, 6144, 4096, 2048, 1024, 512))
@pytest.mark.parametrize("gemma", (True, False))
def test_add_rmsnorm_mxfp8(m, n, gemma):
    if not hasattr(aiter, "add_rmsnorm_mxfp8"):
        pytest.skip("this aiter has no add_rmsnorm_mxfp8")
    from aiter.utility.dtypes import fp8

    w = (torch.randn(n) * 0.3).to(DT).to(DEV)
    x = _rows(m, n, seed=m + n)
    res = _rows(m, n, seed=m + n + 1)
    ref_out, ref_res = torch.empty_like(x), torch.empty_like(x)
    aiter.add_rmsnorm(ref_out, x, res, ref_res, w, EPS, gemma)
    ref_q, ref_s = quantize_fp8(ref_out)

    out, res_out = torch.empty_like(x), torch.empty_like(x)
    q = torch.empty((m, n), dtype=fp8, device=DEV)
    s = torch.empty((m, n // 32), dtype=torch.uint8, device=DEV)
    aiter.add_rmsnorm_mxfp8(out, q, s, x, res, res_out, w, EPS, gemma)
    assert _diff(out, ref_out) == 0
    assert _diff(res_out, ref_res) == 0
    assert _diff(q, ref_q) == 0
    assert _diff(s, ref_s) == 0


# --- site 5: SwiGLU-OAI + MXFP8 ------------------------------------------------------------


@pytest.mark.parametrize("rows", (1, 7, 32, 2048))
@pytest.mark.parametrize("n_inter", (96, 256, 384, 1056, 2048, 2304, 4608))
@pytest.mark.parametrize("limit", (7.0, None))
def test_swiglu_mxfp8(rows, n_inter, limit):
    gate_up = _rows(rows, 2 * n_inter, seed=rows * 3 + n_inter)
    ref_q, ref_s = quantize_fp8(swiglu_oai_split(gate_up, 1.702, 1.0, limit))
    q, s = swiglu_oai_split_mxfp8(gate_up, 1.702, 1.0, limit)
    assert _diff(q, ref_q) == 0
    assert _diff(s, ref_s) == 0


def test_swiglu_mxfp8_batched_shape():
    gate_up = _rows(6, 512, seed=5).view(2, 3, 512)
    q, s = swiglu_oai_split_mxfp8(gate_up, 1.702, 1.0, 7.0)
    assert q.shape == (2, 3, 256) and s.shape == (2, 3, 8)
    ref_q, ref_s = quantize_fp8(swiglu_oai_split(gate_up, 1.702, 1.0, 7.0))
    assert _diff(q, ref_q) == 0 and _diff(s, ref_s) == 0


# --- site 2: the lookahead hand-over ----------------------------------------------------


@pytest.mark.parametrize("m", DECODE_M + (100, 2048))
@pytest.mark.parametrize("fuse", (True, False))
def test_lookahead_put_mxfp8(monkeypatch, m, fuse):
    from atom.model_ops.minimax_m3 import pro_attn
    from atom.model_ops.minimax_m3.pro_attn import ProVariants
    from atom.models.minimax_m3 import pro

    monkeypatch.setattr(
        pro_attn,
        "_variants",
        ProVariants(lookahead_src="resid_in", fuse_mxfp8_quant=fuse),
    )
    cfg = SimpleNamespace(
        sparse_attention_config={
            "sparse_indexer_lookahead_num_layers": 2,
            "sparse_indexer_lookahead_start_layer": 2,
            "sparse_indexer_lookahead_source": "pre_ffn_norm",
        }
    )
    st = pro.ProLookaheadState(cfg)
    norm = _norm(HIDDEN, seed=3)
    native = SimpleNamespace(native_a8_group_rows=32)
    st.norms[4] = norm
    st.index_linears[4] = (native, native)
    resid = _rows(m, HIDDEN, seed=m)
    st.put(2, torch.empty(0), resid)
    got = st.source_for(4, norm)
    ref = norm.forward_cuda(resid)
    if fuse:
        ref_q, ref_s = quantize_fp8(ref)
        assert isinstance(got, tuple)
        assert _diff(got[0], ref_q) == 0 and _diff(got[1], ref_s) == 0
    else:
        assert _diff(got, ref) == 0
    # a reader whose indexer projections are not native group32 FP8 keeps bf16
    st.index_linears[4] = (native, SimpleNamespace(native_a8_group_rows=None))
    st.put(2, torch.empty(0), resid)
    assert _diff(st.source_for(4, norm), ref) == 0


# --- TP > 1: the all-reduce routes (python tests/test_m3pro_mxfp8_fusion.py --ar) --------

AR_M = (
    1,
    4,
    16,
    32,
    64,
    80,
    81,
    100,
    256,
    2048,
    4096,
    8192,
)  # 8192: past the custom AR cap


def _ar_worker(tp, rank, init_method, bench):
    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
        tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant,
    )
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    from atom.model_ops.layernorm import fused_allreduce_gemma_rms_norm_mxfp8

    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp, 1)
    comm = get_tp_group().device_communicator
    torch.distributed.all_reduce(
        torch.zeros(1, device=DEV), group=get_tp_group().device_group
    )
    norm = _norm(HIDDEN, seed=11)
    w = norm.weight.data
    rows = []

    def ref(x, res):
        out, res_out = tensor_model_parallel_fused_allreduce_rmsnorm(
            x, res, w, EPS, gemma_norm=True
        )
        return (out, res_out) + quantize_fp8(out)

    def fused(x, res):
        r = tensor_model_parallel_fused_allreduce_rmsnorm_mxfp8_quant(
            x, res, w, EPS, gemma_norm=True
        )
        return None if r is None else (r[0], r[1], r[2], r[3])

    def check(tag, m, a, b):
        rows.append(
            {
                "case": tag,
                "m": m,
                "route": "unfused" if b is None else "fused",
                "diff": -2 if b is None else max(_diff(x, y) for x, y in zip(a, b)),
            }
        )

    for override in (
        None,
        True,
        False,
    ):  # the configured 1-stage rule, forced 1-stage / 2-stage
        comm._ar_1stage_override = override
        for m in AR_M:
            x = _rows(m, HIDDEN, seed=1000 * rank + m)  # each rank's partial sum
            res = _rows(m, HIDDEN, seed=m + 7)
            check(f"eager 1stage={override}", m, ref(x, res), fused(x, res))
            h = fused_allreduce_gemma_rms_norm_mxfp8(x, res, norm)
            r = ref(x, res)
            check(f"atom-helper 1stage={override}", m, r, (h[0], h[3], h[1], h[2]))
    comm._ar_1stage_override = None

    # ProVariants.ar_lamport_rows: Lamport all-reduce + add_rmsnorm_mxfp8 (needs
    # AITER_CUSTOM_AR_DISABLE_LAMPORT unset; else the helper keeps the fused route)
    for m in AR_M:
        if m > 64:
            continue
        x = _rows(m, HIDDEN, seed=1000 * rank + m)
        res = _rows(m, HIDDEN, seed=m + 7)
        h = fused_allreduce_gemma_rms_norm_mxfp8(x, res, norm, lamport_rows=64)
        check("atom-helper lamport", m, ref(x, res), (h[0], h[3], h[1], h[2]))

    # CUDA graphs: the registered-buffer path of the custom all-reduce
    for m in (1, 4, 16, 32, 64):
        x = _rows(m, HIDDEN, seed=1000 * rank + m)
        res = _rows(m, HIDDEN, seed=m + 7)
        fused(x, res)
        ref(x, res)  # warm both before capture
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            got = fused(x, res)
            want = ref(x, res)
        x.copy_(_rows(m, HIDDEN, seed=2000 * rank + m))
        graph.replay()
        torch.cuda.synchronize()
        check("graph", m, want, got)
        graph.reset()

    if bench:
        rows += _bench_ar(
            ref,
            fused,
            lambda x, res: fused_allreduce_gemma_rms_norm_mxfp8(x, res, norm),
            graph_capture,
            rank,
        )

    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()
    return rows


def _graph_us(fn, graph_capture=None, calls=50, replays=20):
    fn()
    graph = torch.cuda.CUDAGraph()
    if graph_capture is None:
        with torch.cuda.graph(graph):
            for _ in range(calls):
                fn()
    else:
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            for _ in range(calls):
                fn()
    graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(replays):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000 / calls)
    graph.reset()
    return sorted(times)[len(times) // 2]


def _bench_ar(ref, fused, helper, graph_capture, rank):
    """Unfused AR + norm + quantize_fp8 vs ATOM's helper (the fused op, or that same
    unfused pair where the fused op declines the route)."""
    out = []
    for m in BENCH_M:
        x = _rows(m, HIDDEN, seed=1000 * rank + m)
        res = _rows(m, HIDDEN, seed=m + 7)
        out.append(
            {
                "case": "bench AR+norm(+quant) us",
                "m": m,
                "route": "unfused" if fused(x, res) is None else "fused",
                "unfused": _graph_us(partial(ref, x, res), graph_capture),
                "fused": _graph_us(partial(helper, x, res), graph_capture),
            }
        )
    return out


def run_ar(tp, bench):
    from multiprocessing import Pool, set_start_method

    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    set_start_method("spawn", force=True)
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp) as pool:
        jobs = [
            pool.apply_async(_ar_worker, (tp, r, init_method, bench)) for r in range(tp)
        ]
        results = [j.get(timeout=1800) for j in jobs]
    bad = 0
    for cells in zip(*results):
        row = dict(cells[0])
        if "diff" in row:
            row["diff"] = max(c["diff"] for c in cells)
            row["route"] = "/".join(sorted({c["route"] for c in cells}))
            bad += row["diff"] > 0 or row["diff"] == -1
        else:
            row["unfused"] = max(c["unfused"] for c in cells)
            row["fused"] = max(c["fused"] for c in cells)
        print(row)
    print("AR MISMATCHES:", bad)
    return bad


# --- single-GPU decode microbench of the per-layer activation path -------------------


BENCH_M = (1, 4, 16, 64, 128)


def _norm_then_quant(norm, x, res):
    out = norm.forward_cuda(x, res)
    return quantize_fp8(out if res is None else out[0])


def _swiglu_then_quant(gu):
    return quantize_fp8(swiglu_oai_split(gu, 1.702, 1.0, 7.0))


def run_bench(inter):
    norm = _norm(HIDDEN)
    for m in BENCH_M:
        x = _rows(m, HIDDEN, seed=m)
        res = _rows(m, HIDDEN, seed=m + 1)
        gu = _rows(m, 2 * inter, seed=m + 2)
        h = norm.forward_cuda(x)
        sites = {
            # site: (unfused = the op, then quantize_fp8 of its bf16 output; fused op)
            "1 attn-in norm (TP1)": (
                partial(_norm_then_quant, norm, x, res),
                partial(gemma_rms_norm_mxfp8, x, res, norm, write_bf16=False),
            ),
            "2 lookahead put": (
                partial(_norm_then_quant, norm, res, None),
                partial(gemma_rms_norm_mxfp8, res, None, norm, write_bf16=False),
            ),
            "4 ffn-in norm, bf16 kept (TP1)": (
                partial(_norm_then_quant, norm, x, res),
                partial(gemma_rms_norm_mxfp8, x, res, norm),
            ),
            f"5 swiglu inter={inter}": (
                partial(_swiglu_then_quant, gu),
                partial(swiglu_oai_split_mxfp8, gu, 1.702, 1.0, 7.0),
            ),
            "quantize_fp8 alone [m,7168]": (partial(quantize_fp8, h), None),
        }
        for site, (unfused, fused) in sites.items():
            u = _graph_us(unfused)
            f = _graph_us(fused) if fused is not None else float("nan")
            print(
                {
                    "case": "bench TP1 us",
                    "site": site,
                    "m": m,
                    "unfused": round(u, 2),
                    "fused": round(f, 2),
                    "saved": round(u - f, 2),
                }
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ar", action="store_true", help="TP all-reduce route checks")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--tp", type=int, default=8)
    p.add_argument(
        "--inter", type=int, default=640, help="per-rank shared/dense intermediate"
    )
    a = p.parse_args()
    if a.ar or (a.bench and a.tp > 1):
        sys.exit(1 if run_ar(a.tp, a.bench) else 0)
    if a.bench:
        run_bench(a.inter)
    else:
        sys.exit(pytest.main([__file__, "-q"]))
