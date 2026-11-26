# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3-Pro's MoE front end on decode-sized steps as one aiter launch
(``fused_all_reduce_rmsnorm_moe_route``, gfx950 Triton/Gluon): the post-attention
all-reduce, the residual add, the Gemma RMSNorm and its MXFP8 copy (the shared
expert's input), the router logits, the sigmoid top-k with the correction bias, the
renormalisation and routed scale, aiter moe_sorting's outputs and the routed
output's zeroing. It replaces six launches a MoE layer (all-reduce, norm, router GEMM,
gating, two sort phases).

The op's IPC buffers are made once per shape for the TP group (`build`, collective:
every rank, before any graph capture), so a model calls `build` at construction.
"""

import inspect
import logging
from functools import cache

import torch

logger = logging.getLogger("atom")

MAX_ROWS = 512  # the op's limit; larger steps keep the separate launches
# Steps of at most this many rows take a second set of buffers sized for them: fewer reduce
# CTAs reserved, so all of the router's tasks get a CTA each (TP8, 384 experts: 240 router CTAs
# instead of 192 with two tasks each). The same kernel and the same bits.
SMALL_ROWS = 128
BLOCK_M = 16  # the sort block the generic MXFP4 pipeline's tuned rows use
_COMMS: dict[tuple[int, int, int], object] = {}
_SMALL: dict[tuple[int, int, int], object] = {}


def build(hidden: int, experts: int, topk: int, small: bool = False) -> bool:
    """Make the op's buffers for this TP group once; False where it cannot run (an
    aiter without the op or without the generic presort, or an unsupported TP size)."""
    key = (hidden, experts, topk)
    if key in _COMMS:
        return True
    try:
        from aiter.dist.parallel_state import get_tp_group
        from aiter.ops.triton.comms.fused.all_reduce_rmsnorm_moe_route import (
            AllReduceRmsnormMoeRoute,
        )
        from aiter.presorted_moe import match_presorted_moe_generic  # noqa: F401
    except ImportError as e:
        logger.warning(
            "M3-Pro moe_route_glue unavailable (%s); keeping the separate launches", e
        )
        return False
    tp = get_tp_group()
    if tp.world_size not in (2, 4, 8):
        logger.warning(
            "M3-Pro moe_route_glue needs TP 2, 4 or 8, not %d", tp.world_size
        )
        return False
    _COMMS[key] = AllReduceRmsnormMoeRoute(
        tp.cpu_group, MAX_ROWS, hidden, experts, topk, router_fp32=False
    )
    if small:
        _SMALL[key] = AllReduceRmsnormMoeRoute(
            tp.cpu_group, SMALL_ROWS, hidden, experts, topk, router_fp32=False
        )
    logger.info(
        "M3-Pro moe_route_glue: fused all-reduce + norm + router + top-k + sort, "
        "up to %d rows (hidden %d, %d experts, top-%d)",
        MAX_ROWS,
        hidden,
        experts,
        topk,
    )
    return True


def applies(moe, rows: int) -> bool:
    """Whether `route` + `MiniMaxM3MoE.forward_glue` can serve this MoE layer's step."""
    ex = moe.experts
    return (
        rows <= MAX_ROWS
        and (moe.gate.weight.shape[-1], ex.global_num_experts, ex.top_k) in _COMMS
        and moe.shared_experts is not None
        and not moe.fuse_shared_experts
        and ex.scoring_func == "sigmoid"
        and not ex.use_grouped_topk
        and ex.custom_routing_function is None
        and ex.takes_presorted(rows)
    )


@cache
def takes_lookahead() -> bool:
    """Whether this aiter's op also writes a lookahead norm's MXFP8 rows."""
    from aiter.ops.triton.comms.fused import all_reduce_rmsnorm_moe_route as op

    sig = inspect.signature(op.fused_all_reduce_rmsnorm_moe_route)
    return "lookahead_weight" in sig.parameters


def route(x: torch.Tensor, residual: torch.Tensor, norm, moe, lookahead=None) -> dict:
    """The op on this rank's partial sums `x` (the o_proj output) and the residual:
    ``normed`` (bf16, the routed experts' input), ``x_q`` / ``x_scale`` (its MXFP8
    copy, the shared expert's), ``residual_out``, ``topk_ids`` / ``topk_weights`` (the
    weights carry the routed scale where the layer folds it), the BM=16 sort and the
    zeroed ``moe_buf``. ``lookahead`` (`takes_lookahead`): ``(norm, q, q_scale)``,
    ``residual_out`` through that norm as MXFP8 ``la_q`` / ``la_scale`` (into ``q`` /
    ``q_scale`` unless None)."""
    from aiter.ops.triton.comms.fused.all_reduce_rmsnorm_moe_route import (
        fused_all_reduce_rmsnorm_moe_route,
    )

    ex = moe.experts
    key = (x.shape[-1], ex.global_num_experts, ex.top_k)
    comm = _SMALL.get(key) if x.shape[0] <= SMALL_ROWS else None
    comm = _COMMS[key] if comm is None else comm
    la = {}
    if lookahead is not None:
        n, q, s = lookahead
        la = {"lookahead_weight": n.weight, "lookahead_eps": n.variance_epsilon}
        la["out"] = None if q is None else {"la_q": q, "la_scale": s.view(torch.uint8)}
    return fused_all_reduce_rmsnorm_moe_route(
        x,
        residual,
        norm.weight,
        norm.variance_epsilon,
        moe.gate.weight,
        comm,
        correction_bias=moe.e_score_correction_bias,
        scoring="sigmoid",
        renormalize=ex.renormalize,
        routed_scaling_factor=(
            ex.routed_scaling_factor if ex.fold_routed_scaling_factor else 1.0
        ),
        gemma=True,
        quant=True,
        block_size=BLOCK_M,
        zero_moe_buf=True,
        **la,
    )


def presort(r: dict, moe) -> dict:
    """aiter ``register_presorted_moe``'s description of `route`'s sort."""
    ex = moe.experts
    return {
        "sorted_token_ids": r["sorted_ids"],
        "sorted_weights": r["sorted_weights"],
        "sorted_expert_ids": r["sorted_expert_ids"],
        "num_valid_ids": r["num_valid_ids"],
        "topk_ids_ptr": r["topk_ids"].data_ptr(),
        "topk_weights_ptr": r["topk_weights"].data_ptr(),
        "M": r["topk_ids"].shape[0],
        "block_m": BLOCK_M,
        "num_experts": ex.global_num_experts,
        "topk": ex.top_k,
        "covers_all_rows": True,
        "moe_buf_zeroed": True,
        "moe_buf_ptr": r["moe_buf"].data_ptr(),
    }
