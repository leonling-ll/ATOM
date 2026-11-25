# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The paged DSpark block's per-step metadata in one Triton launch.

The MiniMax-M3.1 / M3-Pro draft attends its block as ``running_bs * T``
q_len=1 decode rows, each over its request's ``[context | block]`` through a
page-16 table cut to the sliding window. Every drafting round, outside the
captured block, `MiniMaxM3DSpark.build_paged_block` builds that addressing
here: the block's positions and KV write slots, its context lengths, the
allocation cap, and each block row's page-16 table and length. The torch
spelling of the same build (the proposer's non-DCP slots, `AllocationCap.apply`
and a per-request page-16 table cut per row) was about 27 launches of 3-5 us
each, at any batch size; it is the reference in
tests/test_minimax_m3_pro_dspark.py.

Integer arithmetic only, so every value a block row reads is bit-identical to
that reference. Past the end of its own request's table a row's window holds 0,
and its length never reaches there.

Grid: ``(running_bs, cdiv(P, BLOCK_J))``. A request's ``[T, P]`` tables (P =
8192 page-16s at max-model-len 131072, 65536 at ~1M) are split into
``BLOCK_J``-column chunks over grid axis 1, so no program loops over P; chunk 0
also writes the per-request and per-row scalars.

Not covered: the DCP-sharded slots and the MLA block plan, which MiniMax-M3.1
never takes; the proposer keeps its torch chain for the drafts that do.
"""

import torch
import triton
import triton.language as tl

# ASM_PAGE_SIZE in atom/model_ops/minimax_m3/sparse_attn.py; a constexpr, the
# only kind of global a compiled Triton kernel may read.
PAGE16 = tl.constexpr(16)


# Eager, at whatever running_bs a round has: specializing these ints on 1 and
# % 16 would compile again mid-serving as the batch moves.
@triton.jit(do_not_specialize=["scheduled_bs", "pad_ctx_len"])
def _dspark_block_meta_kernel(
    anchor_pos_ptr,  # [running_bs] int64, staged anchors (pad rows 0)
    block_tables_ptr,  # [running_bs, W] int32
    alloc_pages_ptr,  # [running_bs] int32, allocated pages (pad rows 1)
    blk_positions_ptr,  # [running_bs, T] int64 out
    slots_ptr,  # [running_bs * T] int64 out
    ctx_lens_ptr,  # [running_bs] int32 out (attn_metadata.context_lens)
    bt16_rows_ptr,  # [running_bs * T, P] int32 out
    ctx_rows_ptr,  # [running_bs * T] int32 out
    scheduled_bs,
    bt_stride,
    W,
    P,
    window,
    pad_ctx_len,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
    E: tl.constexpr,  # tokens per block-table entry (16 Pro, 128 M3.1)
    PAGES: tl.constexpr,  # E // 16
    BLOCK_J: tl.constexpr,
):
    b = tl.program_id(0)
    c = tl.program_id(1)
    real = b < scheduled_bs

    anchor = tl.load(anchor_pos_ptr + b).to(tl.int64)
    t = tl.arange(0, BLOCK_T)
    tmask = t < T
    bp = anchor + 1 + t.to(tl.int64)  # block positions

    # Slots: page_id * E + offset; a position the table cannot address, a pad
    # row and a position past the allocation are dropped (-1).
    page_idx = bp // E
    in_table = page_idx < W
    page_idx = tl.minimum(page_idx, W - 1)
    page = tl.load(block_tables_ptr + b * bt_stride + page_idx, mask=tmask, other=0)
    slot = page.to(tl.int64) * E + bp % E
    slot = tl.where(in_table, slot, -1)
    slot = tl.where(real, slot, -1)
    alloc = tl.load(alloc_pages_ptr + b).to(tl.int32) * E
    slot = tl.where(bp >= alloc, -1, slot)

    ctx_len = tl.where(real, (anchor + 1 + T).to(tl.int32), pad_ctx_len)
    ctx_eff = tl.minimum(ctx_len, alloc)

    # Each block row's window: its first page-16 and how many keys it reads.
    if window > 0:
        start16 = tl.maximum(bp - (window - 1), 0) // PAGE16
        start16 = tl.where(real, start16, 0)
    else:
        start16 = tl.zeros([BLOCK_T], dtype=tl.int64)
    out_len = ctx_eff - (start16 * PAGE16).to(tl.int32)

    if c == 0:
        tl.store(blk_positions_ptr + b * T + t, bp, mask=tmask)
        tl.store(slots_ptr + b * T + t, slot, mask=tmask)
        tl.store(ctx_lens_ptr + b, ctx_len)
        tl.store(ctx_rows_ptr + b * T + t, out_len, mask=tmask)

    # Entry j of row (b, t)'s table is page-16 (start16 + j) of request b's
    # table. A pad row reads page 0: its first PAGES entries are sub-pages
    # 0..PAGES-1.
    jj = c * BLOCK_J + tl.arange(0, BLOCK_J)
    col = jj < P
    for tt in tl.static_range(T):
        s = tl.sum(tl.where(t == tt, start16, 0), axis=0)
        e = s + jj
        own = e < P
        ent = tl.load(
            block_tables_ptr + b * bt_stride + e // PAGES, mask=own & col, other=0
        )
        val = ent * PAGES + (e % PAGES).to(tl.int32)
        val = tl.where((b >= scheduled_bs) & (e < PAGES), e.to(tl.int32), val)
        val = tl.where(own, val, 0)  # past the request's own table: never read
        row_base = bt16_rows_ptr + (b * T + tt).to(tl.int64) * P
        tl.store(row_base + jj, val, mask=col)


def dspark_block_meta(
    anchor_positions: torch.Tensor,  # [running_bs] int64
    block_tables: torch.Tensor,  # [running_bs, W] int32
    alloc_pages: torch.Tensor,  # [running_bs] int32
    blk_positions: torch.Tensor,  # [running_bs, T] int64 out
    slots: torch.Tensor,  # [running_bs * T] int64 out
    ctx_lens: torch.Tensor,  # [running_bs] int32 out
    bt16_rows: torch.Tensor,  # [running_bs * T, P] int32 out
    ctx_rows: torch.Tensor,  # [running_bs * T] int32 out
    *,
    scheduled_bs: int,
    entry_tokens: int,
    window: int,
    pad_ctx_len: int,
) -> None:
    """The block's positions, slots and context lengths, and each block row's
    windowed page-16 table and allocation-capped length, in one launch. Rows past
    ``scheduled_bs`` are pad rows: anchored at 0, dropped writes, a context of
    ``pad_ctx_len`` keys on page 0."""
    running_bs, T = blk_positions.shape
    W = block_tables.shape[1]
    pages = entry_tokens // PAGE16.value
    P = pages * W
    assert bt16_rows.shape[1] == P, (bt16_rows.shape, P)
    block_j = min(1024, triton.next_power_of_2(P))
    _dspark_block_meta_kernel[(running_bs, triton.cdiv(P, block_j))](
        anchor_positions,
        block_tables,
        alloc_pages,
        blk_positions,
        slots,
        ctx_lens,
        bt16_rows,
        ctx_rows,
        scheduled_bs,
        block_tables.stride(0),
        W,
        P,
        window,
        pad_ctx_len,
        T=T,
        BLOCK_T=triton.next_power_of_2(T),
        E=entry_tokens,
        PAGES=pages,
        BLOCK_J=block_j,
        num_warps=4,
    )
