# SPDX-License-Identifier: MIT
"""GPU checks of the NVFP4 index scorer: multi-row verify, padded tables, 64-bit paths."""

import pytest
import torch

pytest.importorskip("triton", minversion="3.9")

from atom.model_ops.minimax_m3 import pro_index_nvfp4 as nvfp4

needs_gfx950 = pytest.mark.skipif(not nvfp4.is_supported(), reason="requires gfx950")
SCORERS = [nvfp4.score_bf16q, nvfp4.score]


def _inputs(rng, pages, rows):
    """Random NVFP4 pages with finite E4M3 scales, queries and cq."""
    cache = torch.randint(
        256,
        (pages, nvfp4.page_bytes(nvfp4.PAGE)),
        dtype=torch.uint8,
        device="cuda",
        generator=rng,
    )
    cache[:, nvfp4.PAGE * (nvfp4.IDX_DIM // 2) :].random_(32, 64, generator=rng)
    iq = torch.randn(
        (rows, nvfp4.IDX_HEADS, nvfp4.IDX_DIM),
        device="cuda",
        dtype=torch.bfloat16,
        generator=rng,
    )
    cq = torch.randn(nvfp4.IDX_DIM, device="cuda", dtype=torch.bfloat16, generator=rng)
    return cache, iq, cq


def _two_sequences(rng, width):
    """Two sequences of four query rows each, on pages drawn from one pool."""
    contexts = (width * nvfp4.PAGE - 37, width * nvfp4.PAGE // 2 + 5)
    pages = sum((c + nvfp4.PAGE - 1) // nvfp4.PAGE for c in contexts)
    cache, iq, cq = _inputs(rng, pages, 8)
    perm = torch.randperm(pages, device="cuda", generator=rng).to(torch.int32)
    rows = []
    start = 0
    for c in contexts:
        n = (c + nvfp4.PAGE - 1) // nvfp4.PAGE
        row = torch.zeros(width, dtype=torch.int32, device="cuda")
        row[:n] = perm[start : start + n]
        rows.append(row)
        start += n
    seq_id = torch.tensor([0] * 4 + [1] * 4, dtype=torch.int32, device="cuda")
    pos = torch.cat([torch.arange(c - 4, c) for c in contexts]).to(torch.int32).cuda()
    return cache, iq, cq, rows, seq_id, pos


@needs_gfx950
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("score", SCORERS)
def test_eight_verify_rows_on_a_wide_table(score, causal):
    # The last eight tokens cross a 16-token page boundary and have distinct queries.
    context, width = 250_003, 65_536
    pages = (context + nvfp4.PAGE - 1) // nvfp4.PAGE
    rng = torch.Generator(device="cuda").manual_seed(31)
    cache, iq, cq = _inputs(rng, pages, 8)
    tables = torch.zeros((1, width), dtype=torch.int32, device="cuda")
    tables[0, :pages] = torch.randperm(pages, device="cuda", generator=rng)
    seq_id = torch.zeros(8, device="cuda", dtype=torch.int32)
    pos = torch.arange(context - 8, context, device="cuda", dtype=torch.int32)
    args = (iq, cache, tables, seq_id, pos, width, cq, 1, 9, nvfp4.IDX_DIM**-0.5)

    expected = score(*args, tq=1)
    actual = score(*args, tq=8, causal=causal)
    live = torch.arange(width, device="cuda")[None, :] <= pos[:, None] // nvfp4.PAGE
    torch.testing.assert_close(actual[live], expected[live], rtol=1e-4, atol=1e-4)
    if not causal:
        assert torch.all(actual[~live] == float("-inf"))


@needs_gfx950
@pytest.mark.parametrize("score", SCORERS)
def test_padded_two_row_table(score):
    # Row 1 starts 2**30 entries after row 0: its byte offset exceeds 32 bits.
    stride, width = 2**30, 16
    if torch.cuda.mem_get_info()[0] < 4 * (stride + width) + (1 << 30):
        pytest.skip("needs 5 GiB of free device memory")
    rng = torch.Generator(device="cuda").manual_seed(7)
    cache, iq, cq, rows, seq_id, pos = _two_sequences(rng, width)
    storage = torch.empty(stride + width, dtype=torch.int32, device="cuda")
    padded = storage.as_strided((2, width), (stride, 1))
    padded[0].copy_(rows[0])
    padded[1].copy_(rows[1])
    compact = torch.stack(rows)
    # No pinned blocks: every live score depends on the pages read.
    tail = (width, cq, 0, 0, nvfp4.IDX_DIM**-0.5)
    live = torch.arange(width, device="cuda")[None, :] <= pos[:, None] // nvfp4.PAGE

    for causal in (True, False):
        want = score(iq, cache, compact, seq_id, pos, *tail, tq=4, causal=causal)
        got = score(iq, cache, padded, seq_id, pos, *tail, tq=4, causal=causal)
        mask = live if causal else torch.ones_like(live)
        assert torch.equal(got[mask], want[mask])


@needs_gfx950
@pytest.mark.parametrize("score", SCORERS)
def test_64_bit_addressing_paths(score, monkeypatch):
    # The pointer fallbacks used past _BUFFER_BYTES give the buffer paths' scores.
    width = 64
    rng = torch.Generator(device="cuda").manual_seed(11)
    cache, iq, cq, rows, seq_id, pos = _two_sequences(rng, width)
    tables = torch.stack(rows)
    args = (iq, cache, tables, seq_id, pos, width, cq, 0, 0, nvfp4.IDX_DIM**-0.5)

    for causal in (True, False):
        want = score(*args, tq=4, causal=causal)
        with monkeypatch.context() as patch:
            patch.setattr(nvfp4, "_BUFFER_BYTES", 1)
            got = score(*args, tq=4, causal=causal)
        live = torch.arange(width, device="cuda")[None, :] <= pos[:, None] // nvfp4.PAGE
        mask = live if causal else torch.ones_like(live)
        assert torch.equal(got[mask], want[mask])


@needs_gfx950
def test_cache_pages_past_2_gib():
    # Buffer descriptors cover 0x7ffffffe bytes: a cache just over 2 GiB must take the
    # 64-bit path, or its pages past 2 GiB read as zeros.
    page = nvfp4.page_bytes(nvfp4.PAGE)
    live_pages = 64
    total = 2**31 // page + live_pages + 1
    if torch.cuda.mem_get_info()[0] < total * page + (1 << 30):
        pytest.skip("needs 3 GiB of free device memory")
    rng = torch.Generator(device="cuda").manual_seed(5)
    cache = torch.empty((total, page), dtype=torch.uint8, device="cuda")
    low = torch.arange(1, live_pages + 1, device="cuda")
    high = torch.arange(total - live_pages, total, device="cuda")
    assert cache.numel() >= 2**31 and int(high[0]) * page >= 2**31
    keys, iq, cq = _inputs(rng, live_pages, 8)
    cache[low] = keys
    cache[high] = keys
    cache[0] = 0xFF  # NaN scales on page 0
    context = live_pages * nvfp4.PAGE - 5
    seq_id = torch.zeros(8, device="cuda", dtype=torch.int32)
    pos = torch.arange(context - 8, context, device="cuda", dtype=torch.int32)

    # A 64-block table gives one tile a program; 16384 blocks run the pipelined loop, whose
    # early page-table reads return the garbage ids past the live range.
    for width in (live_pages, 16384):
        tables = []
        for pages in (low, high):
            table = torch.full((1, width), 0x7FFFFFFF, dtype=torch.int32, device="cuda")
            table[0, :live_pages] = pages.to(torch.int32)
            tables.append(table)
        live = torch.arange(width, device="cuda")[None, :] <= pos[:, None] // nvfp4.PAGE
        for tq in (1, 8):
            for score in SCORERS:
                for causal in (True, False):
                    lo, hi = (
                        score(iq, cache, table, seq_id, pos, width, cq, 0, 0,
                              nvfp4.IDX_DIM**-0.5, tq=tq, causal=causal)
                        for table in tables
                    )  # fmt: skip
                    mask = live if causal else torch.ones_like(live)
                    where = (score.__name__, width, tq, causal)
                    assert torch.equal(hi[mask], lo[mask]), where
                    assert torch.isfinite(hi[live]).all(), where
