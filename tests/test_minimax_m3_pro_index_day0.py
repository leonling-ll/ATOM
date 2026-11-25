# SPDX-License-Identifier: MIT
"""``index_day0``'s host side: ``pro_kernels.day0_index_plan`` (a split prefill's rows cut
into aiter's ``minimax_m3_index_top128`` requests over 64-token pages), ``index_day0``'s
gather and launch, and when ``ProSparseAttentionImpl`` takes the path. The kernel itself is
aiter's (``op_tests/test_flydsl_minimax_m3_pro_indexer.py``); here it is a recorder."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("triton", reason="pro_kernels defines @triton.jit kernels")
pytest.importorskip("aiter", reason="pro_attn imports the ATOM attention base")

from atom.model_ops.minimax_m3 import pro_attn
from atom.model_ops.minimax_m3 import pro_kernels as K
from atom.model_ops.minimax_m3.pro_attn import ProSparseAttentionImpl, ProVariants


def _tables(n_seqs, width):
    # every (sequence, column) its own page id, so a gathered page says where it came from
    return torch.arange(n_seqs * width, dtype=torch.int32).view(n_seqs, width) * 3 + 7


def test_plan_cuts_the_ranges_at_sequence_ends():
    # two sequences of 100 and 200 new rows, the second after 300 cached tokens; this
    # rank scores rows [0, 150) and [250, 300)
    bt = _tables(2, 40)
    pages, table, cu_q, seq_lens, max_q, max_seq, rows = K.day0_index_plan(
        bt, [0, 100, 300], [100, 500], ((0, 150), (250, 300))
    )
    # sequence 0 needs 2 64-token pages (8 16-token ones, the last two past its 7 repeat
    # its last), sequence 1 8 (32 16-token ones), sequence 0's first
    want = [bt[0, min(c, 6)] for c in range(8)] + [bt[1, c] for c in range(32)]
    assert pages.tolist() == [int(p) for p in want]
    assert pages.dtype == torch.long
    # three requests: sequence 0's 100 rows, then sequence 1's [300, 350) and [450, 500)
    assert cu_q.tolist() == [0, 100, 150, 200]
    assert seq_lens.tolist() == [100, 350, 500]
    assert table.tolist() == [
        [0, 1, 0, 0, 0, 0, 0, 0],
        [2, 3, 4, 5, 6, 7, 0, 0],
        [2, 3, 4, 5, 6, 7, 8, 9],
    ]
    assert table.dtype == cu_q.dtype == seq_lens.dtype == torch.int32
    assert (max_q, max_seq, rows) == (100, 500, 200)


def test_plan_of_no_rows_is_none():
    assert K.day0_index_plan(_tables(1, 8), [0, 10], [10], ((10, 20), (20, 20))) is None


def test_index_day0_gathers_four_pages_a_page(monkeypatch):
    seen = {}

    def rec(q, cache, table, cu_q, seq_lens, max_q, max_seq, score, out, *rest):
        seen.update(q=q, cache=cache, score=score, out=out, rest=rest, max_seq=max_seq)
        out.fill_(5)

    monkeypatch.setattr(K, "_day0_index", rec)
    bt = _tables(2, 40)
    plan = K.day0_index_plan(bt, [0, 100, 300], [100, 500], ((0, 150), (250, 300)))
    ic = torch.randn(int(bt.max()) + 1, K.CENTROID_ROWS, K.IDX_DIM).bfloat16()
    q = torch.randn(256, K.IDX_HEADS * K.IDX_DIM).bfloat16()
    out = torch.full((256, K.TOPK), -7, dtype=torch.int32)
    K.index_day0(q, ic, plan, 0.25, 1, 9, out)
    want = ic[plan[0]].view(-1, K.DAY0_INDEX_PAGES * K.CENTROID_ROWS, K.IDX_DIM)
    assert torch.equal(seen["cache"], want)
    assert seen["q"].shape == (200, K.IDX_HEADS, K.IDX_DIM)
    # 500 tokens: 32 blocks, the score buffer a multiple of 16 blocks wide
    assert seen["score"].shape == (200, 32) and seen["score"].dtype == torch.float16
    assert seen["rest"] == (0.25, 1, 9)
    assert (out[:200] == 5).all() and (out[200:] == -7).all()


def test_blocker_without_the_op(monkeypatch):
    monkeypatch.setattr(K, "_day0_index", False)
    K.day0_index_blocker.cache_clear()
    try:
        assert K.day0_index_blocker(1, 9, 1 << 20) == (
            "this aiter has no minimax_m3_index_top128"
        )
    finally:
        K.day0_index_blocker.cache_clear()


def _impl():
    m = object.__new__(ProSparseAttentionImpl)
    m.dcp_world, m.max_model_len = 1, 1 << 20
    return m


ON = {"index_day0": True, "index_compress": True, "head_reduce": "max"}


@pytest.mark.parametrize(
    "kw, dcp, dtype, ok",
    [
        (ON, 1, torch.bfloat16, True),
        ({**ON, "index_day0": False}, 1, torch.bfloat16, False),
        ({**ON, "index_compress": False}, 1, torch.bfloat16, False),
        ({**ON, "head_reduce": "sum"}, 1, torch.bfloat16, False),
        (ON, 2, torch.bfloat16, False),
        (ON, 1, torch.float8_e4m3fn, False),
    ],
)
def test_the_path_needs_its_variants(kw, dcp, dtype, ok, monkeypatch):
    monkeypatch.setattr(K, "day0_index_blocker", lambda *a: None)
    m = _impl()
    m.dcp_world = dcp
    ic = torch.empty(1, K.CENTROID_ROWS, K.IDX_DIM, dtype=dtype)
    assert m._day0_index_ok(ProVariants(**kw), ic, 1, 9) is ok


def test_a_blocked_path_logs_once(monkeypatch):
    monkeypatch.setattr(K, "day0_index_blocker", lambda *a: "needs gfx950")
    monkeypatch.setattr(pro_attn, "_day0_index_warned", set())
    logged = []
    monkeypatch.setattr(pro_attn.logger, "warning", lambda *a: logged.append(a))
    ic = torch.empty(1, K.CENTROID_ROWS, K.IDX_DIM, dtype=torch.bfloat16)
    for _ in range(3):
        assert _impl()._day0_index_ok(ProVariants(**ON), ic, 1, 9) is False
    assert len(logged) == 1 and "needs gfx950" in logged[0]
