# SPDX-License-Identifier: MIT
"""The indexer diagnostic (``pro_diag``) on synthetic data, on a CPU.

What is pinned down is the bookkeeping the analysis relies on, not any reading of the
indexer: every reading's captured dense-attention mass lies between the forced-blocks-only
floor and the oracle (top-k by the true mass), the record is complete, and a reading whose
index vectors are built to point at the blocks the dense attention uses scores far above
one that does not.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="pro_attn imports the ATOM attention base")

from atom.model_ops.minimax_m3 import pro_diag
from atom.model_ops.minimax_m3.pro_attn import (
    ProSparseAttentionImpl,
    ProVariants,
)

H, D, NI, DI, PG = 2, 32, 2, 16, 16
N = pro_diag._MIN_POS + 200  # one chunk with 4296 tokens = 269 blocks > topk


def _impl(topk=12):
    m = object.__new__(ProSparseAttentionImpl)
    m.page, m.topk, m.init_blocks, m.local_blocks = PG, topk, 1, 2
    m.num_heads, m.head_dim, m.rotary_dim = H, D, 8
    m.num_idx_heads, m.index_head_dim, m.value_head_dim = NI, DI, D - 8
    m.scale = D**-0.5
    m.layer_num = 3
    g = torch.Generator().manual_seed(1)
    m.__dict__["index_compress_query"] = SimpleNamespace(
        data=torch.randn(DI, generator=g)
    )
    ones = lambda n: SimpleNamespace(
        weight=SimpleNamespace(data=torch.zeros(n)), variance_epsilon=1e-6
    )
    m.index_q_norm, m.index_k_norm = ones(DI), ones(DI)
    maxpos = N + 8
    ang = torch.arange(maxpos)[:, None] * (0.01 ** (torch.arange(4) / 4.0))[None, :]
    m.rotary_emb = SimpleNamespace(
        cos_cache=ang.cos()[:, None, :], sin_cache=ang.sin()[:, None, :], mscale=1.0
    )
    return m


@pytest.fixture
def one_tp(monkeypatch):
    monkeypatch.setattr(pro_diag, "_tp_allreduce", lambda x: x)
    monkeypatch.setattr(pro_diag, "_tp_rank_size", lambda: (0, 1))


def _inputs(seed=0, planted=None):
    """Random q / k; ``planted`` = block ids the dense attention is made to concentrate on
    (every query is given a key direction there), with the raw index vectors random."""
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(N, 1, D, generator=g)
    q = torch.randn(N, H, D, generator=g) * 0.3
    if planted is not None:
        direction = torch.randn(D, generator=g)
        direction = direction / direction.norm()
        for b in planted:
            k[b * PG : (b + 1) * PG, 0] = direction * 12.0
        q[:] = direction * 3.0 + q * 0.2
    raw_iq = torch.randn(N, NI * DI, generator=g)
    raw_ik = torch.randn(N, DI, generator=g)
    raw = {name: (raw_iq, raw_ik) for name in ("own", "ffn", "res", "resn")}
    return q, k, raw


def _run(tmp_path, monkeypatch, q, k, raw, tag="t", nq="6"):
    monkeypatch.setenv("ATOM_M3PRO_DIAG_DIR", str(tmp_path))
    monkeypatch.setenv("ATOM_M3PRO_DIAG_Q", nq)
    impl = _impl()
    v = ProVariants(diag=True, diag_tag=tag, dense_only=True)
    pro_diag.collect(impl, q, k, torch.arange(N), raw, [0, N], [N], v)
    path = tmp_path / f"{tag}.jsonl"
    assert path.exists()
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_record_is_complete_and_bounded(tmp_path, monkeypatch, one_tp):
    q, k, raw = _inputs()
    (rec,) = _run(tmp_path, monkeypatch, q, k, raw)
    assert rec["layer"] == 3 and rec["n"] == N and len(rec["pos"]) == 6
    assert min(rec["pos"]) >= pro_diag._MIN_POS and max(rec["pos"]) == N - 1
    # sources x rope x relu x (block scores x 4 head reductions)
    assert (
        len(rec["combos"])
        == len(pro_diag.SOURCES) * 2 * 2 * len(pro_diag.BLOCK_SCORES) * 4
    )
    eps = 1e-4
    for i in range(6):
        floor, oracle = rec["forced"][i], rec["oracle"][i]
        assert 0 <= floor <= oracle <= 1 + eps
        assert rec["random"][i] <= oracle + eps
        for name, mass in rec["combos"].items():
            assert floor - eps <= mass[i] <= oracle + eps, name


def test_tighter_budgets_and_rank_correlation(tmp_path, monkeypatch, one_tp):
    """Every reading is also scored at the tighter top-k budgets (their own oracle), and by the
    rank correlation of its block scores with the dense mass."""
    q, k, raw = _inputs(seed=5)
    (rec,) = _run(tmp_path, monkeypatch, q, k, raw)  # topk 12, budgets 16 / 32 / 64
    assert sorted(rec["budgets"], key=int) == ["16", "32", "64"]
    eps = 1e-4
    sane = [
        name for name in rec["combos"] if name.endswith("relu=0")
    ]  # relu ties break nesting
    for i in range(6):
        floor = rec["forced"][i]
        prev_oracle = rec["oracle"][i]
        prev = {name: rec["combos"][name][i] for name in sane}
        for b in ("16", "32", "64"):
            r = rec["budgets"][b]
            assert prev_oracle - eps <= r["oracle"][i] <= 1 + eps
            assert r["random"][i] <= r["oracle"][i] + eps
            for name, mass in r["combos"].items():
                assert floor - eps <= mass[i] <= r["oracle"][i] + eps, (b, name)
            for name in sane:  # a larger budget keeps the smaller one's blocks
                assert r["combos"][name][i] >= prev[name] - eps, (b, name)
                prev[name] = r["combos"][name][i]
            prev_oracle = r["oracle"][i]
    assert set(rec["rho"]) == set(rec["combos"])
    for name, rho in rec["rho"].items():
        assert len(rho) == 6 and all(-1 - eps <= x <= 1 + eps for x in rho), name
    # random index vectors do not predict the attention: correlations scatter around 0
    mean_rho = sum(sum(v) for v in rec["rho"].values()) / (6 * len(rec["rho"]))
    assert abs(mean_rho) < 0.2


def test_calibration_kl_is_recorded_for_the_sane_corner(tmp_path, monkeypatch, one_tp):
    q, k, raw = _inputs(seed=7)
    (rec,) = _run(tmp_path, monkeypatch, q, k, raw)
    betas = rec["kl_betas"]
    assert betas == pro_diag.KL_BETAS
    assert (
        len(rec["kl"]) == len(pro_diag.SOURCES) * 2 * len(pro_diag.BLOCK_SCORES) * 4
    )  # relu off only
    assert all(name.endswith("|relu=0") for name in rec["kl"])
    for name, rows in rec["kl"].items():
        assert len(rows) == len(betas) and all(len(r) == 6 for r in rows), name
        assert all(x >= -1e-3 for r in rows for x in r), name
    assert len(rec["kl_uniform"]) == 6 and all(x >= -1e-3 for x in rec["kl_uniform"])


def test_grid_narrows_the_readings(tmp_path, monkeypatch, one_tp):
    q, k, raw = _inputs(seed=8)
    monkeypatch.setenv("ATOM_M3PRO_DIAG_DIR", str(tmp_path))
    monkeypatch.setenv("ATOM_M3PRO_DIAG_Q", "4")
    grid = {
        "sources": {"a/b": ("ffn", "own")},
        "ropes": (True,),
        "relus": (False,),
        "blocks": ["toklse", "pool4max"],
        "reduces": ["psum", "sum"],
        "kl": False,
    }
    v = ProVariants(diag=True, diag_tag="grid", dense_only=True)
    pro_diag.collect(_impl(), q, k, torch.arange(N), raw, [0, N], [N], v, grid=grid)
    (rec,) = [
        json.loads(line) for line in (tmp_path / "grid.jsonl").read_text().splitlines()
    ]
    assert sorted(rec["combos"]) == sorted(
        f"a/b|{b}|{r}|rope=1|relu=0" for b in grid["blocks"] for r in grid["reduces"]
    )
    assert "kl" not in rec and set(rec["rho"]) == set(rec["combos"])


def test_a_calibrated_attention_predictor_has_zero_kl_at_beta_one(
    tmp_path, monkeypatch, one_tp
):
    """Index vectors built so that the token-level softmax over the index logits *is* the dense
    attention of the matching main head: the toklse + psum reading must be exactly calibrated,
    KL ~ 0 at beta = 1 and clearly positive at every other beta."""
    g = torch.Generator().manual_seed(11)

    def unit_rms(x):
        return (
            x / x.pow(2).mean(-1, keepdim=True).sqrt()
        )  # the index norm leaves these alone

    iq = unit_rms(torch.randn(N, NI, DI, generator=g))
    ik = unit_rms(torch.randn(N, DI, generator=g))
    # index logits = iq.ik * DI**-0.5; main logits = q.k * D**-0.5 with q = lam * (iq, 0), k = lam * (ik, 0)
    lam = (D**0.5 / DI**0.5) ** 0.5
    q = torch.zeros(N, H, D)
    k = torch.zeros(N, 1, D)
    q[..., :DI] = lam * iq  # NI == H: one index head per main head
    k[:, 0, :DI] = lam * ik
    raw = {"own": (iq.reshape(N, NI * DI), ik)}
    monkeypatch.setenv("ATOM_M3PRO_DIAG_DIR", str(tmp_path))
    monkeypatch.setenv("ATOM_M3PRO_DIAG_Q", "6")
    v = ProVariants(diag=True, diag_tag="calib", dense_only=True)
    grid = {
        "sources": {"own/own": ("own", "own")},
        "ropes": (False,),
        "relus": (False,),
        "blocks": ["toklse", "tokmax"],
        "reduces": ["psum"],
    }
    pro_diag.collect(_impl(), q, k, torch.arange(N), raw, [0, N], [N], v, grid=grid)
    (rec,) = [
        json.loads(x) for x in (tmp_path / "calib.jsonl").read_text().splitlines()
    ]
    one = rec["kl_betas"].index(1.0)
    rows = rec["kl"]["own/own|toklse|psum|rope=0|relu=0"]
    assert max(abs(x) for x in rows[one]) < 2e-3, rows[one]
    for i, row in enumerate(rows):
        if i != one:
            assert min(row) > 5e-3, (rec["kl_betas"][i], row)
    # a hard max over the tokens is not that predictor
    assert min(rec["kl"]["own/own|tokmax|psum|rope=0|relu=0"][one]) > 5e-3
    assert all(0 < x for x in rec["kl_uniform"])


def test_parametric_pooled_readings(tmp_path, monkeypatch, one_tp):
    """``pool<agg>@<temp>`` is the pooled reading with its own temperature: with the default
    temperature and consecutive groups it is exactly the named ``pool4<agg>``; strided groups
    pool different tokens; unknown names are rejected."""
    q, k, raw = _inputs(seed=12)
    monkeypatch.setenv("ATOM_M3PRO_DIAG_DIR", str(tmp_path))
    monkeypatch.setenv("ATOM_M3PRO_DIAG_Q", "4")
    grid = {
        "sources": {"s": ("own", "own")},
        "ropes": (False,),
        "relus": (False,),
        "kl": False,
        "blocks": [
            "pool4max",
            "poolmax@1.0",
            "poolSmax@1.0",
            "pool4lse_t",
            "poollse@0.08838834764831845",
        ],
        "reduces": ["sum"],
    }
    v = ProVariants(diag=True, diag_tag="pooled", dense_only=True)
    pro_diag.collect(_impl(), q, k, torch.arange(N), raw, [0, N], [N], v, grid=grid)
    (rec,) = [
        json.loads(x) for x in (tmp_path / "pooled.jsonl").read_text().splitlines()
    ]
    c = rec["combos"]
    assert c["s|pool4max|sum|rope=0|relu=0"] == c["s|poolmax@1.0|sum|rope=0|relu=0"]
    assert (
        rec["rho"]["s|pool4max|sum|rope=0|relu=0"]
        == rec["rho"]["s|poolmax@1.0|sum|rope=0|relu=0"]
    )
    assert (
        rec["rho"]["s|poolSmax@1.0|sum|rope=0|relu=0"]
        != rec["rho"]["s|pool4max|sum|rope=0|relu=0"]
    )
    assert (
        rec["rho"]["s|pool4lse_t|sum|rope=0|relu=0"]
        == rec["rho"]["s|poollse@0.08838834764831845|sum|rope=0|relu=0"]
    )
    with pytest.raises(ValueError):
        pro_diag._block_spec("nonsense")


def test_budget_list_comes_from_the_environment(tmp_path, monkeypatch, one_tp):
    monkeypatch.setenv("ATOM_M3PRO_DIAG_BUDGETS", "8, 12,20")
    q, k, raw = _inputs(seed=6)
    (rec,) = _run(
        tmp_path, monkeypatch, q, k, raw
    )  # topk is 12: the main budget is not repeated
    assert sorted(rec["budgets"], key=int) == ["8", "20"]
    assert pro_diag._budgets(12) == [8, 20]


def test_oracle_is_the_best_possible_pick(tmp_path, monkeypatch, one_tp):
    q, k, raw = _inputs(seed=2)
    (rec,) = _run(tmp_path, monkeypatch, q, k, raw)
    best = max(max(m[i] for m in rec["combos"].values()) for i in range(6))
    assert best <= max(rec["oracle"]) + 1e-4


def test_a_reading_that_points_at_the_attended_blocks_beats_a_blind_one(
    tmp_path, monkeypatch, one_tp
):
    """Plant dense attention on 8 far-away blocks; give those blocks' index keys one shared
    direction and every query's index query the same: the token-max reading must capture
    (nearly) all of the oracle's mass, while random index vectors capture about the floor.
    """
    planted = [20, 40, 60, 80, 100, 120, 140, 160]
    q, k, raw = _inputs(seed=3, planted=planted)
    (blind,) = _run(tmp_path, monkeypatch, q, k, raw, tag="blind")
    g = torch.Generator().manual_seed(9)
    d = torch.randn(DI, generator=g)
    raw_ik = torch.randn(N, DI, generator=g) * 0.2
    for b in planted:
        raw_ik[b * PG : (b + 1) * PG] = d * 3.0
    raw_iq = (d * 3.0).repeat(NI).expand(N, -1).clone()
    sharp = {"own": (raw_iq, raw_ik), "ffn": (raw_iq, raw_ik), "res": (raw_iq, raw_ik)}
    (pointed,) = _run(tmp_path, monkeypatch, q, k, sharp, tag="pointed")
    key = "own/own|tokmax|sum|rope=0|relu=0"
    gain = lambda rec, i: (rec["combos"][key][i] - rec["forced"][i]) / max(
        rec["oracle"][i] - rec["forced"][i], 1e-9
    )
    for i in range(6):
        assert gain(pointed, i) > 0.9, (i, gain(pointed, i))
    assert sum(gain(blind, i) for i in range(6)) / 6 < 0.5


def test_too_short_a_prefill_is_skipped(tmp_path, monkeypatch, one_tp):
    q, k, raw = _inputs()
    monkeypatch.setenv("ATOM_M3PRO_DIAG_DIR", str(tmp_path))
    v = ProVariants(diag=True, diag_tag="short", dense_only=True)
    n = pro_diag._MIN_POS  # not enough blocks for a real selection
    pro_diag.collect(_impl(), q[:n], k[:n], torch.arange(n), raw, [0, n], [n], v)
    assert not os.path.exists(tmp_path / "short.jsonl")
