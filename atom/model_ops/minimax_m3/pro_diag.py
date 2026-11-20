# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Developer diagnostic for MiniMax-M3-Pro's lightning indexer.

The checkpoint does not say how the 16 index heads' scores of a 16-token block become a
block score, where a lookahead layer's indexer input comes from, or what the softmax-pool
centroids do. A trained indexer is a predictor of the dense attention, so each reading is
scored here by how much of the dense attention it selects:

    ATOM_M3PRO_VARIANTS='{"diag": true, "diag_tag": "code"}'

makes the model run *dense* attention (a trajectory no reading of the indexer can bend),
and on ``Q`` queries sampled from a long prefill (positions >= 4096, so more than 128
blocks are visible) computes, per layer:

* ``mbar`` -- the dense attention mass of every 16-token block, averaged over all 128 heads
  (the heads' sums are all-reduced over the TP group);
* for each reading, the mass of the blocks it selects (top-128 with the forced init /
  local blocks), plus the references: ``forced`` (init + local blocks only), ``oracle``
  (top-128 by ``mbar``) and ``random`` (the expectation of a random pick).

Besides the real top-128 budget every reading is also scored at tighter budgets
(``ATOM_M3PRO_DIAG_BUDGETS``, default 16, 32, 64 -> the ``budgets`` record), where readings
that all look alike at 128 separate, by the rank correlation of its block scores with
``mbar`` over the selectable blocks (``rho``; a sharp signal: the right reading of a trained
indexer sits near 0.99 where wrong ones are far below), and, with relu off, by how
well ``softmax(beta * score)`` predicts ``mbar`` (``kl``, one row per beta in ``KL_BETAS``;
a reading that is a calibrated attention predictor has its smallest KL at beta = 1).

One JSON line per layer goes to ``$ATOM_M3PRO_DIAG_DIR/<diag_tag>.jsonl`` (rank 0; the
directory defaults to ``m3pro_diag`` under the system temp directory).
The prompt must be one prefill chunk (``--max-num-batched-tokens`` >= its length).
"""

import itertools
import json
import logging
import os
import re
import tempfile

import torch

from atom.model_ops.layernorm import GemmaRMSNorm

logger = logging.getLogger("atom")

_NEG = float("-inf")
_MIN_POS = 4096  # sampled queries start here: 256+ blocks visible, selection is real

# (query source, key source): "own" the layer's own attention input, "ffn" / "res" / "resn" the
# FFN-norm output / the residual stream / the residual put through the reading layer's
# input_layernorm, of the layer two back (layers < 2: own).
SOURCES = {
    "own/own": ("own", "own"),
    "ffn/ffn": ("ffn", "ffn"),
    "ffn/own": ("ffn", "own"),
    "res/res": ("res", "res"),
    "res/own": ("res", "own"),
    "resn/resn": ("resn", "resn"),
    "resn/own": ("resn", "own"),
}
# (name, family, pooling-softmax temperature). tok*: over the 16 tokens of a block; pool4*: over
# the 4 softmax-pooled centroids (groups of 4 consecutive tokens), "_t" = pooling logits scaled
# by 128**-0.5.
BLOCK_SCORES = [
    ("tokmax", "tok", None),
    ("tokmean", "tok", None),
    ("toklse", "tok", None),
    ("pool4max", "pool", 1.0),
    ("pool4max_t", "pool", 128**-0.5),
    ("pool4mean", "pool", 1.0),
    ("pool4mean_t", "pool", 128**-0.5),
    ("pool4lse", "pool", 1.0),
    ("pool4lse_t", "pool", 128**-0.5),
]
HEAD_REDUCES = ["sum", "max", "lse", "psum"]
# calibration: how well does softmax(beta * score) predict the dense block mass (KL, nats)?
KL_BETAS = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0]
# parametric pooled readings, e.g. "poolSlse@0.5": pool (consecutive groups of 4) or poolS (strided
# groups), the aggregate of the 4 centroid scores (max / lse / mean), the pooling-softmax temperature
_POOL_NAME = re.compile(r"(pool|poolS)(max|lse|mean)@([0-9.]+)")


def _block_spec(name: str):
    """(name, family, pooling temperature, centroid aggregate) of a block-score name."""
    for bname, fam, temp in BLOCK_SCORES:
        if bname == name:
            agg = (
                "max"
                if name.startswith("pool4max")
                else "lse" if name.startswith("pool4lse") else "mean"
            )
            return bname, fam, temp, agg
    m = _POOL_NAME.fullmatch(name)
    if m:
        return name, m.group(1), float(m.group(3)), m.group(2)
    raise ValueError(f"unknown block score {name!r}")


def _budgets(topk: int) -> list[int]:
    """Extra top-k budgets (``ATOM_M3PRO_DIAG_BUDGETS``, default ``16,32,64``) besides ``topk``."""
    raw = os.environ.get("ATOM_M3PRO_DIAG_BUDGETS", "16,32,64")
    return sorted({int(x) for x in raw.split(",") if x.strip()} - {topk})


def _tp_allreduce(x: torch.Tensor) -> torch.Tensor:
    from aiter.dist.parallel_state import get_tp_group

    g = get_tp_group()
    if g.world_size > 1:
        torch.distributed.all_reduce(x, group=g.device_group)
    return x


def _tp_rank_size() -> tuple[int, int]:
    from aiter.dist.parallel_state import get_tp_group

    g = get_tp_group()
    return g.rank_in_group, g.world_size


def _reduce(bs: torch.Tensor, how: str, valid: torch.Tensor) -> torch.Tensor:
    """``[Q, NI, NB]`` per-head block scores -> ``[Q, NB]``."""
    if how == "sum":
        return bs.sum(1)
    if how == "max":
        return bs.amax(1)
    if how == "lse":
        return torch.logsumexp(bs, dim=1)
    if how == "psum":  # every head's softmax over the visible blocks, summed
        masked = bs.masked_fill(~valid.unsqueeze(1), _NEG)
        return torch.softmax(masked, dim=-1).sum(1)
    raise ValueError(how)


def _predict(bs: torch.Tensor, how: str, valid: torch.Tensor) -> torch.Tensor:
    """The distribution over blocks a reading predicts: ``[Q, NI, NB]`` per-head block logits
    -> ``[Q, NB]`` summing to 1 over the visible blocks. psum: every head's own softmax,
    averaged (the dense attention's head average); lse / sum / max: one softmax of the reduced
    logits."""
    x = bs.masked_fill(~valid.unsqueeze(1), _NEG)
    if how == "psum":
        return torch.softmax(x, dim=-1).mean(1)
    if how == "lse":
        return torch.softmax(torch.logsumexp(x, dim=1), dim=-1)
    if how == "sum":
        return torch.softmax(x.sum(1), dim=-1)
    if how == "max":
        return torch.softmax(x.amax(1), dim=-1)
    raise ValueError(how)


def _kl(mbar: torch.Tensor, p: torch.Tensor) -> list[float]:
    """KL(mbar || p) per query, in nats."""
    ok = mbar > 0
    lm = torch.where(ok, mbar, torch.ones_like(mbar)).log()
    lp = p.clamp(min=1e-30).log()
    kl = torch.where(ok, mbar * (lm - lp), torch.zeros_like(mbar)).sum(-1)
    return [round(float(x), 4) for x in kl.tolist()]


@torch.no_grad()
def collect(impl, q, k, positions, raw, cu, ctx, v, grid=None) -> None:
    """Score every reading of the indexer on one prefill (see the module doc).

    ``grid`` narrows or widens the readings: ``sources`` (name -> (query source, key source), the
    sources being keys of ``raw``), ``ropes``, ``relus``, ``blocks`` (names of BLOCK_SCORES),
    ``reduces``, and ``kl`` (also record the calibration KL, relu off only)."""
    if raw is None or len(ctx) != 1:
        return
    grid = grid or {}
    sources = grid.get("sources", SOURCES)
    ropes = tuple(grid.get("ropes", (True, False)))
    relus = tuple(grid.get("relus", (False, True)))
    blocks = [
        _block_spec(b) for b in (grid.get("blocks") or [b[0] for b in BLOCK_SCORES])
    ]
    reduces = list(grid.get("reduces", HEAD_REDUCES))
    kl_on = bool(grid.get("kl", True))
    n = int(cu[1] - cu[0])
    if n != ctx[0] or n <= _MIN_POS + 8:
        return  # one chunk only: the keys of an earlier chunk are not kept per reading
    dev = q.device
    page = impl.page
    nq = int(os.environ.get("ATOM_M3PRO_DIAG_Q", "32"))
    nq = min(nq, n - _MIN_POS)
    pos = torch.linspace(_MIN_POS, n - 1, nq, device=dev).round().long()
    nb = (n + page - 1) // page
    lpad = nb * page
    rank, tp = _tp_rank_size()
    total_heads = impl.num_heads * tp

    # ---- the dense attention mass of every block, over all heads -------------------
    kf = k[:n, 0].float()  # [n, D]
    qs = q[pos].float()  # [Q, h, D]
    s = torch.einsum("qhd,ld->qhl", qs, kf) * (impl.scale * v.scale_mult)
    j_tok = torch.arange(n, device=dev)
    s = s.masked_fill(j_tok[None, None, :] > pos[:, None, None], _NEG)
    p = torch.softmax(s, dim=-1)
    del s
    p = torch.nn.functional.pad(p, (0, lpad - n))
    m = p.view(nq, -1, nb, page).sum(-1).sum(1)  # [Q, NB] summed over local heads
    del p
    _tp_allreduce(m)
    mbar = m / total_heads  # sums to 1 over the visible blocks

    qb = pos // page
    jb = torch.arange(nb, device=dev)[None, :]
    valid = jb <= qb[:, None]
    init, local = impl.init_blocks, impl.local_blocks
    is_init = valid & (jb < init)
    is_local = valid & (jb >= qb[:, None] + 1 - local)
    forced = is_init | is_local
    nforced = forced.sum(-1)
    forced_mass = (mbar * forced).sum(-1)
    nvalid = valid.sum(-1)
    k_free = (torch.clamp(nvalid, max=impl.topk) - nforced).clamp(min=0)
    n_free = (nvalid - nforced).clamp(min=1)
    random_mass = forced_mass + (1.0 - forced_mass) * k_free / n_free
    oracle = impl._select_from_score(mbar.clone(), qb, init, local)
    oracle_mass = (mbar * oracle).sum(-1)
    # tighter budgets than the real top-128: with 128 blocks the selection saturates (any sane
    # reading captures >94% of the oracle's headroom), so the smaller budgets are what tells
    # the readings apart.
    budgets = _budgets(impl.topk)
    ref_b: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for b in budgets:
        k_free_b = (torch.clamp(nvalid, max=b) - nforced).clamp(min=0)
        random_b = forced_mass + (1.0 - forced_mass) * k_free_b / n_free
        sel_b = impl._select_from_score(mbar.clone(), qb, init, local, topk=b)
        ref_b[b] = (random_b, (mbar * sel_b).sum(-1))

    # rank correlation of a block score with the dense mass, over the blocks the selection can
    # still choose (valid, neither init nor local)
    free = valid & ~forced
    n_free_t = free.sum(-1, keepdim=True).float().clamp(min=1.0)

    def _ranks(x: torch.Tensor) -> torch.Tensor:
        r = x.masked_fill(~free, _NEG).argsort(-1).argsort(-1).float()
        r = r - (~free).sum(-1, keepdim=True).float()
        return r.masked_fill(~free, 0.0)

    def _center(r: torch.Tensor) -> torch.Tensor:
        return (r - r.sum(-1, keepdim=True) / n_free_t).masked_fill(~free, 0.0)

    rm_c = _center(_ranks(mbar))
    rm_n = rm_c.pow(2).sum(-1).sqrt().clamp(min=1e-9)

    def rank_corr(score: torch.Tensor) -> list[float]:
        rs = _center(_ranks(score))
        rho = (rs * rm_c).sum(-1) / (rs.pow(2).sum(-1).sqrt().clamp(min=1e-9) * rm_n)
        return [round(float(x), 4) for x in rho.tolist()]

    # ---- the index vectors of every source, normed and roped -----------------------
    cos, sin = impl._cos_sin(positions[:n], v)
    eps = impl.index_q_norm.variance_epsilon
    ni, di = impl.num_idx_heads, impl.index_head_dim

    def prep(src: str, rope: bool):
        iq_raw, ik_raw = raw.get(src, raw["own"])
        iq = iq_raw[pos].reshape(nq, ni, di)
        ik = ik_raw[:n].reshape(n, 1, di)
        iq = GemmaRMSNorm.forward_static(impl.index_q_norm.weight.data, eps, iq, None)
        ik = GemmaRMSNorm.forward_static(impl.index_k_norm.weight.data, eps, ik, None)
        if rope:
            iq = impl._rope(iq, cos[pos], sin[pos], v.rope_style)
            ik = impl._rope(ik, cos, sin, v.rope_style)
        return iq.float(), ik[:, 0].float()

    cache: dict = {}

    def get(src, rope):
        if (src, rope) not in cache:
            cache[(src, rope)] = prep(src, rope)
        return cache[(src, rope)]

    cq = impl.index_compress_query.data.float()
    sc = di**-0.5
    fut = j_tok[None, None, :] > pos[:, None, None]
    counts = (
        (pos[:, None] + 1 - jb * page).clamp(0, page).float()
    )  # visible tokens/block
    out: dict[str, list[float]] = {}
    out_b: dict[int, dict[str, list[float]]] = {b: {} for b in budgets}
    rho_out: dict[str, list[float]] = {}
    kl_out: dict[str, list[list[float]]] = {}

    def capture(score: torch.Tensor, topk: int | None = None) -> list[float]:
        sel = impl._select_from_score(score, qb, init, local, topk=topk)
        return [round(float(x), 5) for x in (mbar * sel).sum(-1).tolist()]

    for (sname, (qsrc, ksrc)), rope, relu in itertools.product(
        sources.items(), ropes, relus
    ):
        iq, _ = get(qsrc, rope)
        _, ik = get(ksrc, rope)
        dots = torch.einsum("qnd,ld->qnl", iq, ik) * sc  # [Q, NI, n]
        if relu:
            dots = torch.relu(dots)
        dm = dots.masked_fill(fut, _NEG)
        dm = torch.nn.functional.pad(dm, (0, lpad - n), value=_NEG)
        tok = {}
        for bname, fam, temp, agg in blocks:
            if fam == "tok":
                if bname == "tokmax":
                    bs = dm.view(nq, ni, nb, page).amax(-1)
                elif bname == "tokmean":
                    z = dm.masked_fill(dm == _NEG, 0.0).view(nq, ni, nb, page).sum(-1)
                    bs = z / counts.clamp(min=1).unsqueeze(1)
                else:  # toklse
                    bs = torch.logsumexp(dm.view(nq, ni, nb, page), dim=-1)
            else:
                if "cent" not in tok:
                    tok["cent"] = {}
                if (fam, temp) not in tok["cent"]:
                    ikp = torch.nn.functional.pad(ik, (0, 0, 0, lpad - n))
                    g = ikp.view(
                        nb, 4, 4, di
                    )  # [block, group, token in group]: token 4 * group + c
                    if (
                        fam == "poolS"
                    ):  # strided groups: tokens c, c+4, c+8, c+12 form centroid c
                        g = g.transpose(1, 2)
                    g = g.reshape(nb * 4, 4, di)  # one row of 4 tokens per centroid
                    w = torch.softmax((g * cq).sum(-1) * temp, dim=-1)
                    tok["cent"][(fam, temp)] = (w.unsqueeze(-1) * g).sum(
                        -2
                    )  # [NB*4, DI]
                dc = torch.einsum("qnd,cd->qnc", iq, tok["cent"][(fam, temp)]) * sc
                if relu:
                    dc = torch.relu(dc)
                dc = dc.view(nq, ni, nb, page // 4)
                if agg == "max":
                    bs = dc.amax(-1)
                elif agg == "lse":
                    bs = torch.logsumexp(dc, dim=-1)
                else:
                    bs = dc.mean(-1)
            for red in reduces:
                score = _reduce(bs, red, valid)
                key = f"{sname}|{bname}|{red}|rope={int(rope)}|relu={int(relu)}"
                out[key] = capture(score)
                for b in budgets:
                    out_b[b][key] = capture(score, b)
                rho_out[key] = rank_corr(score)
            if kl_on and not relu:
                for red in reduces:
                    rows = []
                    for beta in KL_BETAS:
                        if (
                            bname == "toklse"
                        ):  # the temperature acts on the token logits
                            bsb = torch.logsumexp(
                                dm.view(nq, ni, nb, page) * beta, dim=-1
                            )
                        else:
                            bsb = bs * beta
                        rows.append(_kl(mbar, _predict(bsb, red, valid)))
                    kl_out[f"{sname}|{bname}|{red}|rope={int(rope)}|relu=0"] = rows

    layer = impl.layer_num
    rec = {
        "layer": layer,
        "n": n,
        "pos": pos.tolist(),
        "nvalid": nvalid.tolist(),
        "forced": [round(float(x), 5) for x in forced_mass.tolist()],
        "init": [round(float(x), 5) for x in (mbar * is_init).sum(-1).tolist()],
        "oracle": [round(float(x), 5) for x in oracle_mass.tolist()],
        "random": [round(float(x), 5) for x in random_mass.tolist()],
        "combos": out,
        "rho": rho_out,
        "budgets": {
            str(b): {
                "random": [round(float(x), 5) for x in ref_b[b][0].tolist()],
                "oracle": [round(float(x), 5) for x in ref_b[b][1].tolist()],
                "combos": out_b[b],
            }
            for b in budgets
        },
    }
    if kl_on:
        rec["kl"] = kl_out  # key -> [beta][query]
        rec["kl_betas"] = KL_BETAS
        rec["kl_uniform"] = _kl(mbar, valid.float() / nvalid.clamp(min=1).unsqueeze(-1))
    if rank == 0:
        d = os.environ.get(
            "ATOM_M3PRO_DIAG_DIR", os.path.join(tempfile.gettempdir(), "m3pro_diag")
        )
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{v.diag_tag}.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
