# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Torch (CPU) reference of Pro mono phase 1's ``upto = "norm"`` layer kernel,
in production's numerics at <= 16 rows under the combo config.

* o_proj: ``quantize_fp8`` of the bf16 core rows (MXFP8 1x32: code =
  ceil_pow2(max(amax, 1e-10) * f32(1 / 448)), e4m3 RNE of the clamped
  quotient), times the native MXFP8 weight, fp32 accumulate, bf16 partial;
* the all-reduce: the 8 bf16 partials summed in fp32 in rank order, bf16;
* aiter ``add_rmsnorm_mxfp8(..., gemma=True)``: v = f32(a) + f32(res) unrounded,
  h_mid = bf16(v), rstd = rsqrt(sum(v^2) / 7168 + eps), x = bf16(v rstd (1 + w)),
  x8 = MXFP8 of x;
* the lookahead (aiter ``fused_qk_rmsnorm_group_quant``, gemma, on the bf16
  residual): la8 = MXFP8 of bf16(bf16(v) rstd2 (1 + w2)).

Not ``GemmaRMSNorm.forward_static``: for bf16 input it adds the residual in
bf16 (rounded) and norms the rounded value.
"""

import torch

HIDDEN = 7168
TP = 8
FP8_MAX = 448.0
_INV_FP8_MAX = torch.tensor(1.0 / FP8_MAX, dtype=torch.float32)  # f32(1/448)
AMAX_FLOOR = 1e-10


# a residual row's norm over its token's all-reduced row's norm, by token: the
# residual barely matters, matters as much, or dominates -- none swamped
RES_RATIOS = (1.0, 0.1, 10.0)


def token_sizes(s, seed):
    """One size a token over 6 decades. Every rank's core row of a token and
    its residual row take the same size, so no rank's partial and no residual
    is drowned by another's."""
    g = torch.Generator().manual_seed(seed)
    return 10.0 ** (torch.rand(s, generator=g) * 6 - 3)


def rows(sizes, n, seed):
    """Rows of the given sizes with a few x64 outlier columns (the Lamport op
    test's generator): every MXFP8 code edge gets exercised."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(len(sizes), n, generator=g)
    x[:, torch.randint(0, n, (max(1, n // 512),), generator=g)] *= 64.0
    return (x * sizes.float().unsqueeze(-1)).to(torch.bfloat16)


def residual_rows(ar_bf16, seed):
    """Residual rows whose norms are ``RES_RATIOS`` (by token, cyclic) of the
    all-reduced rows' norms."""
    s, n = ar_bf16.shape
    d = rows(torch.ones(s), n, seed).float()
    ratio = torch.tensor([RES_RATIOS[t % len(RES_RATIOS)] for t in range(s)])
    want = ratio * ar_bf16.float().norm(dim=-1)
    return (d * (want / d.norm(dim=-1)).unsqueeze(-1)).to(torch.bfloat16)


def case_inputs(s, weights, seed):
    """One case's inputs from the ranks' o_proj weights ``weights`` [(e4m3,
    E8M0)]: (the ranks' bf16 core rows [s, K], their bf16 o partials [s, 7168],
    the bf16 residual [s, 7168])."""
    sizes = token_sizes(s, seed)
    k = weights[0][0].shape[1]
    cores = [rows(sizes, k, seed + 1 + r) for r in range(len(weights))]
    parts = [o_partial(c, *w) for c, w in zip(cores, weights)]
    return cores, parts, residual_rows(ar_sum(parts), seed + 100)


def mxfp8_codes(amax: torch.Tensor) -> torch.Tensor:
    """E8M0 codes of f32 block maxima, clamped to [1, 254]."""
    a = torch.clamp(amax.float(), min=AMAX_FLOOR) * _INV_FP8_MAX
    bits = a.view(torch.int32)
    code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return code.clamp(1, 254)


def mxfp8(x_bf16: torch.Tensor):
    """MXFP8 1x32 of bf16 [m, n] -> (e4m3 [m, n], uint8 codes [m, n / 32])."""
    m, n = x_bf16.shape
    x = x_bf16.float().view(m, n // 32, 32)
    code = mxfp8_codes(x.abs().amax(-1))
    inv = ((254 - code) << 23).view(torch.float32)  # 2^(127 - code), exact
    q = torch.clamp(x * inv.unsqueeze(-1), -FP8_MAX, FP8_MAX)
    return q.to(torch.float8_e4m3fn).view(m, n), code.to(torch.uint8)


def dequant(q: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    m, n = q.shape
    scale = ((codes.to(torch.int32) << 23).view(torch.float32)).unsqueeze(-1)
    return (q.float().view(m, n // 32, 32) * scale).view(m, n)


def o_partial(core_bf16, w_q, w_codes):
    """One rank's o_proj partial: bf16(MXFP8(core) @ W^T), fp32 accumulate."""
    xq, xs = mxfp8(core_bf16)
    acc = dequant(xq, xs).double() @ dequant(w_q, w_codes).double().T
    return acc.float().to(torch.bfloat16)


def ar_sum(parts):
    acc = parts[0].float()
    for p in parts[1:]:
        acc = acc + p.float()
    return acc.to(torch.bfloat16)


def gemma_rows(v_f32, w_bf16, eps):
    ss = (v_f32.double() ** 2).sum(-1, keepdim=True).float()
    rstd = torch.rsqrt(ss / float(HIDDEN) + eps)
    return ((v_f32 * rstd) * (w_bf16.float() + 1.0)).to(torch.bfloat16)


def layer_reference(parts, res, g_post, g_ahead, eps, eps_ahead):
    """The kernel's outputs from the ranks' o partials ``parts``."""
    v = ar_sum(parts).float() + res.float()
    h_mid = v.to(torch.bfloat16)
    x = gemma_rows(v, g_post, eps)
    x8, x8s = mxfp8(x)
    out = {"h_mid": h_mid, "x": x, "x8": x8, "x8s": x8s}
    if g_ahead is not None:
        out["la8"], out["la8s"] = mxfp8(gemma_rows(h_mid.float(), g_ahead, eps_ahead))
    return out


def code_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """How many representable values apart two e4m3 (or E8M0) tensors are,
    element-wise (sign-magnitude bytes mapped onto one ordered line)."""

    def line(t):
        u = t.contiguous().view(torch.uint8).to(torch.int32)
        if t.dtype == torch.uint8:
            return u
        mag = u & 0x7F
        return torch.where(u & 0x80 != 0, -mag, mag)

    return (line(a) - line(b)).abs()


def row_relnorm(got: torch.Tensor, want: torch.Tensor) -> float:
    """The largest relative error norm over the rows (one norm over the whole
    tensor would let the loudest row hide errors in the others)."""
    a, b = got.float().flatten(1), want.float().flatten(1)
    return float(((a - b).norm(dim=-1) / b.norm(dim=-1).clamp_min(1e-30)).max())


def lookahead_consistency(got, g_ahead, eps_ahead) -> int:
    """How many of the kernel's la8 values disagree with the MXFP8 lookahead
    norm of its own h_mid by more than one e4m3 step (at the coarser of the two
    scales), plus the scale codes more than 1 apart. The input is exact
    (bf16(v) is h_mid); only the sum order of the sum of squares and the rsq
    may differ, which moves a value by at most one step (or a scale by one
    code, the block then re-rounded)."""
    q, s = mxfp8(gemma_rows(got["h_mid"].float(), g_ahead, eps_ahead))
    a, b = dequant(got["la8"], got["la8s"]), dequant(q, s)
    code = torch.maximum(got["la8s"], s).to(torch.int32).repeat_interleave(32, -1)
    tiny = ((code - 9) << 23).view(torch.float32)  # the subnormal step
    tol = torch.maximum(a.abs(), b.abs()) * 2.0**-3 + tiny
    return int(((a - b).abs() > tol).sum() + (code_distance(got["la8s"], s) > 1).sum())


def layer_report(got, want, g_ahead=None, eps_ahead=None):
    """The kernel's outputs ``got`` against the reference ``want``: the largest
    per-row relative error norms, the scale codes more than 1 apart, the share of
    bit-exact h_mid, and the exact self-consistency of the MXFP8 outputs."""
    x8, x8s = mxfp8(got["x"])
    rep = {
        "h_mid_exact": float(
            (got["h_mid"].view(torch.int16) == want["h_mid"].view(torch.int16))
            .float()
            .mean()
        ),
        "h_mid": row_relnorm(got["h_mid"], want["h_mid"]),
        "x": row_relnorm(got["x"], want["x"]),
        "x8": row_relnorm(
            dequant(got["x8"], got["x8s"]), dequant(want["x8"], want["x8s"])
        ),
        "x8s_far": int((code_distance(got["x8s"], want["x8s"]) > 1).sum()),
        # the kernel quantizes the bf16 x it stores: bit for bit
        "x8_self": bool(
            torch.equal(got["x8"].view(torch.uint8), x8.view(torch.uint8))
            and torch.equal(got["x8s"], x8s)
        ),
    }
    if g_ahead is not None:
        rep["la8"] = row_relnorm(
            dequant(got["la8"], got["la8s"]), dequant(want["la8"], want["la8s"])
        )
        rep["la8s_far"] = int((code_distance(got["la8s"], want["la8s"]) > 1).sum())
        rep["la8_self_far"] = lookahead_consistency(got, g_ahead, eps_ahead)
    return rep


# The kernel's o_proj sums K in another order than the fp64 reference, so a
# bf16 partial can differ in its last bit (design section 12); everything after
# it is production's recipe. The largest per-row relative error norm a row may
# show, set so that every partial element 1 ulp off (the worst that allows)
# still passes: on this test's inputs that gives at most 0.0065 (h_mid, x) and
# 0.029 (x8, la8) on CPU (an f32 o_proj summed in 128-K chunks: 1.8e-5 and
# 3.5e-4). A wrong operand, scale, column, token, rank or residual is >= 0.03.
BOUNDS = {"h_mid": 2.0**-7, "x": 2.0**-7, "x8": 2.0**-4, "la8": 2.0**-4}
# share of h_mid bit for bit equal to the reference's: 1.0 with the chunked f32
# o_proj above, 0.51 .. 0.56 with every partial element 1 ulp off
H_MID_EXACT_FLOOR = 0.45


def layer_failures(rep):
    """The bounds a ``layer_report`` breaks, by name (empty: it passes)."""
    bad = [k for k, b in BOUNDS.items() if k in rep and not rep[k] <= b]
    bad += [k for k in ("x8s_far", "la8s_far", "la8_self_far") if rep.get(k, 0)]
    if not rep["x8_self"]:
        bad.append("x8_self")
    if not rep["h_mid_exact"] >= H_MID_EXACT_FLOOR:
        bad.append("h_mid_exact")
    return bad
