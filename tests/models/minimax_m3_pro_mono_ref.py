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


def rows(m, n, seed, scale=1.0):
    """Rows over 6 decades with a few x64 outlier columns (the Lamport op
    test's generator): every MXFP8 code edge gets exercised."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(m, n, generator=g)
    x *= 10.0 ** (torch.rand(m, 1, generator=g) * 6 - 3)
    x[:, torch.randint(0, n, (max(1, n // 512),), generator=g)] *= 64.0
    return (x * scale).to(torch.bfloat16)


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
