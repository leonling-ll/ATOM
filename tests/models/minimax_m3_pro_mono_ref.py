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


# ================================================================= the FFN
# ``upto = "ffn"``: this rank's FFN partial with the routed experts in the
# A16W4 recipe (``act = "bf16"``, the default: production's a16w4 numerics,
# the bf16 x and bf16 mids against the MXFP4 planes dequantized exactly, fp32
# accumulation), the W4A4 one (``"fp4"``: aiter a4w4 numerics, MXFP4
# activations) or the W4A8 one (``"fp8"``: the norm's MXFP8 x8 and MXFP8 mids,
# ``quantize_fp8``'s rule), and production's MXFP8 shared expert. The widths
# come from the tensors, so a CPU test can run it narrow.

TOPK = 4
ROUTE_SCALE = 2.0
SWIGLU_ALPHA = 1.702
SWIGLU_LIMIT = 7.0
_INV6 = torch.tensor(1.0 / 6.0, dtype=torch.float32)  # aiter's f32(1/6)
E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0]
    + [-4.0, -6.0]
)


def _code_ceil(v):
    """mono's ``code_ceil``: the biased exponent of the smallest power of two
    >= v, clamped to [1, 254]."""
    bits = v.float().contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return e.clamp(1, 254)


def _f32_to_e2m1(q):
    """aiter ``emit_f32_to_e2m1`` in integer ops: RNE, saturating at 6 (the
    nibble, sign in bit 3)."""
    qx = q.contiguous().view(torch.int32)
    sign = qx & -(2**31)
    a = (qx & 0x7FFFFFFF).to(torch.int64)
    denorm_bias = 0x4A800000
    denorm = (
        a.to(torch.int32).view(torch.float32)
        + torch.tensor(denorm_bias, dtype=torch.int32).view(torch.float32)
    ).view(torch.int32) - denorm_bias
    mant_odd = (a >> 22) & 1
    normal = ((a + 0xC11FFFFF + mant_odd) & 0xFFFFFFFF) >> 22
    e = torch.where((a < 0x40C00000) & (a >= 0x3F800000), normal, torch.full_like(a, 7))
    e = torch.where(a < 0x3F800000, denorm.to(torch.int64), e)
    return ((sign >> 28) & 8).to(torch.int64) | e


def mxfp4(x):
    """MXFP4 1x32 of [m, n] (bf16 values) as aiter's a4w4 quant does it: E8M0
    RoundUp of amax * f32(1/6), the quotient RNE to e2m1 -> (nibbles int64 [m,
    n], codes int32 [m, n / 32])."""
    m, n = x.shape
    r = x.float().view(m, n // 32, 32)
    code = _code_ceil(r.abs().amax(-1) * _INV6)
    q = r / ((code << 23).view(torch.float32)).unsqueeze(-1)  # exact: a power of 2
    return _f32_to_e2m1(q.reshape(m, n)), code


def dq_mxfp4(nib, code):
    m, n = nib.shape
    v = E2M1.to(nib.device)[nib].view(m, n // 32, 32)
    return (v * ((code.to(torch.int32) << 23).view(torch.float32)).unsqueeze(-1)).view(
        m, n
    )


def qdq_mxfp4(x):
    return dq_mxfp4(*mxfp4(x))


def dq_mxfp4_bytes(b, code):
    """Packed e2m1 bytes [m, n / 2] (low nibble the even column) and E8M0 codes
    [m, n / 32] -> f32 [m, n]: a routed weight plane's logical rows."""
    lo, hi = (b & 0xF).long(), (b >> 4).long()
    nib = torch.stack([lo, hi], -1).view(b.shape[0], -1)
    return dq_mxfp4(nib, code)


def swiglu_oai(g, u, alpha=SWIGLU_ALPHA, limit=SWIGLU_LIMIT):
    """aiter ``swiglu_mul_batch``: gate upper-clamped, up two-sided."""
    g = torch.clamp(g, max=limit)
    u = torch.clamp(u, -limit, limit)
    return g * torch.sigmoid(alpha * g) * (u + 1.0)


def _bf(x):
    return x.float().to(torch.bfloat16).float()


def route_reference_of_logits(logits, bias):
    """The selection on bf16 logits [S, E]: sigmoid, the top 4 of sigmoid +
    bias (equal scores: the lower expert first), the 4 unbiased sigmoids
    renormalized in that order x 2.0. -> (logits, picks [S, 4], weights [S,
    4], scores [S, E] f32)."""
    sig = torch.sigmoid(logits.float())
    score = sig + bias.float()
    picks = torch.sort(score, dim=-1, descending=True, stable=True).indices[:, :TOPK]
    sel = sig.gather(-1, picks)
    total = sel[:, 0]
    for k in range(1, TOPK):
        total = total + sel[:, k]
    weights = sel * (ROUTE_SCALE / total).unsqueeze(-1)
    return logits, picks, weights, score


def route_reference(x, w_gate, bias):
    """The router on the norm's bf16 x [S, H]: bf16 logits of the bf16 gate
    [E, H] (fp32 accumulate), then ``route_reference_of_logits``."""
    logits = (x.double() @ w_gate.double().T).to(torch.bfloat16)
    return route_reference_of_logits(logits, bias)


def qdq_mxfp8(x):
    """MXFP8 1x32 of bf16 values [m, n], dequantized (``quantize_fp8``)."""
    return dequant(*mxfp8(x.to(torch.bfloat16)))


ROUTED_ACTS = ("fp4", "fp8", "bf16")


def routed_input(x, x8, x8s, act):
    """The routed experts' activations [S, H] as fp32 values: the bf16 x
    itself (A16W4), MXFP4(x) (W4A4) or the norm's x8 (W4A8)."""
    if act == "bf16":
        return x.float()
    return qdq_mxfp4(x) if act == "fp4" else dequant(x8, x8s)


def qdq_mid_of(act):
    """The routed mids' quantize-dequantize: none for A16W4 (the bf16 mid is
    the down's operand), MXFP4 / MXFP8 for W4A4 / W4A8."""
    return {"bf16": lambda m: m.float(), "fp4": qdq_mxfp4, "fp8": qdq_mxfp8}[act]


def ffn_reference(
    x, x8, x8s, picks, weights, routed, shared, swiglu=swiglu_oai, act="bf16"
):
    """This rank's FFN partial [S, H] bf16 from the norm's x (bf16) and x8
    (e4m3, codes), the route (picks, weights [S, 4]) and the weights:
    ``routed(e)`` -> (w13 [2 I, H], w2 [H, I]) dequantized, the gate rows then
    the up rows over the planes' padded I (the pad zero, as loaded);
    ``shared`` = (gate_up [2 Is, H], down [H, Is]) dequantized. ``act``: the
    routed activations, MXA = none ("bf16", xa = x: production's a16w4 math,
    the K sums exact here and rounded once to f32 where the GPU's round each
    MFMA's), MXFP4 ("fp4", xa = MXFP4(x)) or MXFP8 ("fp8", xa = x8).

        routed = bf16(sum_k bf16(w_k (MXA(bf16(SwiGLU(xa w13^T))) w2^T)))
        shared = bf16(MXFP8(bf16(SwiGLU(bf16(x8 w_sgu^T)))) w_sd^T)
        ffn    = bf16(routed + shared)
    """
    assert act in ROUTED_ACTS, act
    s = x.shape[0]
    xq = routed_input(x, x8, x8s, act)
    qdq_mid = qdq_mid_of(act)
    out = []
    for t in range(s):
        tot = None
        for k in range(TOPK):
            w13, w2 = routed(int(picks[t, k]))
            gu = (w13.double() @ xq[t].double()).float()
            i = gu.shape[0] // 2
            mid = _bf(swiglu(gu[:i], gu[i:]))
            acc = (w2.double() @ qdq_mid(mid[None])[0].double()).float()
            slot = _bf(acc * weights[t, k].float())
            tot = slot if tot is None else tot + slot
        out.append(_bf(tot))
    routed_out = torch.stack(out)
    w_sgu, w_sd = shared
    gu = _bf((dequant(x8, x8s).double() @ w_sgu.double().T).float())
    i = gu.shape[1] // 2
    y = swiglu(gu[:, :i], gu[:, i:]).to(torch.bfloat16)
    sh = _bf((dequant(*mxfp8(y)).double() @ w_sd.double().T).float())
    return (routed_out + sh).to(torch.bfloat16)


# The largest per-row relative error norm of the kernel's FFN partial against
# ``ffn_reference`` on the kernel's own x, picks and weights. The GEMVs sum K
# in another order (gate / up, the down, the shared expert), which flips an
# MXFP4 mid code now and then: an fp4 grid step is 4 - 8 % of a value. Set
# (``test_minimax_m3_pro_mono_ffn``) above that noise and below the errors a
# wrong scale, activation, operand or route leaves.
FFN_BOUND = 2.0**-5
# the kernel's gathered router logits may differ from the reference's by one
# bf16 step (sum order); its picks must be the top 4 of its own logits'
# scores but for ties within ``TIE_TOL`` (hardware exp2 / rcp against torch's
# sigmoid), and its weights their renormalization within ``WEIGHT_TOL``
LOGIT_STEPS = 1
TIE_TOL = 1e-5
WEIGHT_TOL = 1e-4


def bf16_steps(a, b):
    """How many bf16 values apart two bf16 tensors are, element-wise."""

    def line(t):
        u = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(u < 0, -(u & 0x7FFF), u)

    return (line(a) - line(b)).abs()


def route_failures(logits, picks, weights, ref_logits, bias):
    """The kernel's router -- its gathered logits [S, E] bf16, its picks and
    weights [S, 4] -- against ``route_reference``'s logits, and against the
    reference's selection and renormalization run on the kernel's own logits.
    -> the problems, by name."""
    bad = []
    if int(bf16_steps(logits, ref_logits).max()) > LOGIT_STEPS:
        bad.append("logits")
    _, want_picks, want_w, score = route_reference_of_logits(logits, bias)
    for t in range(picks.shape[0]):
        mine, want = set(picks[t].tolist()), set(want_picks[t].tolist())
        if mine == want:
            continue
        fourth = float(score[t, want_picks[t, TOPK - 1]])
        if len(mine) != TOPK or any(
            abs(float(score[t, e]) - fourth) > TIE_TOL for e in mine ^ want
        ):
            bad.append(f"token {t}: picks {sorted(mine)}, want {sorted(want)}")
    sig = torch.sigmoid(logits.float()).gather(-1, picks.long())
    want_w = sig * (ROUTE_SCALE / sig.sum(-1, keepdim=True))
    if float((weights.float() - want_w).abs().max()) > WEIGHT_TOL:
        bad.append("weights")
    return bad


def ffn_failures(got, want):
    """The FFN partial's per-row bound, by name (empty: it passes)."""
    rel = row_relnorm(got, want)
    return [] if rel <= FFN_BOUND else [f"ffn rel {rel:.3g}"]


def ffn_mutations(x, x8, x8s, picks, weights, routed, shared, n_experts, act="bf16"):
    """The FFN partial with one bug injected each, by name: what the GPU
    test's ``ffn_failures`` must catch (``test_minimax_m3_pro_mono_ffn``)."""
    w_sgu, w_sd = shared
    s = x.shape[0]

    def run(**kw):
        args = {
            "x": x, "x8": x8, "x8s": x8s, "picks": picks, "weights": weights,
            "routed": routed, "shared": shared, "act": act,
        }  # fmt: skip
        return ffn_reference(**(args | kw))

    # a lost K block of the activations the routed experts read (x or x8)
    block5 = torch.arange(x.shape[1]) // 32 != 5
    lost_k = (
        {"x8": (x8.view(torch.uint8) * block5.to(torch.uint8)).view(x8.dtype)}
        if act == "fp8"
        else {"x": x * block5.to(x.dtype)}
    )

    def routed_with(f):
        return lambda e: f(*routed(e))

    zero_sh = (torch.zeros_like(w_sgu), torch.zeros_like(w_sd))
    other = picks.clone()
    # the 4th pick moved to the next expert the token did not pick
    for t in range(s):
        e = int(picks[t, -1])
        while e in picks[t].tolist():
            e = (e + 1) % n_experts
        other[t, -1] = e
    rows = torch.arange(s)
    out = {
        "no route scale": run(weights=weights / ROUTE_SCALE),
        "one weight lost": run(weights=weights * (torch.arange(TOPK) != 2)),
        "picks of another token": run(picks=picks[(rows + 1) % s]) if s > 1 else None,
        "another 4th expert": run(picks=other),
        "no shared expert": run(shared=zero_sh),
        "no routed experts": run(weights=torch.zeros_like(weights)),
        "gate and up swapped": run(
            routed=routed_with(lambda w13, w2: (w13.roll(w13.shape[0] // 2, 0), w2))
        ),
        "shared gate and up swapped": run(
            shared=(w_sgu.roll(w_sgu.shape[0] // 2, 0), w_sd)
        ),
        "swiglu unclamped": run(swiglu=lambda g, u: swiglu_oai(g, u, limit=1e30)),
        "silu, no up + 1": run(
            swiglu=lambda g, u: torch.clamp(g, max=7)
            * torch.sigmoid(1.702 * g)
            * torch.clamp(u, -7, 7)
        ),
        "sigmoid alpha 1": run(swiglu=lambda g, u: swiglu_oai(g, u, alpha=1.0)),
        "a K block of x lost": run(**lost_k),
        "x8 scale code + 1": run(x8s=x8s + (torch.arange(x8s.shape[1]) == 3).byte()),
        "w2 rows shifted by 16": run(
            routed=routed_with(lambda w13, w2: (w13, w2.roll(16, 0)))
        ),
    }
    return {k: v for k, v in out.items() if v is not None}


# The test weights' E8M0 code ranges [lo, hi): the routed gate / up loud
# enough that |gate|, |up| pass the SwiGLU limit 7 (the clamp is exercised),
# w2 and the shared expert's set so the routed and the shared outputs are of
# one size (neither drowns the other: ``test_minimax_m3_pro_mono_ffn``).
W13_CODES = (120, 125)
W2_CODES = (111, 116)
SHARED_CODES = (125, 128)
SHARED_STD = 0.02
GATE_STD = 0.02
BIAS_STD = 0.02


def mxfp4_plane(rows, cols, codes, g, valid_rows=None, valid_cols=None):
    """A random MXFP4 weight's logical bytes [rows, cols / 2] (low nibble the
    even column) and codes [rows, cols / 32]; rows / columns past ``valid_*``
    zero, as the loader pads them."""
    b = torch.randint(0, 256, (rows, cols // 2), generator=g, dtype=torch.uint8)
    c = torch.randint(*codes, (rows, cols // 32), generator=g, dtype=torch.uint8)
    if valid_rows is not None:
        b[~valid_rows] = 0
    if valid_cols is not None:
        b[:, valid_cols // 2 :] = 0
    return b, c


def ffn_case(s, h, i_valid, i_pad, e, i_sh, seed):
    """A narrow FFN case for the CPU tests: x (with its MXFP8), the router,
    every expert's dequantized planes (padded ``i_pad``, ``i_valid`` real),
    the shared expert, and the reference route."""
    g = torch.Generator().manual_seed(seed)
    gain = 1 + 0.3 * torch.randn(h, generator=g)
    x = (torch.randn(s, h, generator=g) * gain).to(torch.bfloat16)
    x8, x8s = mxfp8(x)
    w_gate = (torch.randn(e, h, generator=g) * GATE_STD).to(torch.bfloat16)
    bias = torch.randn(e, generator=g) * BIAS_STD
    valid = torch.arange(2 * i_pad) % i_pad < i_valid
    experts = {}
    for ex in range(e):
        w13 = dq_mxfp4_bytes(*mxfp4_plane(2 * i_pad, h, W13_CODES, g, valid))
        w2 = dq_mxfp4_bytes(*mxfp4_plane(h, i_pad, W2_CODES, g, None, i_valid))
        experts[ex] = (w13, w2)

    def f8(r, c):
        w = (torch.randn(r, c, generator=g) * SHARED_STD).to(torch.float8_e4m3fn)
        k = torch.randint(*SHARED_CODES, (r, c // 32), generator=g, dtype=torch.uint8)
        return dequant(w, k)

    shared = (f8(2 * i_sh, h), f8(h, i_sh))
    logits, picks, weights, _ = route_reference(x, w_gate, bias)
    return {
        "x": x, "x8": x8, "x8s": x8s, "w_gate": w_gate, "bias": bias,
        "routed": experts.__getitem__, "shared": shared, "logits": logits,
        "picks": picks, "weights": weights,
    }  # fmt: skip


def case_ffn(case, **kw):
    """``ffn_reference`` of a ``ffn_case`` (``act`` in ``kw``: the routed
    activations, A16W4 by default)."""
    c = case
    return ffn_reference(
        c["x"], c["x8"], c["x8s"], c["picks"], c["weights"], c["routed"],
        c["shared"], **kw,
    )  # fmt: skip
