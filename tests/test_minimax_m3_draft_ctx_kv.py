# SPDX-License-Identifier: MIT
"""The MiniMax-M3.1 / M3-Pro draft's context K/V from one GEMM over the five
layers' concatenated K/V weight rows (`MiniMaxM3DSpark.write_context_kv`),
against the per-layer path it replaces (`DSparkDraftModel.write_context_kv`:
each layer's full qkv GEMM, q rows dropped), at Pro's per-rank TP8 shapes.

The weights are processed by the linear's own load step, so they carry the
layout (preshuffled weight and scale on gfx950) the served draft holds. The
GEMM kernel is chosen by output width, so the concatenated GEMM can sum in
another fp32 order than the per-layer one: an element may land one bf16 step
away, on a small share of the elements."""

import types

import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="the draft imports the ATOM model stack")

from aiter import QuantType, dtypes

from atom.model_ops.blockscale import native_quant_linear
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.linear import LinearBase
from atom.models.minimax_m3 import dspark as m3_dspark
from atom.models.minimax_m3 import pro_dspark

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# Pro at TP8, per rank: 16 q heads of 256, one K and one V head of 256
HIDDEN, Q, KV, LAYERS = 7168, 4096, 256, 5


class _Qkv(types.SimpleNamespace):
    """A native MXFP8 1x32 qkv_proj after its load step: the forward is
    `LinearBase.forward`'s native branch, `make_row_view` the linear's own."""

    make_row_view = LinearBase.make_row_view

    def __call__(self, x, x_scale=None):
        return native_quant_linear(
            x,
            self.weight,
            self.weight_scale,
            x_scale=x_scale,
            weight_group_rows=1,
            weight_preshuffled=getattr(self.weight, "is_shuffled", False),
        )


def _qkv(seed, n=Q + 2 * KV, k=HIDDEN):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.5).to(dtypes.fp8)
    s = torch.randint(118, 126, (n, k // 32), device="cuda", generator=g)
    layer = _Qkv(
        weight=nn.Parameter(w, requires_grad=False),
        weight_scale=nn.Parameter(
            s.to(torch.uint8).view(dtypes.fp8_e8m0), requires_grad=False
        ),
        quant_type=QuantType.per_1x32,
        params_dtype=dtypes.fp8,
        native_a8_group_rows=1,
        weight_scale_row_group=1,
        runs_bf16=False,
        bias=None,
        _parameters={},
    )
    LinearBase.process_weights_after_loading(layer)
    return layer


def _attn(seed, writes):
    attn = m3_dspark.M3DSparkAttention.__new__(m3_dspark.M3DSparkAttention)
    nn.Module.__init__(attn)
    attn.qkv_proj = _qkv(seed)
    attn.q_size, attn.kv_size = Q, KV
    attn.kv = types.SimpleNamespace(bound=True)
    attn._bound_cache = lambda: ("cache", seed)

    def write(q, k, v, num_heads, positions, slot_mapping, cache):
        assert num_heads == 0 and cache == ("cache", seed)
        writes.append((seed, k.clone(), v.clone()))

    attn._norm_rope_write = write
    return attn


def _draft(monkeypatch, fused_epilogues, writes):
    ctx = types.SimpleNamespace(context=types.SimpleNamespace(is_dummy_run=False))
    monkeypatch.setattr(m3_dspark, "get_forward_context", lambda: ctx)
    monkeypatch.setattr("atom.utils.forward_context.get_forward_context", lambda: ctx)
    Draft = pro_dspark.MiniMaxM3ProDSpark
    draft = Draft.__new__(Draft)
    nn.Module.__init__(draft)
    layers = []
    for i in range(LAYERS):
        layer = nn.Module()
        layer.self_attn = _attn(10 + i, writes)
        layer.fused_epilogues = fused_epilogues
        # the per-layer path's hop: `DSparkGQADecoderLayer.write_context_kv`
        layer.write_context_kv = types.MethodType(
            m3_dspark.DSparkGQADecoderLayer.write_context_kv, layer
        )
        layers.append(layer)
    draft.layers = nn.ModuleList(layers)
    draft.fc = lambda x: x  # the taps arrive already projected
    norm = GemmaRMSNorm(HIDDEN, eps=1e-6)
    g = torch.Generator().manual_seed(3)
    norm.weight.data = (torch.randn(HIDDEN, generator=g) * 0.3).to(torch.bfloat16)
    draft.hidden_norm = norm.cuda()
    return draft


def _write(draft, x, writes, fused):
    writes.clear()
    draft._ctx_kv = None if fused else ()
    positions = torch.arange(x.shape[0], device="cuda")
    draft.write_context_kv([x], positions, positions)
    assert [w[0] for w in writes] == [10 + i for i in range(LAYERS)]
    return [(k, v) for _, k, v in writes]


@needs_gpu
@pytest.mark.parametrize("fused_epilogues", [True, False])
@pytest.mark.parametrize("rows", [1, 56, 128, 256, 2048])
def test_one_gemm_gives_every_layers_context_kv(monkeypatch, rows, fused_epilogues):
    torch.manual_seed(rows)
    writes = []
    draft = _draft(monkeypatch, fused_epilogues, writes)
    x = (torch.randn(rows, HIDDEN, device="cuda") * 2).to(torch.bfloat16)
    ref = _write(draft, x, writes, fused=False)
    got = _write(draft, x, writes, fused=True)
    weight, _, shuffled = draft._ctx_kv
    assert weight.shape == (LAYERS * 2 * KV, HIDDEN)
    assert shuffled == draft.layers[0].self_attn.qkv_proj.weight.is_shuffled
    differ = total = 0
    for (gk, gv), (rk, rv) in zip(got, ref):
        for g, r in ((gk, rk), (gv, rv)):
            assert g.shape == r.shape == (rows, KV)
            # one bf16 step: 2**-7 of the value, 2**-8 rounded either way
            torch.testing.assert_close(g, r, atol=1e-5, rtol=2**-7)
            differ += int((g != r).sum())
            total += g.numel()
    print(f"rows {rows} fused {fused_epilogues}: {differ} of {total} differ")
    assert differ <= 1e-3 * total


@needs_gpu
def test_the_per_layer_path_stays_where_one_gemm_cannot_run(monkeypatch):
    """A PCP-split prefill, an unbound cache or a projection that is not
    native MXFP8 keep the per-layer path."""
    writes = []
    draft = _draft(monkeypatch, True, writes)
    x = torch.randn(8, HIDDEN, device="cuda").to(torch.bfloat16)
    positions = torch.arange(8, device="cuda")

    proj = draft.layers[2].self_attn.qkv_proj
    proj.runs_bf16 = True
    assert draft._context_kv_weight() is None
    proj.runs_bf16 = False

    calls = []
    monkeypatch.setattr(
        m3_dspark.DSparkDraftModel,
        "write_context_kv",
        lambda self, *a: calls.append(a),
    )
    draft._ctx_kv = None
    draft.layers[0].self_attn.kv.bound = False
    draft.write_context_kv([x], positions, positions)
    draft.layers[0].self_attn.kv.bound = True
    draft.write_context_kv([x], positions, positions, 8)
    assert len(calls) == 2 and draft._ctx_kv is None and not writes
