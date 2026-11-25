# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's ``qk_pair`` variant: qk_proj and index_qk_proj in one aiter launch.

``pro._qk_pair`` hands both native MXFP8 GEMMs, each with its own input, to aiter's
``gemm_a8w8_blockscale_bpreshuffle_pair`` and returns its two outputs; it declines
(None: the two GEMMs run as before) when the variant is off, aiter lacks the pair op,
or either linear is not a plain preshuffled MXFP8 GEMM.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

aiter = pytest.importorskip("aiter", reason="pro imports the ATOM model stack")

from atom.model_ops.minimax_m3 import pro_attn
from atom.model_ops.minimax_m3.pro_attn import ProVariants
from atom.models.minimax_m3 import pro

M, K = 5, 64


def _linear(n, **over):
    weight = torch.zeros(n, K)
    weight.is_shuffled = True
    lin = SimpleNamespace(
        weight=weight,
        weight_scale=torch.zeros(n, K // 32),
        native_a8_group_rows=1,
        bias=None,
        is_output_padded=False,
        tp_dim=0,
        tp_size=8,
        reduce_results=True,
    )
    for key, value in over.items():
        setattr(lin, key, value)
    return lin


@pytest.fixture
def pair_op(monkeypatch):
    """aiter's pair op, recording its calls and writing 1 / 2 into the outputs."""
    calls = []

    def fake(x, w, xs, ws, x2, w2, xs2, ws2, out, out2):
        calls.append((x, w, xs, ws, x2, w2, xs2, ws2, out, out2))
        out.fill_(1.0)
        out2.fill_(2.0)

    monkeypatch.setattr(
        aiter, "gemm_a8w8_blockscale_bpreshuffle_pair", fake, raising=False
    )
    return calls


def _use(monkeypatch, **kw):
    monkeypatch.setattr(pro_attn, "_variants", ProVariants(**kw))


def _inputs():
    x, xs = torch.randn(M, K), torch.randn(M, K // 32)
    ix, ixs = torch.randn(M, K), torch.randn(M, K // 32)
    return x, xs, ix, ixs


def test_pair_runs_both_gemms_in_one_call(monkeypatch, pair_op):
    _use(monkeypatch, qk_pair=True)
    qk, iqk = _linear(48), _linear(32)
    x, xs, ix, ixs = _inputs()
    got = pro._qk_pair(qk, x, xs, iqk, ix, ixs)
    assert got is not None and len(pair_op) == 1
    cx, cw, cxs, cws, cx2, cw2, cxs2, cws2, out, out2 = pair_op[0]
    assert cx is x and cxs is xs and cw is qk.weight and cws is qk.weight_scale
    assert cx2 is ix and cxs2 is ixs and cw2 is iqk.weight and cws2 is iqk.weight_scale
    assert got[0] is out and got[1] is out2
    assert out.shape == (M, 48) and out2.shape == (M, 32)
    assert out.dtype == out2.dtype == torch.bfloat16
    assert bool((got[0] == 1).all()) and bool((got[1] == 2).all())


def test_pair_is_off_by_default(monkeypatch, pair_op):
    _use(monkeypatch)
    x, xs, ix, ixs = _inputs()
    assert pro._qk_pair(_linear(48), x, xs, _linear(32), ix, ixs) is None
    assert not pair_op


@pytest.mark.parametrize(
    "which,over",
    [
        (0, {"bias": torch.zeros(48)}),
        (1, {"native_a8_group_rows": 32}),
        (1, {"native_a8_group_rows": None}),
        (0, {"is_output_padded": True}),
        (0, {"tp_dim": 1}),
        (1, {"weight": torch.zeros(32, K)}),  # not preshuffled
    ],
)
def test_pair_declines_a_gemm_it_does_not_compute(monkeypatch, pair_op, which, over):
    _use(monkeypatch, qk_pair=True)
    lins = [_linear(48), _linear(32)]
    lins[which] = _linear(lins[which].weight.shape[0], **over)
    x, xs, ix, ixs = _inputs()
    assert pro._qk_pair(lins[0], x, xs, lins[1], ix, ixs) is None
    assert not pair_op


def test_pair_declines_unquantized_input_and_unequal_rows(monkeypatch, pair_op):
    _use(monkeypatch, qk_pair=True)
    x, xs, ix, ixs = _inputs()
    assert pro._qk_pair(_linear(48), x, None, _linear(32), ix, ixs) is None
    assert pro._qk_pair(_linear(48), x, xs, _linear(32), ix[:3], ixs[:3]) is None
    assert not pair_op


def test_pair_declines_without_the_aiter_op(monkeypatch):
    monkeypatch.delattr(aiter, "gemm_a8w8_blockscale_bpreshuffle_pair", raising=False)
    _use(monkeypatch, qk_pair=True)
    x, xs, ix, ixs = _inputs()
    assert pro._qk_pair(_linear(48), x, xs, _linear(32), ix, ixs) is None


def _row_view_parent(n):
    """A native 1x32 MXFP8 LinearBase holding a preshuffled weight, built without
    the quantization machinery: what ``make_row_view`` reads of ``qkv_proj``."""
    from aiter import QuantType
    from torch import nn

    from atom.model_ops.linear import LinearBase

    lin = LinearBase.__new__(LinearBase)
    nn.Module.__init__(lin)
    lin.weight = nn.Parameter(torch.zeros(n, K), requires_grad=False)
    lin.weight.is_shuffled = True
    lin.weight_scale = nn.Parameter(torch.zeros(n, K // 32), requires_grad=False)
    lin.native_a8_group_rows = 1
    lin.quant_type = QuantType.per_1x32
    lin.bias = None
    lin.is_output_padded = False
    lin.tp_dim, lin.tp_size, lin.reduce_results = 0, 8, True
    lin.output_size = n
    return lin


def test_pair_takes_the_row_views_of_one_qkv_weight(monkeypatch, pair_op):
    # 32-row aligned, as a native 1x32 row view must be (Pro: 4352 + 2176 rows)
    _use(monkeypatch, qk_pair=True)
    parent = _row_view_parent(96)
    qk, iqk = parent.make_row_view(0, 64), parent.make_row_view(64, 32)
    x, xs, ix, ixs = _inputs()
    got = pro._qk_pair(qk, x, xs, iqk, ix, ixs)
    assert got is not None and len(pair_op) == 1
    assert got[0].shape == (M, 64) and got[1].shape == (M, 32)
    _, cw, _, cws, _, cw2, _, cws2, _, _ = pair_op[0]
    assert cw.data_ptr() == parent.weight.data_ptr()
    assert cw2.data_ptr() == parent.weight[64:].data_ptr()
    assert cws.shape == (64, K // 32) and cws2.shape == (32, K // 32)


class _Attn:
    impl = SimpleNamespace()

    def __call__(self, q, k, iq, positions, qkv):
        return torch.cat([q, k, iq, qkv], dim=-1)


def _attention(qkv_proj):
    """The parts of ``MiniMaxM3ProAttention`` its forward reads, around the given
    ``qkv_proj``; qk_proj / index_qk_proj are rows 0-47 / 48-79 of it."""
    return SimpleNamespace(
        qkv_proj=qkv_proj,
        qk_proj=_linear(48),
        index_qk_proj=_linear(32),
        qk_sizes=[40, 8],
        index_qk_sizes=[24, 8],
        attn=_Attn(),
        o_proj=lambda out, **kw: out,
    )


def test_front_input_skips_the_pair(monkeypatch, pair_op):
    _use(monkeypatch, qk_pair=True)
    calls = []

    def qkv_proj(x, x_scale):
        calls.append(x)
        return torch.zeros(x.shape[0], 80, dtype=torch.bfloat16)

    x, xs, ix, ixs = _inputs()
    front = (torch.cat([x, ix]), torch.cat([xs, ixs]))
    out = pro.MiniMaxM3ProAttention.forward(
        _attention(qkv_proj), None, x.to(torch.bfloat16), front=front
    )
    assert len(calls) == 1 and calls[0] is front[0]
    assert not pair_op
    assert out.shape == (M, 80)


def test_without_front_the_pair_runs(monkeypatch, pair_op):
    _use(monkeypatch, qk_pair=True)
    x, xs, ix, ixs = _inputs()
    out = pro.MiniMaxM3ProAttention.forward(
        _attention(None), None, x, hidden_states_scale=xs, index_src=(ix, ixs)
    )
    assert len(pair_op) == 1 and pair_op[0][4] is ix
    assert out.shape == (M, 80)
    assert bool((out[:, :48] == 1).all()) and bool((out[:, 48:] == 2).all())
