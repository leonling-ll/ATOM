# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's lookahead hand-over: which state a lookahead layer's indexer reads.

``ProLookaheadState`` keeps the last ``ahead`` layers' FFN input and residual; a layer
``L >= start`` reads layer ``L - ahead``'s, in the form its ``lookahead_src`` variant names.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="pro imports the ATOM model stack")

from atom.model_ops.minimax_m3 import pro_attn
from atom.model_ops.minimax_m3.pro_attn import ProVariants
from atom.models.minimax_m3 import pro


def _state(ahead=2, start=2):
    cfg = SimpleNamespace(
        sparse_attention_config={
            "sparse_indexer_lookahead_num_layers": ahead,
            "sparse_indexer_lookahead_start_layer": start,
            "sparse_indexer_lookahead_source": "pre_ffn_norm",
        }
    )
    return pro.ProLookaheadState(cfg)


@pytest.fixture
def use(monkeypatch):
    """Run the state under the given variants."""

    def _use(**kw):
        monkeypatch.setattr(pro_attn, "_variants", ProVariants(**kw))

    return _use


# stands in for the reading layer's input_layernorm: Gemma RMSNorm, x * rsqrt(mean(x^2) + eps) * (1 + w)
_W = torch.tensor([0.0, 1.0, -0.5, 2.0])
_NORM = SimpleNamespace(weight=SimpleNamespace(data=_W), variance_epsilon=1e-6)


def _normed(x):
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * (1.0 + _W)


def _resid(layer):
    return torch.tensor([[1.0, 2.0, 3.0, 4.0 + layer]] * 3)


def _fill(st, layers):
    for lay in layers:
        st.put(lay, torch.full((3, 4), float(lay)), _resid(lay))


def test_default_reads_the_residual_two_back_through_the_readers_input_norm(use):
    use()
    st = _state()
    _fill(st, [0, 1])
    got = st.source_for(2, _NORM)
    assert torch.allclose(got, _normed(_resid(0)))  # layer 0's residual, normed


def test_ffn_in_source_is_the_ffn_norm_output(use):
    use(lookahead_src="ffn_in")
    st = _state()
    _fill(st, [0, 1])
    assert torch.equal(st.source_for(2, _NORM), torch.full((3, 4), 0.0))


def test_raw_residual_source_is_not_normed(use):
    use(lookahead_src="resid")
    st = _state()
    _fill(st, [0, 1])
    assert torch.equal(st.source_for(3, _NORM), _resid(1))


def test_layers_before_start_and_lookahead_off_read_their_own_input(use):
    st = _state(start=2)
    use()
    _fill(st, [0, 1])
    assert st.source_for(0, _NORM) is None and st.source_for(1, _NORM) is None
    use(lookahead=False)
    assert st.source_for(2, _NORM) is None


def test_missing_state_gives_none(use):
    use()
    st = _state()
    assert (
        st.source_for(2, _NORM) is None
    )  # layer 0 never ran: fall back to the layer's own input


def test_only_the_last_ahead_layers_are_kept_and_entries_are_copies(use):
    use()
    st = _state()
    x, r = torch.zeros(2, 4), torch.ones(2, 4)
    st.put(0, x, r)
    x.fill_(5.0)
    r.fill_(7.0)  # the producer reuses its buffers
    assert torch.equal(st.resid[0], torch.ones(2, 4))
    st.put(1, x, r)
    st.put(2, x, r)
    assert 0 not in st.resid and set(st.resid) == {1, 2}


class _Norm:
    """A registered input norm; ``forward_cuda`` is aiter's one-kernel Gemma RMSNorm."""

    weight = SimpleNamespace(data=_W)
    variance_epsilon = 1e-6

    def forward_cuda(self, x):
        return _normed(x).to(x.dtype)


def test_registered_norms_norm_the_residual_as_it_is_put(use):
    use()
    st = _state()
    st.norms = {i: _Norm() for i in range(4)}
    x, r = torch.zeros(3, 4), _resid(0).clone()
    st.put(0, x, r)
    r.fill_(9.0)  # the next layer's fused add updates the residual in place
    st.put(1, x, _resid(1))
    st.put(2, x, _resid(2))  # its reader, layer 4, is past the last layer
    assert torch.allclose(st.source_for(2, None), _normed(_resid(0)))
    assert torch.allclose(st.source_for(3, None), _normed(_resid(1)))
    assert not st.resid and not st.resid_in  # nothing copied, each entry read once

    late = _state(start=3)
    late.norms = st.norms
    late.put(0, x, r)  # its reader, layer 2, has no lookahead
    assert not late.resid_in and not late.resid

    use(lookahead_src="resid")  # the raw residual still goes through the copy
    st.put(0, x, _resid(0))
    st.put(1, x, _resid(1))
    assert torch.equal(st.source_for(3, None), _resid(1)) and not st.resid_in


def test_projections_sharing_an_input_quantise_it_once(monkeypatch):
    calls = []
    fake = SimpleNamespace(quantize_fp8=lambda x: calls.append(x) or ("q", "s"))
    monkeypatch.setitem(sys.modules, "atom.model_ops.blockscale", fake)
    native, plain = SimpleNamespace(native_a8_group_rows=1), SimpleNamespace()
    x = torch.ones(2, 32)
    assert (
        pro._quantized_once(x, None, [native, native]) == ("q", "s") and len(calls) == 1
    )
    # a linear that quantises otherwise, or an input already quantised: as it is
    assert pro._quantized_once(x, None, [native, plain]) == (x, None)
    scale = torch.ones(2, 1)
    assert (
        pro._quantized_once(x, scale, [native, native]) == (x, scale)
        and len(calls) == 1
    )


def test_diag_offers_every_candidate(use):
    use(diag=True)
    st = _state()
    _fill(st, [0, 1])
    src = st.diag_sources(2, _NORM)
    assert set(src) == {"ffn", "res", "resn"}
    assert torch.allclose(src["resn"], _normed(src["res"]))
    assert st.diag_sources(1, _NORM) == {}
