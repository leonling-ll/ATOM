# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Pro mono phase 1's gate (``pro_mono.route``) and lookahead rule, on the CPU.

The gate decides, at every MoE layer, whether the layer's FFN segment runs on
the in-kernel all-reducing layer kernel, so each decision must be one every
TP rank takes alike, and with the phase off nothing of mono may run: no
runner, no collective past the one latch, no kernel."""

from types import SimpleNamespace

import pytest
import torch

from atom.models.minimax_m3.pro_mono import route as route_mod
from atom.models.minimax_m3.pro_mono.config import MAX_ROWS
from atom.models.minimax_m3.pro_mono.route import ProMonoRoute, variants_refusal
from atom.models.minimax_m3.pro_mono.weights import lookahead_reader
from atom.mono.runtime.consensus import MonoUnsupported


def _variants(**over):
    """The ``ProVariants`` fields the route reads (the dataclass lives in
    ``pro_attn``, which needs aiter)."""
    fields = {
        "mono_ffn_rows": 16,
        "diag": False,
        "fuse_mxfp8_quant": True,
        "lookahead": True,
        "lookahead_src": "resid_in",
        "ar_lamport_rows": 16,
        "mono_routed_act": "fp8",
    }
    fields.update(over)
    return SimpleNamespace(**fields)


class FakeRunner:
    def __init__(self, layers=None, widths=None):
        self.layers = set(range(4, 68) if layers is None else layers)
        self.widths = widths  # None: every width builds
        self.prepared = []

    def prepare(self, rows):
        self.prepared.append(rows)
        return self.widths is None or rows in self.widths

    def serves(self, layer_id):
        return layer_id in self.layers


class Harness:
    """A route over a fake runner, counting its reads and creations."""

    def __init__(self, monkeypatch, refusal=None, runner=None, **variants):
        self.reads = 0
        self.created = 0
        self.runner = runner or FakeRunner()
        self.v = _variants(**variants)
        self.mx_norm = True
        self.fwd = SimpleNamespace(
            context=SimpleNamespace(is_prefill=False), ubatch_slices=None
        )
        monkeypatch.setattr(route_mod, "get_forward_context", lambda: self.fwd)
        monkeypatch.setattr(route_mod.envs, "ATOM_MONO_CHECK", False)
        self.capturing = False
        monkeypatch.setattr(
            torch.cuda, "is_current_stream_capturing", lambda: self.capturing
        )
        self.route = ProMonoRoute(refusal, self._create, self._read, lambda: None)

    def _create(self):
        self.created += 1
        return self.runner

    def _read(self):
        self.reads += 1
        return self.v, self.mx_norm

    def take(self, rows, layer_id=4, prefill=False):
        self.fwd.context.is_prefill = prefill
        return self.route.take(layer_id, rows)


@pytest.mark.parametrize("limit", [1, 8, MAX_ROWS])
def test_gate_matrix(monkeypatch, limit):
    """Decode and verify forwards of 1 .. limit rows take mono; 0 rows,
    more than the limit, a prefill, micro-batches or no context do not."""
    h = Harness(monkeypatch, mono_ffn_rows=limit)
    for rows in range(1, limit + 1):
        assert h.take(rows) is h.runner, rows
    for rows in (0, limit + 1, MAX_ROWS + 1, 32, 1024):
        assert h.take(rows) is None, rows
    assert h.take(1, prefill=True) is None
    h.fwd.context.is_prefill = False
    h.fwd.ubatch_slices = [object()]
    assert h.route.take(4, 1) is None
    h.fwd.ubatch_slices = None
    h.fwd.context = None
    assert h.route.take(4, 1) is None
    # the runner is created once and a width prepared only when the gate holds
    assert h.created == 1
    assert sorted(set(h.runner.prepared)) == list(range(1, limit + 1))


def test_a_layer_the_runner_did_not_bind_runs_production(monkeypatch):
    h = Harness(monkeypatch, runner=FakeRunner(layers=[4, 5]))
    assert h.take(4, layer_id=5) is h.runner
    assert h.take(4, layer_id=6) is None


def test_a_width_that_did_not_build_runs_production(monkeypatch):
    h = Harness(monkeypatch, runner=FakeRunner(widths={1, 2, 4, 8}))
    assert h.take(8) is h.runner
    assert h.take(3) is None
    assert h.take(4) is h.runner


def test_a_graph_captured_before_the_runner_exists_runs_production(monkeypatch):
    """The runner's allocation and peer handshake cannot be captured: a
    capture before any eager step of the phase records production."""
    h = Harness(monkeypatch)
    h.take(4, prefill=True)  # the latch
    h.capturing = True
    assert h.take(4) is None and h.created == 0
    h.capturing = False
    assert h.take(4) is h.runner
    h.capturing = True
    assert h.take(4) is h.runner


def test_latched_once_on_the_first_forward_even_a_prefill(monkeypatch):
    """Every rank runs the first forward, so the latch's collective is placed
    there whatever its shape; a later edit of the variants is not seen."""
    h = Harness(monkeypatch, mono_ffn_rows=8)
    assert h.take(4, prefill=True) is None
    assert h.reads == 1 and h.route.rows == 8
    h.v.mono_ffn_rows = 16
    h.v.lookahead_src = "ffn_in"
    assert h.take(12) is None
    assert h.take(8) is h.runner
    assert h.reads == 1 and h.route.rows == 8


def test_off_by_default_runs_nothing_of_mono(monkeypatch):
    """``mono_ffn_rows`` 0: one latch, then no runner, no width, no read."""
    h = Harness(monkeypatch, mono_ffn_rows=0)
    for rows in (1, 8, 16):
        assert h.take(rows) is None
    assert (h.reads, h.created, h.runner.prepared) == (1, 0, [])


def test_a_refused_deployment_never_latches(monkeypatch):
    h = Harness(monkeypatch, refusal="TP 4")
    assert h.take(1) is None
    assert (h.reads, h.created) == (0, 0)


def test_a_failed_runner_turns_it_off_for_good(monkeypatch):
    h = Harness(monkeypatch)

    def refuse():
        h.created += 1
        raise MonoUnsupported("layer 4 o_proj: not the checkpoint's native 1x32")

    h.route._mono._create = refuse
    assert h.take(1) is None
    assert h.take(1) is None
    assert h.created == 1


@pytest.mark.parametrize(
    "over, why",
    [
        ({"diag": True}, "the diag variant"),
        ({"fuse_mxfp8_quant": False}, "fuse_mxfp8_quant off"),
        ({"lookahead": False}, "lookahead off"),
        ({"lookahead_src": "ffn_in"}, "lookahead_src 'ffn_in'"),
        ({"lookahead_src": "resid"}, "lookahead_src 'resid'"),
        ({"ar_lamport_rows": 8}, "ar_lamport_rows 8 below 16"),
        ({"ar_lamport_rows": 0}, "ar_lamport_rows 0 below 16"),
    ],
)
def test_variants_the_hand_off_cannot_serve_latch_off(monkeypatch, over, why):
    assert variants_refusal(_variants(**over), True) == why
    h = Harness(monkeypatch, **over)
    assert h.take(1) is None
    assert h.created == 0 and h.route.rows == 0


def test_a_bad_row_limit_is_left_to_the_latch():
    # no TypeError against ar_lamport_rows: latch_rows names the bad value
    assert variants_refusal(_variants(mono_ffn_rows="16"), True) is None


def test_without_the_mxfp8_norms_it_is_off(monkeypatch):
    assert variants_refusal(_variants(), False) == "aiter without the MXFP8 norms"
    assert variants_refusal(_variants(), True) is None
    h = Harness(monkeypatch)
    h.mx_norm = False
    assert h.take(1) is None and h.created == 0


def test_ranks_that_disagree_turn_it_off_on_every_rank(monkeypatch):
    """One rank refused its variants (latching 0) while this one read 16: the
    latch's (bad, max, -min) collective sees 0 .. 16 and serves neither."""
    import atom.distributed.group_agreement as ga

    def max_each(values, group):
        rows = [tuple(values), (0, 0, 0, 0, 0)]
        return tuple(max(col) for col in zip(*rows))

    monkeypatch.setattr(ga, "group_max_each", max_each)
    h = Harness(monkeypatch)
    assert h.take(1) is None
    assert h.created == 0


@pytest.mark.parametrize("act", ["fp4", "fp8"])
def test_the_routed_format_is_latched_with_the_rows(monkeypatch, act):
    """The runner builds with the latched ``mono_routed_act`` (W4A8 by
    default); a later edit is not seen, as for the rows."""
    h = Harness(monkeypatch, mono_routed_act=act)
    assert h.take(1) is h.runner and h.route.routed_act == act
    h.v.mono_routed_act = "fp4" if act == "fp8" else "fp8"
    assert h.take(1) is h.runner and h.route.routed_act == act


def test_an_unknown_routed_format_latches_off(monkeypatch):
    h = Harness(monkeypatch, mono_routed_act="bf16")
    assert h.take(1) is None
    assert h.created == 0 and h.route.rows == 0


def test_check_mode_is_latched_from_the_environment(monkeypatch):
    h = Harness(monkeypatch)
    monkeypatch.setattr(route_mod.envs, "ATOM_MONO_CHECK", True)
    assert h.take(1) is h.runner and h.route.check
    monkeypatch.setattr(route_mod.envs, "ATOM_MONO_CHECK", False)
    assert h.route.check


def _state(ahead=2, start=2, layers=68):
    return SimpleNamespace(
        ahead=ahead, start=start, norms={i: object() for i in range(layers)}
    )


def test_lookahead_reader_is_layer_l_plus_2_while_one_exists():
    st = _state()
    assert [lookahead_reader(lid, st) for lid in (4, 30, 65)] == [6, 32, 67]
    assert lookahead_reader(66, st) is None and lookahead_reader(67, st) is None


def test_no_lookahead_reader_without_the_lookahead_or_before_its_start():
    assert lookahead_reader(4, _state(ahead=0)) is None
    assert lookahead_reader(4, _state(start=7)) is None
    assert lookahead_reader(5, _state(start=7)) == 7
