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
        "mono_routed_act": "bf16",
        "day0_decode_rows": 24,
        "ar_fused_rows": 0,
        "ar_fused_min_rows": 0,
        # read by production's segment, not the route (served as they are)
        "moe_tail_ar": False,
        "moe_two_stream_rows": 0,
        "moe_decode": "aiter",
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
        # AR #1 from 1 row up on the one-launch kernel, not the Lamport sum + norm
        ({"ar_fused_rows": 128}, "ar_fused_rows 128 overlaps 1 .. 16"),
        ({"ar_fused_rows": 1}, "ar_fused_rows 1 overlaps 1 .. 16"),
    ],
)
def test_variants_the_hand_off_cannot_serve_latch_off(monkeypatch, caplog, over, why):
    import logging

    assert variants_refusal(_variants(**over), True) == why
    h = Harness(monkeypatch, **over)
    with caplog.at_level(logging.WARNING, logger=route_mod.logger.name):
        assert h.take(1) is None
    assert h.created == 0 and h.route.rows == 0
    # the off line names the refusal, not the 0 it latched
    assert f"mono FFN off: {why}" in caplog.text


@pytest.mark.parametrize(
    "over, why",
    [
        ({"ar_fused_rows": 128, "ar_fused_min_rows": 16}, None),
        ({"ar_fused_rows": 128, "ar_fused_min_rows": 24}, None),
        # production follows the latched limit (16 here), whatever the knob
        ({"ar_fused_rows": 128, "ar_fused_min_rows": 8}, None),
        ({"ar_fused_rows": 0, "ar_fused_min_rows": 8}, None),
    ],
)
def test_ar_fused_rows_above_the_row_limit_keep_mono(monkeypatch, over, why):
    """ar_fused_min_rows set: mono stays on; production keeps the one-launch op above its rows."""
    assert variants_refusal(_variants(**over), True) == why
    h = Harness(monkeypatch, **over)
    if why is None:
        assert h.take(16) is h.runner and h.route.rows == 16
    else:
        assert h.take(1) is None and h.route.rows == 0


@pytest.mark.parametrize("min_rows", [16, 32])
@pytest.mark.parametrize("day0", [24, 48])
def test_mono_to_32_rows_with_the_one_launch_ar_above_it_keeps_mono(min_rows, day0):
    """mono to 32 rows (a wider build), the one-launch AR and the folds above it: no refusal
    from these variants (the width a build supports is the latch's, not checked here).
    """
    over = {
        "mono_ffn_rows": 32,
        "ar_lamport_rows": 32,
        "ar_fused_rows": 128,
        "ar_fused_min_rows": min_rows,
        "day0_decode_rows": day0,
    }
    assert variants_refusal(_variants(**over), True) is None


@pytest.mark.parametrize(
    "day0, rows", [(24, 16), (17, 16), (0, 16), (9, 8), (16, 16), (1, 16), (24, 32)]
)
def test_a_day0_decode_at_any_row_limit_is_served(monkeypatch, day0, rows):
    """The Agentic recipe's 24 against mono's 16 or 32, the day0 decode off,
    or from within the row limit: the day0 decode hands a mono layer bf16 rows
    (``ProSparseAttentionImpl._mxfp8_out``)."""
    over = {"day0_decode_rows": day0, "mono_ffn_rows": rows}
    if rows > 16:
        over["ar_lamport_rows"] = rows  # AR #1 on the Lamport path at every mono width
    assert variants_refusal(_variants(**over), True) is None
    h = Harness(monkeypatch, **over)
    assert h.take(rows) is h.runner and h.route.rows == rows


def test_a_refused_variant_is_logged_with_its_reason(monkeypatch, caplog):
    import logging

    h = Harness(monkeypatch, ar_fused_rows=128)
    with caplog.at_level(logging.WARNING, logger=route_mod.logger.name):
        assert h.take(1) is None
    assert "off: ar_fused_rows 128 overlaps 1 .. 16" in caplog.text


@pytest.mark.parametrize(
    "over",
    [
        {"moe_tail_ar": True},
        {"moe_two_stream_rows": 64},
        {"moe_decode": "day0"},
        {"moe_tail_ar": True, "moe_two_stream_rows": 64, "moe_decode": "day0"},
    ],
)
def test_variants_the_segment_serves_as_they_are(monkeypatch, over):
    """``moe_tail_ar``, the two-stream MoE and the day0 routed experts change
    production's FFN, which a mono layer does not run (``variants_refusal``)."""
    assert variants_refusal(_variants(**over), True) is None
    h = Harness(monkeypatch, **over)
    assert h.take(1) is h.runner and h.route.rows == 16


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


@pytest.mark.parametrize("act", ["bf16", "fp4", "fp8"])
def test_the_routed_format_is_latched_with_the_rows(monkeypatch, act):
    """The runner builds with the latched ``mono_routed_act`` (A16W4 by
    default); a later edit is not seen, as for the rows."""
    h = Harness(monkeypatch, mono_routed_act=act)
    assert h.take(1) is h.runner and h.route.routed_act == act
    h.v.mono_routed_act = "fp4" if act != "fp4" else "fp8"
    assert h.take(1) is h.runner and h.route.routed_act == act


@pytest.mark.parametrize(
    "act, name", [("bf16", "A16W4"), ("fp8", "W4A8"), ("fp4", "W4A4")]
)
def test_the_latch_line_names_the_routed_format(monkeypatch, caplog, act, name):
    """The latch line names the format (a run's log says which build served:
    the queue scripts grep it)."""
    import logging

    h = Harness(monkeypatch, mono_routed_act=act)
    with caplog.at_level(logging.INFO, logger=route_mod.logger.name):
        assert h.take(1) is h.runner
    assert f"routed experts {name}" in caplog.text


def test_an_unknown_routed_format_latches_off(monkeypatch):
    h = Harness(monkeypatch, mono_routed_act="fp16")
    assert h.take(1) is None
    assert h.created == 0 and h.route.rows == 0


def test_check_mode_is_latched_from_the_environment(monkeypatch):
    h = Harness(monkeypatch)
    monkeypatch.setattr(route_mod.envs, "ATOM_MONO_CHECK", True)
    assert h.take(1) is h.runner and h.route.check
    monkeypatch.setattr(route_mod.envs, "ATOM_MONO_CHECK", False)
    assert h.route.check


# recipes/Agentic-MiniMax-M3.1-Pro.md's ATOM_M3PRO_VARIANTS, which name no mono key
AGENTIC_VARIANTS = {
    "kernels": True, "attn": "aiter", "topk": "split", "prep": "aiter",
    "index_split": True, "score_tq": 8, "ar_lamport_rows": 16, "k_only": True,
    "k_plain_fp8": True, "prefill_attn": "day0", "index_split_decode": 64,
    "moe_route_glue": True, "index_compress": True,
}  # fmt: skip
# recipes/MiniMax-M3-Pro.md's kernel launch: no Lamport AR #1 rows
GENERIC_VARIANTS = {"kernels": True, "attn": "aiter", "topk": "split"}


def _parsed(obj):
    """``obj`` as ``ATOM_M3PRO_VARIANTS`` parses it, ``ProVariants``' own
    defaults filling every field it does not name."""
    pro_attn = pytest.importorskip(
        "atom.model_ops.minimax_m3.pro_attn",
        reason="pro_attn needs aiter, including aiter.ops.triton.fused_kv_cache",
    )
    return pro_attn._parse_variants(obj)


def test_the_agentic_recipe_runs_mono_a16w4_up_to_16_rows_by_default(monkeypatch):
    h = Harness(monkeypatch)
    h.v = _parsed(AGENTIC_VARIANTS)
    assert h.take(1) is h.runner and h.take(16) is h.runner
    assert h.take(17) is None
    assert (h.route.rows, h.route.routed_act) == (16, "bf16")


def test_a_launch_without_the_lamport_rows_keeps_production_and_says_why(
    monkeypatch, caplog
):
    import logging

    h = Harness(monkeypatch)
    h.v = _parsed(GENERIC_VARIANTS)
    with caplog.at_level(logging.WARNING, logger=route_mod.logger.name):
        assert h.take(1) is None
    assert h.created == 0 and h.route.rows == 0
    assert "off: ar_lamport_rows 0 below 16" in caplog.text


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
