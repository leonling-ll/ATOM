# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's decode front: ``qkv_proj`` holds ``[q; k; index q; index k]``,
``qk_proj`` / ``index_qk_proj`` are row views of it, and up to ``FRONT_ROWS`` rows one
GEMM over a ring slot's ``[x; lookahead input]`` (``ProLookaheadState.front_rows``)
gives all four.

CPU (aiter importable): the packing and its views, the ring hand-over through the real
decoder layer over 68 layers (with the mono layer kernel on the MoE layers too, the
compiled kernel faked), the cases that keep two GEMMs. GPU: the one GEMM against an
fp32 reference and against the two, the norms' output arguments, a CUDA graph.
"""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("aiter", reason="the model stack imports aiter")

from aiter import QuantType, dtypes

from atom.config import QuantizationConfig
from atom.distributed import indexer_cp
from atom.model_ops import blockscale
from atom.model_ops import layernorm as ln
from atom.model_ops import linear as linear_mod
from atom.model_ops.layernorm import GemmaRMSNorm
from atom.model_ops.minimax_m3 import pro_attn
from atom.model_ops.minimax_m3.pro_attn import ProVariants
from atom.models.minimax_m3 import pro
from atom.models.minimax_m3.pro_mono import runner as mono_runner
from atom.models.minimax_m3.pro_mono.layout import DIMS
from atom.models.minimax_m3.pro_mono.weights import lookahead_reader
from atom.quant_spec import LayerQuantConfig

U8, FP8, E8M0 = torch.uint8, torch.float8_e4m3fn, torch.float8_e8m0fnu
gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# one TP8 rank of Pro: 16 q heads x 256, the K head, 16 index heads x 128
PRO = {"hidden": 7168, "head": 256, "heads": 16, "idx_heads": 16, "idx_dim": 128}
# small shapes for the CPU tests, 32-row aligned like Pro's (q, k | index q, index k)
SMALL = {"hidden": 256, "head": 32, "heads": 2, "idx_heads": 2, "idx_dim": 32}


def _sizes(d):
    """(qk_sizes, index_qk_sizes) of one rank."""
    qk = [d["heads"] * d["head"], d["head"]]
    return qk, [d["idx_heads"] * d["idx_dim"], d["idx_dim"]]


@pytest.fixture
def tp(monkeypatch):
    def _tp(world=1, rank=0):
        group = SimpleNamespace(world_size=world, rank_in_group=rank)
        monkeypatch.setattr(linear_mod, "get_tp_group", lambda: group)
        monkeypatch.setattr(linear_mod, "native_group32_runs_bf16", lambda: False)
        monkeypatch.setattr(indexer_cp, "indexer_cp_enabled", lambda: False)

    return _tp


def _mx_config() -> QuantizationConfig:
    qc = QuantizationConfig()
    qc.global_spec = LayerQuantConfig(
        quant_type=QuantType.per_1x32,
        quant_dtype=dtypes.fp8,
        weight_block_size=(1, 32),
        is_dynamic=True,
    )
    return qc


def _ckpt(d, tp, seed, device="cpu"):
    """Checkpoint tensors (all ranks' q heads): E4M3 rows and their E8M0 scales."""
    g = torch.Generator().manual_seed(seed)
    rows = {
        "q": d["heads"] * tp * d["head"],
        "k": d["head"],
        "index_q": d["idx_heads"] * d["idx_dim"],
        "index_k": d["idx_dim"],
    }
    out = {}
    for name, n in rows.items():
        w = (torch.randn(n, d["hidden"], generator=g) * 0.05).to(FP8)
        s = torch.randint(118, 124, (n, d["hidden"] // 32), generator=g, dtype=U8)
        out[name] = (w.to(device), s.view(E8M0).to(device))
    return out


def _linears(d, tp):
    """The packed projection and the two the base built (``QKVParallelLinear`` q + k,
    ``MergedReplicatedLinear`` index q + index k)."""
    qc, (_, (iq, ik)) = _mx_config(), _sizes(d)
    pre = "model.layers.0.self_attn."
    packed = linear_mod.MinimaxM3QKVParallelLinearWithIndexer(
        d["hidden"],
        d["head"],
        d["heads"] * tp,
        1,
        d["idx_heads"],
        d["idx_dim"],
        quant_config=qc,
        prefix=pre + "qkv_proj",
        v_head_size=0,
        replicate_index_heads=True,
    )
    qk = linear_mod.QKVParallelLinear(
        d["hidden"],
        d["head"],
        d["heads"] * tp,
        1,
        quant_config=qc,
        prefix=pre + "qk_proj",
        v_head_size=0,
    )
    idx = linear_mod.MergedReplicatedLinear(
        d["hidden"], [iq, ik], quant_config=qc, prefix=pre + "index_qk_proj"
    )
    return packed, qk, idx


def _load(layers, ck):
    packed, qk, idx = layers
    for name, (w, s) in ck.items():
        for layer, shard in (
            (packed, name),
            (
                (qk, name)
                if name in ("q", "k")
                else (idx, ["index_q", "index_k"].index(name))
            ),
        ):
            layer.weight_loader(layer.weight, w, shard)
            layer.weight_loader(layer.weight_scale, s, shard)


def _process(layers):
    for layer in layers:
        layer.needs_preshuffled_weight = True  # shuffled on every arch, as on gfx950
        layer.process_weights_after_loading()


class _Sink(nn.Module):
    """The attention's place: keeps the four projections; returns ``out(q, k, iq, ik)``."""

    def __init__(self, out=None):
        super().__init__()
        self.got, self.out, self.impl = [], out, SimpleNamespace()

    def forward(self, q, k, iq, positions, qkv=None):
        self.got.append((q, k, iq, qkv))
        return self.out(q, k, iq, qkv) if self.out else q.new_zeros(q.shape[0], 1)


def _attention(qkv_proj, d, out=None):
    a = pro.MiniMaxM3ProAttention.__new__(pro.MiniMaxM3ProAttention)
    nn.Module.__init__(a)
    a.qkv_proj = qkv_proj
    a.qk_sizes, a.index_qk_sizes = _sizes(d)
    a._views = None
    a.attn, a.o_proj = _Sink(out), nn.Identity()
    return a


# --- CPU: the packing ---------------------------------------------------------------


@pytest.mark.parametrize("rank", [0, 5])
def test_packed_weight_is_the_two_linears_bytes(tp, rank):
    tp(8, rank)
    d, ck = SMALL, _ckpt(SMALL, 8, seed=rank)
    layers = _linears(d, 8)
    _load(layers, ck)
    packed, qk, idx = layers
    nq = d["heads"] * d["head"]
    lo = rank * nq
    want = [ck["q"][0][lo : lo + nq], ck["k"][0], ck["index_q"][0], ck["index_k"][0]]
    want_s = [ck["q"][1][lo : lo + nq], ck["k"][1], ck["index_q"][1], ck["index_k"][1]]
    assert torch.equal(packed.weight.view(U8), torch.cat(want).view(U8))
    assert torch.equal(packed.weight_scale.view(U8), torch.cat(want_s).view(U8))

    _process(layers)
    views = _attention(packed, d)._row_views()
    n_qk = sum(_sizes(d)[0])
    for view, base, start in ((views[0], qk, 0), (views[1], idx, n_qk)):
        assert view.weight.is_shuffled and base.weight.is_shuffled
        # the shuffle of the concatenation, sliced at a 32-row boundary, is the shuffle
        # of each part: the same bytes as the separately stored linear
        assert torch.equal(view.weight.view(U8), base.weight.view(U8))
        assert torch.equal(view.weight_scale.view(U8), base.weight_scale.view(U8))
        # and the views own nothing: they read the packed tensors
        w, s = packed.weight.data, packed.weight_scale.data
        assert view.weight.data_ptr() == w.data_ptr() + start * w.stride(0)
        assert view.weight_scale.data_ptr() == s.data_ptr() + start * s.stride(0)
        assert view.output_size == base.output_size


def test_flash_packing_keeps_its_layout(tp):
    tp(8, 3)
    flash = linear_mod.MinimaxM3QKVParallelLinearWithIndexer(
        64, 32, 16, 8, 8, 32, quant_config=_mx_config(), prefix="qkv_proj"
    )
    got = {
        s: flash._shard_offset_size(s) for s in ("q", "k", "v", "index_q", "index_k")
    }
    assert got == {
        "q": (0, 64),
        "k": (64, 32),
        "v": (96, 32),
        "index_q": (128, 32),
        "index_k": (160, 32),
    }
    assert flash.output_size == 192 and not flash.index_q_replicated
    with pytest.raises(ValueError, match="shard like KV heads"):
        linear_mod.MinimaxM3QKVParallelLinearWithIndexer(
            64, 32, 16, 8, 4, 32, quant_config=_mx_config(), prefix="qkv_proj"
        )


# --- CPU: the ring hand-over through the decoder layer --------------------------------


class _Gemm:
    """``qkv_proj``'s place, and its row views': an exact stand-in GEMM, ``y[r, c] =
    1000 r + c`` over its own columns, recording each input."""

    native_a8_group_rows = 1

    def __init__(self, calls, start=0, n=None, name="qkv", runs_bf16=False):
        self.calls, self.start, self.name = calls, start, name
        self.runs_bf16 = runs_bf16
        qk, idx = _sizes(SMALL)
        self.n = sum(qk + idx) if n is None else n

    def make_row_view(self, start, n):
        return _Gemm(self.calls, start, n, "index" if start else "qk", self.runs_bf16)

    def __call__(self, x, x_scale=None):
        # copies: the ring rows are rewritten by later layers
        scale = None if x_scale is None else x_scale.clone()
        self.calls.append((self.name, x.clone(), scale))
        r = torch.arange(x.shape[0], dtype=torch.float32)[:, None]
        return r * 1000 + torch.arange(self.start, self.start + self.n)


class _Norms:
    """The MXFP8 norms' place: fresh random bytes into the given rows (or new tensors),
    remembered as layer ``cur``'s input or, inside ``put``, its reader's lookahead."""

    def __init__(self):
        self.g = torch.Generator().manual_seed(0)
        self.cur, self.in_put, self.x, self.la = None, None, {}, {}
        self.x_ignored = False  # the input norm allocates its own output

    def mx(self, x, residual, norm, write_bf16=True, q_out=None, q_scale_out=None):
        m, h = x.shape
        if self.in_put is None and self.x_ignored:
            q_out = None
        if q_out is None:
            q_out = torch.empty(m, h, dtype=FP8)
            q_scale_out = torch.empty(m, h // 32, dtype=E8M0)
        q_out.view(U8).copy_(torch.randint(0, 120, (m, h), generator=self.g, dtype=U8))
        q_scale_out.view(U8).copy_(
            torch.randint(0, 255, (m, h // 32), generator=self.g, dtype=U8)
        )
        seen = (q_out.view(U8).clone(), q_scale_out.view(U8).clone())
        if self.in_put is None:
            self.x[self.cur] = seen
        else:
            self.la[self.in_put + 2] = seen
        assert residual is None  # the input norm of layer 0 and the lookahead norm
        return (x if write_bf16 else None), q_out, q_scale_out, None

    def ar(
        self,
        h,
        residual,
        norm,
        lamport_rows=0,
        fused_rows=0,
        addend=None,
        q_out=None,
        q_scale_out=None,
    ):
        assert addend is None  # moe_tail_ar is off here
        _, q, s, _ = self.mx(h, None, norm, False, q_out, q_scale_out)
        return h + residual, q, s, h + residual

    @staticmethod
    def ar_plain(h, residual, norm, addend=None):
        assert addend is None
        return h + residual, h + residual


class _TorchNorm(GemmaRMSNorm):
    def forward(self, x, residual=None):
        return self.forward_native(x, residual)

    forward_cuda = forward


class _MLP(nn.Module):
    gate_up_proj = SimpleNamespace(native_a8_group_rows=None)

    def forward(self, x, *args, **kwargs):
        return x


def _state(ahead=2, start=2):
    return pro.ProLookaheadState(
        SimpleNamespace(
            sparse_attention_config={
                "sparse_indexer_lookahead_num_layers": ahead,
                "sparse_indexer_lookahead_start_layer": start,
            }
        )
    )


class _MonoRoute:
    """``ProMonoRoute`` taking every MoE layer at every width."""

    def __init__(self, runner):
        self.runner = runner

    def take(self, layer_id, rows):
        return self.runner


def _mono(monkeypatch, st, layers, rows, norms):
    """A ``ProMonoRunner`` over ``layers`` whose ``segment`` and ``launch`` are the real
    ones and whose compiled kernel writes zeros to every output it is handed and, as the
    lookahead norm, fresh random bytes to ``la8`` / ``la8s`` (remembered as the reader's
    lookahead, as ``_Norms`` does for ``put``). Returns the layers launched."""
    h = SMALL["hidden"]
    # the outputs at the stand-ins' hidden size
    monkeypatch.setattr(
        mono_runner,
        "OUTPUTS",
        {
            name: (dtype, h // 32 if name.endswith("s") else h)
            for name, (dtype, _) in mono_runner.OUTPUTS.items()
        },
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "stream")
    r = mono_runner.ProMonoRunner.__new__(mono_runner.ProMonoRunner)
    r.state, r.check, r.debug, r.rank = st, False, False, 0
    w = SimpleNamespace(
        w_o=torch.zeros(1), s_o=torch.zeros(1), g_post=torch.zeros(1),
        g_ahead=torch.zeros(1), eps=1e-6, eps_ahead=1e-6,
    )  # fmt: skip
    table = torch.zeros(1, dtype=torch.int64)
    r.layers = {
        lid: mono_runner.MonoLayer(w, lookahead_reader(lid, st), table)
        for lid in layers
    }
    r.first, r.last = min(layers), max(layers)
    r.scratch, r.plain = torch.zeros(1, dtype=U8), torch.zeros(1, dtype=U8)
    r.peers = SimpleNamespace(addresses=torch.zeros(1, dtype=torch.int64))
    r.mailboxes = SimpleNamespace(begin_step=lambda: None)
    r.builds = {rows: {lw.kind: "launcher" for lw in r.layers.values()}}
    launched = []

    def kernel(launcher, *packed):
        args = dict(zip(mono_runner.ABI.names, packed[:-1]))
        launched.append(args["layer"])
        for name, (dtype, cols) in mono_runner.OUTPUTS.items():
            nbytes = rows * cols * torch.empty(0, dtype=dtype).element_size()
            if args.get(name, 0) and name not in mono_runner.LOOKAHEAD_OUTPUTS:
                ctypes.memset(args[name], 0, nbytes)
        if args["la8"]:
            seen = (
                torch.randint(0, 120, (rows, h), generator=norms.g, dtype=U8),
                torch.randint(0, 255, (rows, h // 32), generator=norms.g, dtype=U8),
            )
            ctypes.memmove(args["la8"], seen[0].data_ptr(), seen[0].numel())
            ctypes.memmove(args["la8s"], seen[1].data_ptr(), seen[1].numel())
            norms.la[args["layer"] + 2] = seen

    monkeypatch.setattr(mono_runner, "_run_compiled", kernel)
    return r, launched


@pytest.fixture
def front(monkeypatch):
    """``run(layers, rows, world, lamport, runs_bf16, x_ignored, mono_from,
    **variants)``: that many real ``MiniMaxM3ProDecoderLayer`` forwards over the
    stand-ins above; from layer ``mono_from`` on (None: none) MoE layers the mono
    route takes (``_mono``)."""
    norms = _Norms()
    monkeypatch.setattr(pro, "gemma_rms_norm_mxfp8", norms.mx)
    monkeypatch.setattr(pro, "fused_allreduce_gemma_rms_norm_mxfp8", norms.ar)
    monkeypatch.setattr(pro, "fused_allreduce_gemma_rms_norm", norms.ar_plain)
    monkeypatch.setattr(pro, "_aiter_mx_norm", lambda: True)
    monkeypatch.setattr(
        blockscale,
        "quantize_fp8",
        lambda x: (x.to(FP8), x.new_zeros(*x.shape[:-1], x.shape[-1] // 32)),
    )

    def run(
        layers,
        rows,
        world=1,
        lamport=True,
        runs_bf16=False,
        x_ignored=False,
        mono_from=None,
        **variants,
    ):
        norms.x_ignored = x_ignored
        monkeypatch.setattr(ln, "get_tensor_model_parallel_world_size", lambda: world)
        monkeypatch.setattr(ln, "_lamport_ar_on", lambda: lamport)
        monkeypatch.setattr(pro_attn, "_variants", ProVariants(**variants))
        st, calls, h = _state(), [], SMALL["hidden"]
        put = st.put

        def put_traced(layer, ffn_in, resid):
            norms.in_put = layer
            put(layer, ffn_in, resid)
            norms.in_put = None

        st.put = put_traced
        decs = []

        # what a mono layer's attention hands its kernel: bf16 [T, o_proj K]
        def core(*args):
            return torch.zeros(rows, DIMS.o_k, dtype=torch.bfloat16)

        for i in range(layers):
            dec = pro.MiniMaxM3ProDecoderLayer.__new__(pro.MiniMaxM3ProDecoderLayer)
            nn.Module.__init__(dec)
            moe = mono_from is not None and i >= mono_from
            dec._pro_state, dec.layer_num, dec.is_moe_layer = st, i, moe
            dec.self_attn = _attention(
                _Gemm(calls, runs_bf16=runs_bf16), SMALL, core if moe else None
            )
            dec.input_layernorm = _TorchNorm(h)
            dec.post_attention_layernorm = _TorchNorm(h)
            dec.mlp = _MLP()
            st.norms[i] = dec.input_layernorm
            st.index_linears[i] = (dec.self_attn.qkv_proj,)
            decs.append(dec)
        norms.launched = []
        if mono_from is not None:
            runner, norms.launched = _mono(
                monkeypatch, st, range(mono_from, layers), rows, norms
            )
            for dec in decs:
                dec._mono = _MonoRoute(runner)
        x, res, per_layer = torch.randn(rows, h), None, []
        for dec in decs:
            norms.cur, start = dec.layer_num, len(calls)
            x, res = dec(torch.arange(rows), x, res)
            per_layer.append((calls[start:], dec.self_attn.attn.got[-1]))
        return per_layer, norms

    return run


@pytest.mark.parametrize("mono_from", [None, 4], ids=["mono off", "mono on"])
@pytest.mark.parametrize("rows", [1, 8, 16])
def test_every_layer_reads_the_lookahead_put_two_layers_back(front, rows, mono_from):
    """With mono on, the MoE layers 4-67 run the layer kernel, whose lookahead norm
    (``la8`` / ``la8s``) feeds readers 6-67: it must land in the reader's ring rows as
    ``put``'s does (layers 0-3) for the reader's one GEMM."""
    per_layer, norms = front(68, rows, mono_from=mono_from)
    assert norms.launched == ([] if mono_from is None else list(range(4, 68)))
    n_qk = sum(_sizes(SMALL)[0])
    for layer, (calls, (q, k, iq, ik)) in enumerate(per_layer):
        assert [c[0] for c in calls] == ["qkv"], layer  # one GEMM
        _, a, a_scale = calls[0]
        x, x_scale = norms.x[layer]
        assert torch.equal(a[:rows].view(U8), x)
        assert torch.equal(a_scale[:rows].view(U8), x_scale)
        if layer < 2:  # no lookahead: the indexer reads x, T rows
            assert a.shape[0] == rows
        else:  # rows [T, 2T): the lookahead norm's bytes, put (or mono) by layer - 2
            la, la_scale = norms.la[layer]
            assert a.shape[0] == 2 * rows
            assert torch.equal(a[rows:].view(U8), la)
            assert torch.equal(a_scale[rows:].view(U8), la_scale)
        # q, k from the first T rows' q / k columns; index q, k from the last T rows'
        idx_row = rows if layer >= 2 else 0
        assert q[0, 0] == 0 and k[0, 0] == q.shape[1] and q.shape[0] == rows
        assert iq[0, 0] == 1000 * idx_row + n_qk and ik[0, 0] == iq[0, 0] + iq.shape[1]
        assert iq.shape[0] == ik.shape[0] == rows
        last = 1000 * (idx_row + rows - 1) + n_qk + iq.shape[1] + ik.shape[1] - 1
        assert ik[-1, -1] == last


@pytest.mark.parametrize(
    "case, kw",
    [
        ("past FRONT_ROWS", {"rows": pro.FRONT_ROWS + 8}),
        ("TP 8, Lamport off", {"world": 8, "lamport": False, "ar_lamport_rows": 16}),
        ("TP 8, ar_lamport_rows 0", {"world": 8}),
        ("diag", {"diag": True}),
        ("fuse_mxfp8_quant off", {"fuse_mxfp8_quant": False}),
        ("runs_bf16", {"runs_bf16": True}),
        ("lookahead_src ffn_in", {"lookahead_src": "ffn_in"}),
        # a route that allocates its own output, e.g. one the route check does not know
        ("input norm ignores q_out", {"x_ignored": True}),
    ],
)
def test_two_gemms_where_the_front_does_not_apply(front, case, kw):
    kw.setdefault("rows", 8)
    per_layer, _ = front(5, **kw)
    for layer, (calls, (q, k, iq, ik)) in enumerate(per_layer):
        names = [c[0] for c in calls]
        assert names[:2] == ["qk", "index"] and "qkv" not in names, (case, layer)
        assert q.shape[0] == iq.shape[0] == kw["rows"]


def test_the_lamport_route_takes_the_front(front):
    per_layer, _ = front(5, 8, world=8, lamport=True, ar_lamport_rows=16)
    assert all([c[0] for c in calls] == ["qkv"] for calls, _ in per_layer)


# --- GPU: the GEMM, the norms, a CUDA graph -------------------------------------------


def _dequant(q, s):
    """fp32 values of MXFP8 rows: E4M3 x 2^(E8M0 - 127) per 32 columns."""
    scale = torch.exp2(s.view(U8).float() - 127).repeat_interleave(32, dim=-1)
    return q.float() * scale


def _rel(got, ref):
    return float((got.float() - ref).abs().max() / ref.abs().max())


def _mx_rows(m, hidden, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = (torch.randn(m, hidden, device="cuda", generator=g) * 0.5).to(FP8)
    s = torch.randint(118, 124, (m, hidden // 32), device="cuda", generator=g, dtype=U8)
    return q, s.view(E8M0)


@pytest.fixture
def pro_linears(tp):
    tp()
    ck = _ckpt(PRO, 1, seed=7, device="cuda")
    with torch.device("cuda"):
        layers = _linears(PRO, 1)
    _load(layers, ck)
    _process(layers)
    return layers, ck


@gpu
@pytest.mark.parametrize("rows", [1, 8, 16])
@pytest.mark.parametrize("lookahead", [False, True])
def test_one_gemm_matches_the_reference_and_the_two(pro_linears, rows, lookahead):
    (packed, base_qk, base_idx), ck = pro_linears
    attn = _attention(packed, PRO)
    a = _mx_rows(2 * rows if lookahead else rows, PRO["hidden"], seed=rows)
    x = (a[0][:rows], a[1][:rows])
    la = (a[0][rows:], a[1][rows:]) if lookahead else None
    pos = torch.arange(rows, device="cuda")
    attn(pos, *x, index_src=la, front=a)
    one = attn.attn.got[-1]
    attn(pos, *x, index_src=la)
    two = attn.attn.got[-1]
    # the two GEMMs through the row views are the base's linears, bit for bit
    assert torch.equal(torch.cat(two[:2], -1), base_qk(*x))
    assert torch.equal(torch.cat(two[2:], -1), base_idx(*(la or x)))

    ix = _dequant(*(la or x))
    refs = [_dequant(*x) @ _dequant(*ck[n]).T for n in ("q", "k")]
    refs += [ix @ _dequant(*ck[n]).T for n in ("index_q", "index_k")]
    for name, g1, g2, ref in zip(("q", "k", "index q", "index k"), one, two, refs):
        e1, e2 = _rel(g1, ref), _rel(g2, ref)
        print(f"rows {rows} lookahead {lookahead} {name}: one {e1:.2e} two {e2:.2e}")
        assert e1 <= 2**-7 and e2 <= 2**-7, (name, e1, e2)


@gpu
@pytest.mark.parametrize("rows", [1, 8, 16])
def test_norms_write_into_the_given_ring_rows(monkeypatch, rows):
    import aiter
    from aiter.dist import communication_op

    h = PRO["hidden"]
    g = torch.Generator(device="cuda").manual_seed(rows)
    norm = GemmaRMSNorm(h).cuda()
    w, x, res = torch.randn(3, rows, h, device="cuda", generator=g).to(torch.bfloat16)
    norm.weight.data = w[0] * 0.3
    ring = (
        torch.full((3, 32, h), 0x55, dtype=U8, device="cuda").view(FP8),
        torch.full((3, 32, h // 32), 0x66, dtype=U8, device="cuda").view(E8M0),
    )

    def check(new, got, slot, lo):
        want_q, want_s = ring[0].view(U8).clone(), ring[1].view(U8).clone()
        want_q[slot, lo : lo + rows] = new[1].view(U8)
        want_s[slot, lo : lo + rows] = new[2].view(U8)
        assert got[1].data_ptr() == ring[0][slot, lo].data_ptr()
        assert got[2].data_ptr() == ring[1][slot, lo].data_ptr()
        assert torch.equal(ring[0].view(U8), want_q)
        assert torch.equal(ring[1].view(U8), want_s)
        for a, b in zip(new, got):
            assert (a is None and b is None) or torch.equal(a.view(U8), b.view(U8))

    def into(slot, lo):
        return ring[0][slot, lo : lo + rows], ring[1][slot, lo : lo + rows]

    new = ln.gemma_rms_norm_mxfp8(x, None, norm, write_bf16=False)
    check(new, ln.gemma_rms_norm_mxfp8(x, None, norm, False, *into(1, rows)), 1, rows)
    if not hasattr(aiter, "add_rmsnorm_mxfp8"):
        pytest.skip("this aiter has no add_rmsnorm_mxfp8")
    monkeypatch.setattr(
        communication_op, "tensor_model_parallel_all_reduce", lambda t: t
    )
    new = ln._lamport_ar_rms_norm_mxfp8(x, res, norm)
    check(new, ln._lamport_ar_rms_norm_mxfp8(x, res, norm, *into(2, 0)), 2, 0)


@gpu
@pytest.mark.parametrize("rows", [8, 16])
def test_graph_replay_equals_eager(monkeypatch, pro_linears, rows):
    (packed, _, _), _ = pro_linears
    if not pro._aiter_mx_norm():
        pytest.skip("this aiter has no MXFP8 norms")
    monkeypatch.setattr(ln, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(pro_attn, "_variants", ProVariants())
    h, n = PRO["hidden"], sum(sum(s) for s in _sizes(PRO))
    st, seen = _state(), []
    packed.register_forward_pre_hook(
        lambda m, args: seen.append((m.prefix, args[0].shape[0]))
    )

    def out(q, k, iq, ik):  # the next layer's input follows every projection
        return nn.functional.pad(torch.cat([q, k, iq, ik], -1), (0, h - n)) * 0.01

    decs = []
    for i in range(6):
        dec = pro.MiniMaxM3ProDecoderLayer.__new__(pro.MiniMaxM3ProDecoderLayer)
        nn.Module.__init__(dec)
        dec._pro_state, dec.layer_num, dec.is_moe_layer = st, i, False
        dec.self_attn, dec.mlp = _attention(packed, PRO, out), _MLP()
        dec.input_layernorm = GemmaRMSNorm(h).cuda()
        dec.post_attention_layernorm = GemmaRMSNorm(h).cuda()
        for norm in (dec.input_layernorm, dec.post_attention_layernorm):
            norm.weight.data = torch.randn(h, device="cuda").to(torch.bfloat16) * 0.3
        st.norms[i], st.index_linears[i] = dec.input_layernorm, (packed,)
        decs.append(dec)
    x_in = torch.randn(rows, h, device="cuda").to(torch.bfloat16)
    pos = torch.arange(rows, device="cuda")

    def run():
        x, res = x_in, None
        for dec in decs:
            x, res = dec(pos, x, res)
        return x.clone(), res.clone()

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()  # warm-up, as the runner's eager forward before a capture
    torch.cuda.current_stream().wait_stream(side)
    seen.clear()
    torch.cuda.set_sync_debug_mode("error")
    try:
        eager = run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    # one packed GEMM a layer, on T rows (layers 0, 1) or [x; lookahead] (2T)
    assert seen == [(packed.prefix, rows)] * 2 + [(packed.prefix, 2 * rows)] * 4
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(eager, captured):
        assert torch.equal(a, b)
