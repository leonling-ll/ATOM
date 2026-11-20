# SPDX-License-Identifier: MIT
"""MiniMax-M3-Pro's torch attention: the parts that can be pinned down on a CPU.

The reference attention reads its inputs from the paged caches, so the tests build
the impl without its base constructor (which needs a device) and set the few
attributes the helpers read. What is asserted is the op's definition -- which blocks
the selector keeps, that rope leaves the non-rope dims alone, that attention is causal
and agrees with itself between the prefill and the decode path -- not a second
implementation of it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="pro_attn imports the ATOM attention base")

from atom.config import is_minimax_m3_pro_config
from atom.model_ops.minimax_m3 import pro_attn
from atom.model_ops.minimax_m3.pro_attn import ProSparseAttentionImpl, ProVariants


def _impl(
    *,
    page=4,
    topk=3,
    init=1,
    local=2,
    heads=2,
    head_dim=16,
    rot=4,
    idx_heads=2,
    idx_dim=8,
    vdim=12,
):
    m = object.__new__(ProSparseAttentionImpl)
    m.page, m.topk, m.init_blocks, m.local_blocks = page, topk, init, local
    m.num_heads, m.head_dim, m.rotary_dim = heads, head_dim, rot
    m.num_idx_heads, m.index_head_dim, m.value_head_dim = idx_heads, idx_dim, vdim
    m.scale = head_dim**-0.5
    # (a plain holder: the impl is an nn.Module that never ran its __init__)
    m.__dict__["index_compress_query"] = SimpleNamespace(
        data=torch.randn(idx_dim, generator=torch.Generator().manual_seed(7))
    )
    return m


class TestConfigDetection:
    def _cfg(self, **sparse):
        return SimpleNamespace(
            text_config=SimpleNamespace(sparse_attention_config=sparse)
        )

    def test_pro_has_shared_kv_sparse_attention(self):
        cfg = self._cfg(use_sparse_attention=True, sparse_enable_kv_share=True)
        assert is_minimax_m3_pro_config(cfg)

    def test_flash_is_not_pro(self):
        assert not is_minimax_m3_pro_config(self._cfg(use_sparse_attention=True))

    def test_dense_is_not_pro(self):
        assert not is_minimax_m3_pro_config(self._cfg(sparse_enable_kv_share=True))

    def test_no_sparse_config(self):
        assert not is_minimax_m3_pro_config(
            SimpleNamespace(text_config=SimpleNamespace())
        )


class TestVariants:
    def test_defaults(self):
        v = ProVariants()
        assert v.mscale == "hf" and v.v_src == "post" and v.scale_mult == 1.0
        assert not v.dense_only and v.rope_style == "neox"
        # the reference implementation's indexer: the best pooled centroid, max over heads
        assert (v.block_score, v.head_reduce, v.pool_scale) == (
            "pool4",
            "max",
            128**-0.5,
        )
        assert v.lookahead and v.lookahead_src == "resid_in" and v.index_rope

    def test_unknown_key_is_refused(self):
        with pytest.raises(ValueError, match="unknown keys"):
            pro_attn._parse_variants({"no_such_switch": 1})

    def test_known_keys_parse(self):
        v = pro_attn._parse_variants({"head_reduce": "max", "init_blocks": 0})
        assert v.head_reduce == "max" and v.init_blocks == 0


class TestSelection:
    """`_select_blocks`: top-k of the head-reduced scores, forced blocks first."""

    def _select(self, impl, scores, qblock):
        # scores [R, NI, NB], reduced over the NI heads as the defaults do
        return impl._select_blocks(scores, torch.tensor(qblock), ProVariants())

    def test_everything_visible_is_selected_when_it_fits(self):
        impl = _impl(topk=8)
        sel = self._select(impl, torch.randn(1, 2, 5), [4])
        assert sel.tolist() == [[True] * 5]

    def test_never_selects_a_future_block(self):
        impl = _impl(topk=4, init=0, local=1)
        sel = self._select(impl, torch.randn(3, 2, 8), [2, 5, 7])
        for row, qb in enumerate([2, 5, 7]):
            assert not sel[row, qb + 1 :].any()

    def test_keeps_exactly_topk_of_a_long_row(self):
        impl = _impl(topk=4, init=1, local=2)
        sel = self._select(impl, torch.randn(1, 2, 20), [19])
        assert int(sel.sum()) == 4

    def test_forced_blocks_beat_better_scores(self):
        impl = _impl(topk=4, init=1, local=2)
        scores = torch.zeros(1, 2, 20)
        scores[..., 5:9] = 100.0  # the best blocks, none of them forced
        sel = self._select(impl, scores, [19])[0]
        assert sel[0] and sel[18] and sel[19]  # init, and the 2 local blocks
        assert int(sel.sum()) == 4 and sel[5:9].sum() == 1  # one slot left for the best

    def test_local_wins_where_init_and_local_coincide(self):
        impl = _impl(topk=2, init=1, local=2)
        sel = self._select(impl, torch.randn(1, 2, 6), [1])[0]
        assert sel[:2].all() and not sel[2:].any()

    def test_scores_pick_the_remaining_slots(self):
        impl = _impl(topk=3, init=1, local=1)
        scores = torch.zeros(1, 2, 10)
        scores[..., 4] = 5.0
        scores[..., 7] = 9.0
        sel = self._select(impl, scores, [9])[0]
        assert sel.nonzero().flatten().tolist() == [0, 7, 9]


class TestBlockScores:
    """`_block_max`: the per-head score of each block of keys."""

    def _inputs(self, nb=3, page=4, ni=2, di=8, rows=2, seed=0):
        g = torch.Generator().manual_seed(seed)
        iq = torch.randn(rows, ni, di, generator=g)
        ik = torch.randn(nb * page, di, generator=g)
        # row r sees the first 5 + 4 r keys
        visible = (
            torch.arange(nb * page)[None, :] < (5 + 4 * torch.arange(rows))[:, None]
        )
        return iq, ik, visible

    def test_max_is_the_best_visible_token(self):
        impl = _impl()
        iq, ik, vis = self._inputs()
        got = impl._block_max(iq, ik, vis, ProVariants(block_score="max"))
        dots = torch.einsum("rnd,ld->rnl", iq, ik).masked_fill(
            ~vis[:, None, :], float("-inf")
        )
        assert torch.allclose(got, dots.view(2, 2, 3, 4).amax(-1))

    def test_lse_is_the_log_attention_mass_of_the_block(self):
        impl = _impl()
        iq, ik, vis = self._inputs()
        got = impl._block_max(iq, ik, vis, ProVariants(block_score="lse"))
        logits = torch.einsum("rnd,ld->rnl", iq, ik) * impl.index_head_dim**-0.5
        logits = logits.masked_fill(~vis[:, None, :], float("-inf"))
        mass = logits.exp().view(2, 2, 3, 4).sum(-1)
        valid = mass > 0
        assert torch.allclose(got.exp()[valid], mass[valid], rtol=1e-5)
        assert (got[~valid] < -1e20).all()  # a block with nothing visible has no mass

    def test_lse_is_bounded_by_the_max_and_log_of_the_block_size(self):
        impl = _impl()
        iq, ik, _ = self._inputs()
        vis = torch.ones(2, 12, dtype=torch.bool)
        lse = impl._block_max(iq, ik, vis, ProVariants(block_score="lse"))
        mx = (
            impl._block_max(iq, ik, vis, ProVariants(block_score="max"))
            * impl.index_head_dim**-0.5
        )
        assert (lse >= mx - 1e-6).all() and (
            lse <= mx + torch.log(torch.tensor(4.0)) + 1e-6
        ).all()

    def test_index_scale_is_a_temperature_of_the_token_scores(self):
        impl = _impl()
        iq, ik, vis = self._inputs()
        hot = impl._block_max(
            iq, ik, vis, ProVariants(block_score="lse", index_scale=3.0)
        )
        ref = impl._block_max(iq * 3.0, ik, vis, ProVariants(block_score="lse"))
        valid = vis.view(2, 3, 4).any(-1)[:, None, :].expand_as(hot)
        assert torch.allclose(hot[valid], ref[valid], rtol=1e-5)

    def test_selection_from_lse_scores_keeps_the_heaviest_blocks(self):
        impl = _impl(topk=3, init=1, local=1, page=4, idx_dim=8)
        iq = torch.zeros(1, 2, 8)
        iq[..., 0] = 1.0
        ik = torch.zeros(40, 8)  # 10 blocks
        ik[4 * 6, 0] = 40.0  # one strong token in block 6
        ik[4 * 3 : 4 * 3 + 4, 0] = 6.0  # four medium tokens in block 3: the runner-up
        v = ProVariants(block_score="lse", head_reduce="lse")
        vis = torch.ones(1, 40, dtype=torch.bool)
        sel = impl._select_blocks(
            impl._block_max(iq, ik, vis, v), torch.tensor([9]), v
        )[0]
        assert sel.nonzero().flatten().tolist() == [0, 6, 9]  # init, best block, local


class TestRope:
    def test_non_rope_dims_are_untouched(self):
        impl = _impl()
        x = torch.randn(5, 3, 16)
        pos = torch.arange(5).float()
        ang = pos[:, None] * torch.tensor([1.0, 0.1])[None]
        out = impl._rope(x, ang.cos(), ang.sin())
        assert torch.equal(out[..., 4:], x[..., 4:])
        assert not torch.allclose(out[1:, :, :4], x[1:, :, :4])

    def test_position_zero_is_identity(self):
        impl = _impl()
        x = torch.randn(2, 3, 16)
        ones, zeros = torch.ones(2, 2), torch.zeros(2, 2)
        assert torch.allclose(impl._rope(x, ones, zeros), x)

    def test_neox_rotates_halves_and_gptj_rotates_pairs(self):
        impl = _impl(rot=4)
        x = torch.zeros(1, 1, 16)
        x[..., :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        c = torch.zeros(1, 2)  # cos 0 / sin 1: a quarter turn
        s = torch.ones(1, 2)
        neox = impl._rope(x, c, s, "neox")[0, 0, :4].tolist()
        gptj = impl._rope(x, c, s, "gptj")[0, 0, :4].tolist()
        assert neox == [-3.0, -4.0, 1.0, 2.0]  # (x1, x2) = (dims 0-1, dims 2-3)
        assert gptj == [-2.0, 1.0, -4.0, 3.0]  # pairs (0,1) and (2,3)


class TestAttention:
    def _setup(self, length, **kw):
        torch.manual_seed(0)
        impl = _impl(**kw)
        k = torch.randn(length, impl.head_dim)
        ik = torch.randn(length, impl.index_head_dim)
        q = torch.randn(length, impl.num_heads, impl.head_dim)
        iq = torch.randn(length, impl.num_idx_heads, impl.index_head_dim)
        return impl, q, iq, k, ik

    def test_v_is_k_without_its_rope_dims(self):
        impl = _impl()
        k = torch.randn(7, 16)
        assert torch.equal(impl._v_of(k, ProVariants()), k[:, 4:])

    def test_causal(self):
        # the first 6 outputs of a 10-token sequence are those of its 6-token prefix
        impl, q, iq, k, ik = self._setup(10, topk=8)
        pos = torch.arange(10)
        full = impl._attend_prefill_seq(q, iq, pos, k, ik, ProVariants())
        part = impl._attend_prefill_seq(
            q[:6], iq[:6], pos[:6], k[:6], ik[:6], ProVariants()
        )
        assert torch.allclose(full[:6], part, atol=1e-5)

    def test_selection_that_keeps_everything_equals_dense(self):
        impl, q, iq, k, ik = self._setup(12, topk=64)
        pos = torch.arange(12)
        sparse = impl._attend_prefill_seq(q, iq, pos, k, ik, ProVariants())
        dense = impl._attend_prefill_seq(
            q, iq, pos, k, ik, ProVariants(dense_only=True)
        )
        assert torch.allclose(sparse, dense, atol=1e-6)

    def test_dropping_blocks_changes_the_answer(self):
        impl, q, iq, k, ik = self._setup(40, topk=3, init=1, local=1)
        pos = torch.arange(40)
        sparse = impl._attend_prefill_seq(q, iq, pos, k, ik, ProVariants())
        dense = impl._attend_prefill_seq(
            q, iq, pos, k, ik, ProVariants(dense_only=True)
        )
        assert not torch.allclose(sparse[-1], dense[-1])
        # an early position still sees everything: nothing was dropped there
        assert torch.allclose(sparse[:8], dense[:8], atol=1e-6)

    @pytest.mark.parametrize(
        "variant",
        [
            {},
            {"block_score": "max", "head_reduce": "sum"},
            {"block_score": "pool4", "head_reduce": "sum", "pool_scale": 1.0},
            {"block_score": "lse", "head_reduce": "sum"},
            {"block_score": "lse", "head_reduce": "lse", "index_scale": 2.0},
        ],
    )
    def test_decode_matches_prefill(self, variant):
        # Prefill of L tokens and the decode step of token L-1 over the same caches.
        impl, q, iq, k, ik = self._setup(37, topk=4, init=1, local=2)
        v = ProVariants(**variant)
        length, page = 37, impl.page
        nb = (length + page - 1) // page
        x = 8  # shuffled-K inner width
        kc4 = torch.zeros(nb + 3, impl.head_dim // x, page, x)
        ic = torch.zeros(nb + 3, page, impl.index_head_dim)
        blocks = torch.arange(nb) + 2  # a non-trivial block table
        pos = torch.arange(length)
        blk, off = blocks[pos // page], pos % page
        kc4[blk, :, off, :] = k.reshape(length, impl.head_dim // x, x)
        ic[blk, off] = ik
        md = SimpleNamespace(block_tables=blocks.unsqueeze(0))
        pre = impl._attend_prefill_seq(q, iq, pos, k, ik, v)
        tok, dec = impl._attend_decode(
            q, iq, md, [(length - 1, length, 0)], kc4, None, ic, None, v
        )
        assert tok.tolist() == [length - 1]
        assert torch.allclose(dec[0], pre[-1], atol=1e-5)


class TestCacheIO:
    """`_write_kv` / `_read_k`: K into the shuffled K cache, V (K's tail) into the
    transposed V cache, an fp8 cache quantized per token with its scales beside it."""

    NB = 5

    def _layer(self, impl, x, dtype):
        fp8 = dtype.itemsize == 1
        scales = lambda: torch.zeros(self.NB, 1, impl.page) if fp8 else None
        return SimpleNamespace(
            k_cache=torch.zeros(
                self.NB, 1, impl.head_dim // x, impl.page, x, dtype=dtype
            ),
            v_cache=torch.zeros(
                self.NB, 1, impl.page // x, impl.head_dim, x, dtype=dtype
            ),
            k_scale=scales(),
            v_scale=scales(),
        )

    def _write(self, dtype, x, v=None):
        v = ProVariants() if v is None else v
        impl = _impl(page=8)
        layer = self._layer(impl, x, dtype)
        g = torch.Generator().manual_seed(1)
        k = torch.randn(11, impl.head_dim, generator=g) * 3
        k[3] = 0.0  # an all-zero row stays zero (no 0 / 0)
        slots = torch.randperm(self.NB * impl.page, generator=g)[: k.shape[0]]
        blk, off = slots // impl.page, slots % impl.page
        impl._write_kv(layer, blk, off, k, v)
        vk = layer.v_cache[:, 0][blk, off // x, : impl.value_head_dim, off % x].float()
        return impl, layer, k, blk, off, vk

    def test_bf16_cache_round_trips_exactly(self):
        impl, layer, k, blk, off, vk = self._write(torch.float32, 4)
        assert torch.equal(impl._read_k(layer.k_cache[:, 0], None, blk, off), k)
        assert torch.equal(vk, k[:, -impl.value_head_dim :])

    def test_fp8_cache_is_within_rounding(self):
        impl, layer, k, blk, off, vk = self._write(torch.float8_e4m3fn, 8)
        dv = impl.value_head_dim
        ks, vs = layer.k_scale[blk, 0, off], layer.v_scale[blk, 0, off]
        # each row's largest value maps to fp8's largest (448)
        assert torch.allclose(ks, k.abs().amax(-1).clamp(min=1e-6) / 448)
        assert torch.allclose(vs, k[:, -dv:].abs().amax(-1).clamp(min=1e-6) / 448)
        got = impl._read_k(layer.k_cache[:, 0], layer.k_scale, blk, off)
        vk = vk * vs[:, None]
        # e4m3 rounds to 3 mantissa bits: within 1/16 of the value, or a subnormal step
        assert ((got - k).abs() <= k.abs() / 16 + ks[:, None] * 2**-9).all()
        assert (
            (vk - k[:, -dv:]).abs() <= k[:, -dv:].abs() / 16 + vs[:, None] * 2**-9
        ).all()
        assert not got[3].any() and not vk[3].any()

    def test_fp8_index_rows_are_a_clamped_cast(self):
        ik = torch.tensor([[0.3, -1000.0, 1000.0, 2.0e-3]])
        got = pro_attn._index_rows(ik, torch.float8_e4m3fn).float()
        assert got[0, 1] == -448.0 and got[0, 2] == 448.0
        assert torch.equal(got, ik.clamp(-448, 448).to(torch.float8_e4m3fn).float())
        assert torch.equal(pro_attn._index_rows(ik, torch.bfloat16), ik.bfloat16())

    def test_pre_v_experiment_leaves_the_v_cache_alone(self):
        _, layer, *_ = self._write(torch.float32, 4, ProVariants(v_src="pre"))
        assert not layer.v_cache.any()
