# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Push collectives for the DCP decode of TP8 + DCP8 steps of hundreds of rows.

Every rank stores its rows straight into each peer's uncached IPC buffer in 16-byte lanes,
then raises one flag a (peer, program); each program of the receiver polls its flags and
finishes in the same launch. Stores into uncached memory cannot stay in a cache: once
``s_waitcnt vmcnt(0)`` retires them the peers see them, so a drain and a barrier before
the flag are the whole release, and the receiver's loads bypass L1 and L2 (``.cv``), as
aiter's FlyDSL transport does for its uncached inboxes.

- ``PushGather``: the q all-gather (last dim, rank-major) and the keys all-gather (first
  dim). A plain copy: the bits of the custom all-gather.
- ``PushMerge``: the LSE merge of every head's partial output into the owner's heads, the
  weighted sum of ``cp_lse_a2a``'s combine. ``order`` / ``lse_order`` pick the order its
  eight terms are added in (``_add8``); the combine rounds each product before the adds.

One workspace a (group, op): its buffer, flags ``[ranks, GRID_MAX]`` of launch numbers and
a launch counter on the device. Every rank must launch the same ops in the same order with
the same shapes, as DCP decode does (and as the mailbox requires). Buffers are single: in a
decode step every rank's merge separates two gathers of one workspace and every rank's q
all-gather two merges, so no peer writes a buffer again before this rank is done with it.
With only the merge pushed (``dcp_push_merge``), the custom pull all-gathers between two
merges wait for every rank's data first, which keeps the same order.
``parities=2`` alternates two buffers for callers without that order (benchmarks, tests).
"""

import logging

import torch
import triton
import triton.language as tl

logger = logging.getLogger("atom")

GRID_MAX = tl.constexpr(1024)  # programs a launch at most (the flag rows' width)
POLL_MAX = tl.constexpr(1 << 24)  # flag polls of a program before it counts a time-out
_STATE = 4  # launch counter, programs done, time-outs, spare


@triton.jit
def _drain(x):
    """Waits until the wave's vector memory operations have completed (x passes through)."""
    return tl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)",
        "=v,0,~{memory}",
        [x],
        dtype=x.dtype,
        is_pure=False,
        pack=1,
    )


@triton.jit
def _rounded(x):
    """fp32 x as an opaque copy: no FMA takes in the product it was rounded from."""
    return tl.inline_asm_elementwise(
        "v_mov_b32 $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit
def _bases(peers_ptr, W: tl.constexpr, DT: tl.constexpr):
    """Every rank's buffer as a ``[W]`` tensor of pointers to DT (16-byte aligned)."""
    return tl.multiple_of(
        tl.load(peers_ptr + tl.arange(0, W)).to(tl.pointer_type(DT)), [16]
    )


@triton.jit
def _base(peers_ptr, d, DT: tl.constexpr):
    """Rank d's buffer as a pointer to DT (16-byte aligned)."""
    return tl.multiple_of(tl.load(peers_ptr + d).to(tl.pointer_type(DT)), 16)


@triton.jit
def _raise_flags(peers_ptr, flag_off, rank, pid, tag, W: tl.constexpr):
    """Every wave's stores retired, then this program's flag in every rank's row ``rank``."""
    _drain(tl.full([1], 0, tl.int32))
    tl.debug_barrier()
    flags = _bases(peers_ptr, W, tl.int32)
    tl.store(
        flags + flag_off + rank * GRID_MAX + pid,
        tl.full([W], 0, tl.int32) + tag,
        cache_modifier=".wt",
    )


@triton.jit
def _await_flags(peers_ptr, flag_off, state_ptr, rank, pid, tag, W: tl.constexpr):
    """Polls this program's flag from every rank until it reads ``tag`` (or a later
    launch's number); a time-out is counted in state[2] and the program goes on."""
    flags = (
        _base(peers_ptr, rank, tl.int32) + flag_off + tl.arange(0, W) * GRID_MAX + pid
    )
    f = tl.load(flags, cache_modifier=".cv", volatile=True)
    spins = tl.full([], 0, tl.int32)
    while (tl.min(f - tag, axis=0) < 0) & (spins < POLL_MAX):
        f = tl.load(flags, cache_modifier=".cv", volatile=True)
        spins += 1
    if spins >= POLL_MAX:
        tl.atomic_add(state_ptr + 2, 1, sem="relaxed")
    tl.debug_barrier()


@triton.jit
def _finish(state_ptr, tag, nprog):
    """The last program of the launch moves the launch counter on (the next launch reads
    it: the kernel boundary orders it, no fence)."""
    done = tl.atomic_add(state_ptr + 1, 1, sem="relaxed", scope="gpu")
    if done == nprog - 1:
        tl.store(state_ptr + 1, 0)
        tl.store(state_ptr, tag)


@triton.jit
def _push_gather_kernel(
    x_ptr,  # [n, CH] int32 words, this rank's rows
    sx,  # row stride of x, words
    peers_ptr,  # int64 [W] every rank's buffer
    out_off,  # words into every buffer: the output
    flag_off,  # int32 elements into every buffer: flags [W, GRID_MAX]
    state_ptr,  # int32 [4] this rank's launch counter, programs done, time-outs
    n,
    rank,
    row_stride,  # out[row * row_stride + src * src_stride + c], words
    src_stride,
    nprog,
    W: tl.constexpr,
    CH: tl.constexpr,
    CHP: tl.constexpr,  # CH rounded up to a power of two
    RB: tl.constexpr,  # rows a program
    RBP: tl.constexpr,
    PHASES: tl.constexpr,  # 1 push, 2 receive, 3 both (1 and 2: one GPU standing in)
):
    pid = tl.program_id(0)
    tag = tl.load(state_ptr) + 1
    if PHASES & 1:
        # every base loaded before the first store: on gfx9 a wait for a load waits for
        # the stores issued before it too
        dst = ()
        for d in tl.static_range(W):
            dst = dst + (_base(peers_ptr, d, tl.int32),)
        r = tl.arange(0, RBP)
        c = tl.arange(0, CHP)
        row = pid * RB + r
        m = ((r < RB) & (row < n))[:, None] & (c < CH)[None, :]
        v = tl.load(x_ptr + row[:, None] * sx + c[None, :], mask=m)
        o = out_off + row[:, None] * row_stride + rank * src_stride + c[None, :]
        for d in tl.static_range(W):
            tl.store(dst[d] + o, v, mask=m, cache_modifier=".cs")
        _raise_flags(peers_ptr, flag_off, rank, pid, tag, W)
    if PHASES & 2:
        _await_flags(peers_ptr, flag_off, state_ptr, rank, pid, tag, W)
        _finish(state_ptr, tag, nprog)


@triton.jit
def _add8(t, ORDER: tl.constexpr):
    """t[0] + ... + t[7]: ORDER 0 in turn; 1 to 6 a tree of three rounds that each add
    the terms an offset apart, the offsets in the order 124, 142, 214, 241, 412, 421."""
    if ORDER == 0:
        s = t[0] + t[1]
        s = s + t[2]
        s = s + t[3]
        s = s + t[4]
        s = s + t[5]
        s = s + t[6]
        s = s + t[7]
    elif ORDER == 1:
        s = ((t[0] + t[1]) + (t[2] + t[3])) + ((t[4] + t[5]) + (t[6] + t[7]))
    elif ORDER == 2:
        s = ((t[0] + t[1]) + (t[4] + t[5])) + ((t[2] + t[3]) + (t[6] + t[7]))
    elif ORDER == 3:
        s = ((t[0] + t[2]) + (t[1] + t[3])) + ((t[4] + t[6]) + (t[5] + t[7]))
    elif ORDER == 4:
        s = ((t[0] + t[2]) + (t[4] + t[6])) + ((t[1] + t[3]) + (t[5] + t[7]))
    elif ORDER == 5:
        s = ((t[0] + t[4]) + (t[1] + t[5])) + ((t[2] + t[6]) + (t[3] + t[7]))
    else:
        s = ((t[0] + t[4]) + (t[2] + t[6])) + ((t[1] + t[5]) + (t[3] + t[7]))
    return s


@triton.jit
def _push_merge_kernel(
    of_ptr,  # [n, W * HL * D] bf16 partial outputs of every head (as int32 words)
    s_of,  # row stride, words
    lse_ptr,  # [n, W * HL * LSE_REP] fp32
    s_lse,
    out_ptr,  # [n, HL * D] bf16 this rank's heads (fp32: the sums unrounded)
    s_out,
    peers_ptr,
    lse_off,  # fp32 elements into every buffer: lse [PARITIES, W, rows_max, HL]
    flag_off,  # (data [PARITIES, W, rows_max, HL * D] bf16 at 0)
    state_ptr,
    n,
    rank,
    rows_max,
    nprog,
    PARITIES: tl.constexpr,
    W: tl.constexpr,
    HL: tl.constexpr,
    D: tl.constexpr,
    LSE_REP: tl.constexpr,
    RB: tl.constexpr,
    ORDER: tl.constexpr,
    LSE_ORDER: tl.constexpr,
    PHASES: tl.constexpr,
):
    tl.static_assert(W == 8)
    WORDS: tl.constexpr = HL * D // 2
    CW: tl.constexpr = 512  # words a chunk a destination
    tl.static_assert(WORDS % CW == 0)
    pid = tl.program_id(0)
    epoch = tl.load(state_ptr)
    tag = epoch + 1
    par = epoch % PARITIES
    d = tl.arange(0, W)
    h = tl.arange(0, HL)
    if PHASES & 1:
        c = tl.arange(0, CW)
        dst = _bases(peers_ptr, W, tl.int32)
        ldst = _bases(peers_ptr, W, tl.float32)
        for r in tl.static_range(RB):
            row = pid * RB + r
            if row < n:
                slot = (par * W + rank) * rows_max + row
                src = of_ptr + row * s_of + d[:, None] * WORDS + c[None, :]
                vs = ()
                for c0 in tl.static_range(0, WORDS, CW):
                    vs = vs + (tl.load(src + c0),)
                lv = tl.load(
                    lse_ptr + row * s_lse + (d[:, None] * HL + h[None, :]) * LSE_REP
                )
                for i in tl.static_range(WORDS // CW):
                    o = slot * WORDS + i * CW + c[None, :]
                    tl.store(dst[:, None] + o, vs[i], cache_modifier=".cs")
                tl.store(
                    ldst[:, None] + lse_off + slot * HL + h[None, :],
                    lv,
                    cache_modifier=".cs",
                )
        _raise_flags(peers_ptr, flag_off, rank, pid, tag, W)
    if PHASES & 2:
        _await_flags(peers_ptr, flag_off, state_ptr, rank, pid, tag, W)
        own_d = _base(peers_ptr, rank, tl.bfloat16)
        own_l = _base(peers_ptr, rank, tl.float32)
        j = tl.arange(0, 64)
        col = h[:, None] * D + j[None, :]
        for r in tl.static_range(RB):
            row = pid * RB + r
            if row < n:
                lses = ()
                vals = ()
                for s in tl.static_range(W):
                    slot = (par * W + s) * rows_max + row
                    x = tl.load(own_l + lse_off + slot * HL + h, cache_modifier=".cv")
                    lses = lses + (x,)
                    for part in tl.static_range(D // 64):
                        v = tl.load(
                            own_d + slot * (HL * D) + part * 64 + col,
                            cache_modifier=".cv",
                        )
                        vals = vals + (v,)
                # a rank with no key of the row reports -inf (and NaN out): no weight
                lsf = ()
                for s in tl.static_range(W):
                    x = lses[s]
                    bad = (x != x) | (x == float("inf"))  # noqa: PLR0124 - NaN test
                    lsf = lsf + (tl.where(bad, float("-inf"), x),)
                mx = lsf[0]
                for s in tl.static_range(1, W):
                    mx = tl.maximum(mx, lsf[s])
                mx = tl.where(mx == float("-inf"), 0.0, mx)
                es = ()
                for s in tl.static_range(W):
                    es = es + (tl.exp(lsf[s] - mx),)
                glse = tl.log(_add8(es, LSE_ORDER)) + mx
                fs = ()
                for s in tl.static_range(W):
                    f = tl.exp(lsf[s] - glse)
                    f = tl.where(f != f, 0.0, f)  # noqa: PLR0124 - NaN test
                    fs = fs + (f[:, None],)
                for part in tl.static_range(D // 64):
                    terms = ()
                    for s in tl.static_range(W):
                        v = vals[s * (D // 64) + part].to(tl.float32)
                        terms = terms + (
                            _rounded(tl.where(fs[s] == 0.0, 0.0, v) * fs[s]),
                        )
                    acc = _add8(terms, ORDER).to(out_ptr.dtype.element_ty)
                    tl.store(out_ptr + row * s_out + part * 64 + col, acc)
        _finish(state_ptr, tag, nprog)


def _rows_per_program(n: int, grid: int) -> int:
    return max(1, -(-n // grid))


def _layout(world: int, words: int, flag_sets: int = 1) -> tuple[int, int]:
    """(flag offset in int32 elements, buffer bytes) of a workspace of ``words`` words and
    ``flag_sets`` sets of flags ``[world, GRID_MAX]``."""
    flag_off = (int(words) + 63) // 64 * 64
    return flag_off, 4 * (flag_off + flag_sets * world * GRID_MAX.value)


class PushComm:
    """One push collective's workspace over a group's ranks (a GroupCoordinator, or an
    object with ``cpu_group``, ``rank_in_group`` and ``world_size``). Collective to build
    (it maps every rank's buffer); every rank then launches its ops alike.

    ``words``: the payload, 4-byte words (the widest launch's); ``flag_sets`` sets of
    flags follow it. ``peers``: ``(views, addresses)`` of ``test_peers``, for several ranks
    in one process (``rank`` this one); else the buffer is uncached peer memory (aiter's
    ``PeerBuffer``)."""

    def __init__(
        self, group, words: int, device=None, peers=None, rank=None, flag_sets=1
    ):
        if device is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self.words = int(words)
        self._buf, self._group = None, None
        if peers is None:
            from aiter.ops.flydsl.dcp_mailbox.peer import PeerBuffer

            self.world, self.rank = group.world_size, group.rank_in_group
            self.flag_off, nbytes = _layout(self.world, self.words, flag_sets)
            self._group = group.cpu_group
            self._buf = PeerBuffer(nbytes, self._group, self.rank, self.world, device)
            self.local, self.addresses = self._buf.bytes, self._buf.addresses
        else:
            self.world, self.rank = len(peers[0]), rank
            self.flag_off, nbytes = _layout(self.world, self.words, flag_sets)
            self.local, self.addresses = peers[0][rank], peers[1]
            assert self.local.numel() >= nbytes, "test_peers sized for fewer words"
        self.state = torch.zeros(_STATE, dtype=torch.int32, device=device)
        self.launches = 0  # ops launched (host side), for check_dcp_push_launches

    @staticmethod
    def test_peers(world: int, words: int, device=None, flag_sets: int = 1):
        """Plain buffers standing in for ``world`` ranks' on one GPU: ``(views, addresses)``."""
        _, nbytes = _layout(world, words, flag_sets)
        views = [
            torch.zeros(nbytes, dtype=torch.uint8, device=device) for _ in range(world)
        ]
        addresses = torch.tensor(
            [v.data_ptr() for v in views], dtype=torch.int64, device=device
        )
        return views, addresses

    def timeouts(self) -> int:
        """Flag polls that gave up, over the launches so far (0 unless a peer went missing)."""
        return int(self.state[2].item())

    def close(self):
        """Collective: release the peer memory."""
        if self._buf is None:
            return
        import torch.distributed as dist

        torch.cuda.synchronize(self.device)
        dist.barrier(group=self._group)
        self._buf.close_peers()
        dist.barrier(group=self._group)
        self._buf.close()
        self._buf = None

    def _words(self, off: int, numel: int) -> torch.Tensor:
        return self.local.view(torch.int32)[off : off + numel]


class PushGather(PushComm):
    """All-gather of ``[rows, width]`` rows (16-byte aligned, up to ``row_bytes`` a row, a
    multiple of 16): ``dim`` -1 rank-major on the last dim, 0 on the first. The output is a
    view of this rank's buffer, valid until the workspace's next launch."""

    def __init__(
        self, group, max_rows: int, row_bytes: int, *, parities: int = 1, **kw
    ):
        assert row_bytes % 16 == 0
        world = len(kw["peers"][0]) if kw.get("peers") is not None else group.world_size
        self.max_rows, self.row_words, self.parities = (
            max_rows,
            row_bytes // 4,
            parities,
        )
        self.slab = max_rows * world * self.row_words
        super().__init__(group, self.slab * parities, **kw)
        self.calls = 0

    def fits(self, x: torch.Tensor) -> bool:
        row = x.shape[-1] * x.element_size() if x.dim() == 2 else 0
        return (
            x.dim() == 2
            and 0 < x.shape[0] <= self.max_rows
            and 0 < row <= 4 * self.row_words
            and row % 16 == 0
            and x.stride(1) == 1
            and (x.stride(0) * x.element_size()) % 16 == 0
            and x.data_ptr() % 16 == 0
        )

    def __call__(
        self, x: torch.Tensor, dim: int = -1, grid: int = 256, phases: int = 3
    ):
        xw = x.view(torch.int32)
        n, w, ch = x.shape[0], self.world, xw.shape[1]
        par = self.calls % self.parities
        if phases & 2:
            self.calls += 1
        row_stride, src_stride = (ch, n * ch) if dim == 0 else (w * ch, ch)
        rb = _rows_per_program(n, min(grid, GRID_MAX.value))
        _push_gather_kernel[(-(-n // rb),)](
            xw,
            xw.stride(0),
            self.addresses,
            par * self.slab,
            self.flag_off,
            self.state,
            n,
            self.rank,
            row_stride,
            src_stride,
            -(-n // rb),
            W=w,
            CH=ch,
            CHP=triton.next_power_of_2(ch),
            RB=rb,
            RBP=triton.next_power_of_2(rb),
            PHASES=phases,
            num_warps=4,
        )
        if phases & 2:
            self.launches += 1
        out = self._words(par * self.slab, n * w * ch).view(x.dtype)
        if dim == 0:
            return out.view(w * n, x.shape[1])
        return out.view(n, w * x.shape[1])


class PushMerge(PushComm):
    """The LSE merge of ``of`` ``[n, W * HL * D]`` bf16 (``lse`` ``[n, W * HL * rep]``
    fp32, each head's LSE ``rep`` times) into this rank's ``out`` ``[n, HL * D]``."""

    def __init__(
        self, group, max_rows: int, heads: int, dim: int, *, parities: int = 1, **kw
    ):
        assert dim % 64 == 0 and heads & (heads - 1) == 0 and heads * dim % 1024 == 0
        world = len(kw["peers"][0]) if kw.get("peers") is not None else group.world_size
        assert world == 8, "the merge adds eight ranks' terms"
        self.max_rows, self.heads, self.dim, self.parities = (
            max_rows,
            heads,
            dim,
            parities,
        )
        self.lse_off = parities * world * max_rows * heads * dim // 2
        super().__init__(
            group, self.lse_off + parities * world * max_rows * heads, **kw
        )

    def fits(self, of, lse, out) -> bool:
        n, hd = out.shape[0], self.heads * self.dim
        return (
            0 < n <= self.max_rows
            and of.dtype == out.dtype == torch.bfloat16
            and lse.dtype == torch.float32
            and tuple(of.shape) == (n, self.world * hd)
            and out.dim() == 2
            and out.shape[1] == hd
            and lse.shape[0] == n
            and lse.shape[1] % (self.world * self.heads) == 0
            and of.stride(1) == out.stride(1) == lse.stride(1) == 1
            and of.data_ptr() % 16 == 0
            and of.stride(0) % 8 == 0
        )

    def __call__(
        self,
        of,
        lse,
        out,
        grid: int = 256,
        order: int = 6,
        lse_order: int = 6,
        phases: int = 3,
    ):
        n = out.shape[0]
        rb = _rows_per_program(n, min(grid, GRID_MAX.value))
        ofw = of.view(torch.int32)
        _push_merge_kernel[(-(-n // rb),)](
            ofw,
            ofw.stride(0),
            lse,
            lse.stride(0),
            out,
            out.stride(0),
            self.addresses,
            self.lse_off,
            self.flag_off,
            self.state,
            n,
            self.rank,
            self.max_rows,
            -(-n // rb),
            PARITIES=self.parities,
            W=self.world,
            HL=self.heads,
            D=self.dim,
            LSE_REP=lse.shape[1] // (self.world * self.heads),
            RB=rb,
            ORDER=order,
            LSE_ORDER=lse_order,
            PHASES=phases,
            num_warps=4,
        )
        if phases & 2:
            self.launches += 1
        return out


class DcpPush:
    """A DCP decode's push workspaces over one group, for steps of up to ``rows`` rows:
    ``q`` (rows of ``heads * qdim`` bf16, or narrower), ``keys`` (``topk`` int64 sort
    keys, or fewer) and ``merge`` (``heads`` x ``vdim``)."""

    def __init__(self, group, rows: int, heads: int, qdim: int, vdim: int, topk: int):
        self.rows = rows
        self.q = PushGather(group, rows, heads * qdim * 2)
        self.keys = PushGather(group, rows, topk * 8)
        self.merge = PushMerge(group, rows, heads, vdim)

    def comms(self):
        return (self.q, self.keys, self.merge)


_WORKSPACES: dict = {}


def register_workspace(key, ws) -> None:
    """Have ``check_dcp_push_launches`` check ``ws`` (a ``PushComm``) too."""
    _WORKSPACES[key] = ws


def prepare_dcp_push(group, rows: int, heads: int, qdim: int, vdim: int, topk: int):
    """The group's ``DcpPush`` (one for all the layers), or None where its peer memory
    cannot be had (every rank alike: aiter agrees on the allocation and the mapping).
    Collective on first use: call it on every rank in the same order, before any graph
    is captured and before the KV cache is sized (the memory is outside torch's)."""
    key = (id(group), rows, heads, qdim, vdim, topk)
    if key not in _WORKSPACES:
        try:
            ws = DcpPush(group, rows, heads, qdim, vdim, topk)
        except RuntimeError as error:
            logger.warning("DCP push peer memory: %s; DCP keeps its collectives", error)
            ws = None
        else:
            mib = sum(c.local.numel() for c in ws.comms()) / (1 << 20)
            logger.info(
                "DCP push collectives ready: up to %d decode rows, %.1f MiB of peer memory",
                rows,
                mib,
            )
        _WORKSPACES[key] = ws
    return _WORKSPACES[key]


def check_dcp_push_launches() -> None:
    """Raise unless every rank of each workspace's group launched its ops alike (a launch
    pairs with the same launch on every peer). Collective: after graph capture."""
    import torch.distributed as dist

    for ws in _WORKSPACES.values():
        if ws is None:
            continue
        for c in ws.comms() if hasattr(ws, "comms") else (ws,):
            t = torch.tensor([c.launches, -c.launches], dtype=torch.int64)
            dist.all_reduce(t, op=dist.ReduceOp.MIN, group=c._group)
            lo, hi = int(t[0]), -int(t[1])
            if lo != hi:
                raise RuntimeError(
                    f"DCP push: ranks launched between {lo} and {hi} of a workspace's ops "
                    f"(this rank {c.launches}); replay would pair them wrongly. Set "
                    '"dcp_push_merge": 0 and report this.'
                )
