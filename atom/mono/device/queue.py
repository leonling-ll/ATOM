# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""A launch's work queues: an i32 count a queue, zeroed at the step's start,
that CTAs take the next task from by an atomic add -- a CTA its earlier work
delayed takes fewer, where a fixed placement gave every CTA the same count.

A launch's words sit at its tag's offset (``tag_offset``), computed on the
host and handed to the kernel: never ``tag * 4`` inside it. A Kimi-K3 64-row
K2 with 8 AttnRes blocks took its queue word from a register holding a peer
base's high word that way -- the LLVM IR right, the ISA not (2026-10-30) --
and claimed garbage.
"""

import flydsl.expr as fx
from flydsl.expr import gpu
from flydsl.expr.typing import T

from atom.mono.device.ops import traced


def tag_offset(layer: int, words: int = 1) -> int:
    """The byte offset of the launch ``layer``'s ``words`` queue words (its
    tag's, ``Mailbox``: ``layer`` + 1) in a queue array a tag."""
    return (layer + 1) * words * 4


def word(addr):
    """The i32 queue word at device address ``addr`` (an Int64)."""
    return fx.inttoptr(fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4), addr)


def take(queue):
    """The next count of ``queue`` (``word``), one thread's: its old value."""
    return fx.llvm.atomic_add(queue, fx.Int32(1), syncscope="agent")


def take_unmerged(queue):
    """``take`` as a ``global_atomic_inc`` (wrapping at 2^31 - 1: a count
    never gets there), which LLVM's atomic optimizer leaves alone. It merges
    a wave's uniform-address ``add`` into one lane's and reads the old value
    back at once (``v_readfirstlane``): a wait for every load in flight before
    it, where ``take``'s caller wants to issue the claim ahead of its loads and
    read it a task later."""
    from flydsl._mlir.dialects.llvm import AtomicBinOp

    return fx.llvm._emit_atomic_rmw(
        AtomicBinOp.uinc_wrap,
        queue,
        fx.Int32(0x7FFFFFFF),
        "agent",
        fx.llvm.AtomicOrdering.Monotonic,
    )


@traced
def take_shared(queue, slot, tid):
    """``take`` for the whole CTA: thread 0 takes, every thread reads it from
    LDS ``slot`` past a barrier."""
    if tid == 0:
        fx.ptr_store(fx.Int32(take(queue)), slot)
    gpu.barrier()
    return fx.ptr_load(slot)
