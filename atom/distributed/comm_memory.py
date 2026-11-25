# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device memory of the communicators aiter builds in ``init_dist_env``.

aiter gives every group of more than one rank its own device communicator: an
RCCL communicator and a custom all-reduce whose IPC buffers are sized by its
``max_size`` (a 1 GiB input pool plus, on gfx9, a 2 GiB scratch region). On
MI355X at TP8 each such communicator costs about 5 GiB of HBM. Two settings
here cut that before the KV pool is sized:

* ``share_identical_groups``: a group over exactly the TP ranks (the EP group
  without expert parallelism, the DCP group at DCP == TP) shares TP's
  communicator instead of building another (aiter's
  ``AITER_REUSE_IDENTICAL_COMM_GROUPS``).
* ``custom_ar_pool``: ``ATOM_CUSTOM_AR_POOL_BYTES`` sets the custom all-reduce
  pool. The pool only has to hold the largest input routed to it: inputs above
  ``AITER_CUSTOM_AR_MAX_SIZE`` (64 MiB by default) go to RCCL whatever the pool,
  and a prefill all-reduce takes the custom kernels up to half the pool, so a
  pool of twice that cutoff routes every all-reduce as the default pool does.

torch's own NCCL process groups build one more RCCL communicator each, on
their first collective (about 2.4 GiB on MI355X at TP8). ``group_broadcast``
and ``host_barrier`` keep the runner's per-step broadcast and its post-KV
barrier off them, so neither communicator is built. Where a collective
needs torch's group (DCP's decode combine is an all-to-all on it),
``build_torch_group_communicator`` builds that communicator during warmup, so
the KV budget charges it instead of finding it after the pool is allocated.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
from collections.abc import Iterator

logger = logging.getLogger("atom")

REUSE_ENV = "AITER_REUSE_IDENTICAL_COMM_GROUPS"
AR_MAX_ENV = "AITER_CUSTOM_AR_MAX_SIZE"
AG_MAX_ENV = "AITER_CUSTOM_AG_MAX_SIZE"
# aiter's CustomAllreduce defaults, which the cutoffs below are derived from.
AITER_POOL_BYTES = 1 << 30
AITER_AR_MAX_BYTES = 64 << 20


def share_identical_groups(expert_parallel: bool) -> bool:
    """Let groups over identical ranks share one communicator, unless the env
    already says otherwise. Left to aiter's default under expert parallelism,
    where the EP group's all-to-all runs beside TP's collectives. Returns
    whether sharing is on."""
    if not expert_parallel:
        os.environ.setdefault(REUSE_ENV, "1")
    return os.environ.get(REUSE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def _ar_max_bytes() -> int:
    raw = os.environ.get(AR_MAX_ENV, "").strip()
    return int(raw) if raw else AITER_AR_MAX_BYTES


def _custom_allreduce_cls():
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    return CustomAllreduce


def _group_world_size(group) -> int:
    import torch

    return torch.distributed.get_world_size(group)


@contextlib.contextmanager
def _env_default(name: str, value: str) -> Iterator[None]:
    """``name`` set to ``value`` inside the block when it is unset."""
    if name in os.environ:
        yield
        return
    os.environ[name] = value
    try:
        yield
    finally:
        os.environ.pop(name, None)


@contextlib.contextmanager
def custom_ar_pool(pool_bytes: int | None) -> Iterator[None]:
    """Build every custom all-reduce inside the block with a ``pool_bytes``
    input pool; None keeps aiter's. Raises if the pool would move a prefill
    all-reduce at or below ``AITER_CUSTOM_AR_MAX_SIZE`` to RCCL.

    Each communicator's all-gather cutoff, which aiter derives from the pool
    (pool / (2 * the group's size)), stays at what aiter's default pool gives
    that group, unless ``AITER_CUSTOM_AG_MAX_SIZE`` is set; aiter caps it at
    the pool."""
    if not pool_bytes:
        yield
        return
    ar_max = _ar_max_bytes()
    if pool_bytes < 2 * ar_max:
        raise ValueError(
            f"ATOM_CUSTOM_AR_POOL_BYTES={pool_bytes} is below twice "
            f"{AR_MAX_ENV} ({ar_max}): prefill all-reduces between "
            f"{pool_bytes // 2} and {ar_max} bytes would move to RCCL"
        )
    cls = _custom_allreduce_cls()
    init = cls.__init__

    @functools.wraps(init)
    def sized_init(self, *args, **kwargs):
        if len(args) < 3:
            kwargs.setdefault("max_size", pool_bytes)
        group = kwargs["group"] if "group" in kwargs else args[0]
        ag_max = AITER_POOL_BYTES // (2 * _group_world_size(group))
        with _env_default(AG_MAX_ENV, str(ag_max)):
            init(self, *args, **kwargs)

    cls.__init__ = sized_init
    logger.info("Custom all-reduce input pool: %d bytes", pool_bytes)
    try:
        yield
    finally:
        cls.__init__ = init


def group_broadcast(group, tensor, src: int = 0):
    """``group.broadcast`` on the group's own RCCL communicator, on the current
    stream, instead of torch's process group, which would build a second
    communicator for the same ranks. ``src`` is the rank within the group."""
    if group.world_size == 1:
        return tensor
    communicator = getattr(group, "device_communicator", None)
    pynccl = getattr(communicator, "pynccl_comm", None)
    if pynccl is None or pynccl.disabled:
        return group.broadcast(tensor, src=src)
    pynccl.broadcast(tensor, src=src)
    return tensor


def host_barrier(group) -> None:
    """Wait for this rank's queued device work, then for every rank of
    ``group`` (a gloo group): the NCCL barrier's guarantee, without the
    communicator it builds."""
    import torch

    torch.cuda.synchronize()
    torch.distributed.barrier(group=group)


def build_dcp_communicator() -> None:
    """Under DCP, build the torch communicator its decode combine all-to-alls
    on: that of ``dcp_utils.get_dcp_group()``, which at DCP = TP is the TP
    group whether or not the DCP group shares TP's communicators."""
    from atom.distributed import dcp_utils

    if dcp_utils.dcp_is_enabled():
        build_torch_group_communicator(dcp_utils.get_dcp_group())


def build_torch_group_communicator(group) -> None:
    """Build the RCCL communicator of ``group``'s torch NCCL process group now,
    with one all-to-all of ``world_size`` ints; nothing on one rank."""
    if group.world_size == 1:
        return
    import torch

    send = torch.zeros(group.world_size, dtype=torch.int32, device=group.device)
    torch.distributed.all_to_all_single(
        torch.empty_like(send), send, group=group.device_group
    )
