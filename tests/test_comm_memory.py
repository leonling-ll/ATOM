# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The communicator settings init_model_env applies before aiter's
init_dist_env, without a GPU: a stand-in for aiter's CustomAllreduce takes the
same constructor arguments and records the pool it was given."""

import os

import pytest

from atom.distributed import comm_memory
from atom.distributed.comm_memory import (
    AG_MAX_ENV,
    AITER_POOL_BYTES,
    AR_MAX_ENV,
    REUSE_ENV,
    custom_ar_pool,
    share_identical_groups,
)

MIB = 1 << 20


class _CustomAllreduce:
    """aiter's CustomAllreduce constructor signature."""

    def __init__(
        self,
        group,
        device,
        max_size=1024 * 1024 * 1024,
        enable_register_for_capturing=True,
    ):
        self.max_size = max_size
        # aiter reads its all-gather cutoff from the env while constructing.
        self.ag_max = os.environ.get(AG_MAX_ENV)


@pytest.fixture
def env(monkeypatch):
    for name in (REUSE_ENV, AR_MAX_ENV, AG_MAX_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(comm_memory, "_custom_allreduce_cls", lambda: _CustomAllreduce)
    # A stand-in group is its own world size.
    monkeypatch.setattr(comm_memory, "_group_world_size", lambda group: group or 8)
    return monkeypatch


def test_identical_groups_share_without_expert_parallel(env):
    assert share_identical_groups(expert_parallel=False)
    assert os.environ[REUSE_ENV] == "1"


def test_identical_groups_keep_an_explicit_setting(env):
    env.setenv(REUSE_ENV, "0")
    assert not share_identical_groups(expert_parallel=False)
    assert os.environ[REUSE_ENV] == "0"


def test_expert_parallel_leaves_aiters_default(env):
    assert not share_identical_groups(expert_parallel=True)
    assert REUSE_ENV not in os.environ


def test_pool_applies_inside_the_block_only(env):
    with custom_ar_pool(128 * MIB):
        assert _CustomAllreduce(group=None, device="cuda:0").max_size == 128 * MIB
        # A caller that names its pool keeps it.
        assert _CustomAllreduce(None, "cuda:0", 256 * MIB).max_size == 256 * MIB
        assert (
            _CustomAllreduce(group=None, device="cuda:0", max_size=64 * MIB).max_size
            == 64 * MIB
        )
    assert _CustomAllreduce(group=None, device="cuda:0").max_size == AITER_POOL_BYTES


def test_pool_restores_the_constructor_on_error(env):
    with pytest.raises(RuntimeError), custom_ar_pool(128 * MIB):
        raise RuntimeError("init failed")
    assert _CustomAllreduce(group=None, device="cuda:0").max_size == AITER_POOL_BYTES


@pytest.mark.parametrize("world_size, ag_max", [(8, 64 * MIB), (4, 128 * MIB)])
def test_pool_keeps_each_groups_default_all_gather_cutoff(env, world_size, ag_max):
    # aiter's default is pool / (2 * group size): at its 1 GiB pool, 64 MiB for
    # a group of 8 and 128 MiB for a group of 4.
    with custom_ar_pool(128 * MIB):
        comm = _CustomAllreduce(group=world_size, device="cuda:0")
    assert comm.ag_max == str(ag_max)
    assert AG_MAX_ENV not in os.environ


def test_pool_keeps_an_explicit_all_gather_cutoff(env):
    env.setenv(AG_MAX_ENV, str(8 * MIB))
    with custom_ar_pool(128 * MIB):
        comm = _CustomAllreduce(group=4, device="cuda:0")
    assert comm.ag_max == str(8 * MIB)
    assert os.environ[AG_MAX_ENV] == str(8 * MIB)


@pytest.mark.parametrize(
    "ar_max, pool", [(None, 64 * MIB), (str(128 * MIB), 128 * MIB)]
)
def test_pool_below_twice_the_all_reduce_cutoff_is_refused(env, ar_max, pool):
    # A prefill all-reduce takes the custom kernels up to half the pool, so a
    # smaller pool would move some to RCCL.
    if ar_max is not None:
        env.setenv(AR_MAX_ENV, ar_max)
    refused = pytest.raises(ValueError, match="ATOM_CUSTOM_AR_POOL_BYTES")
    with refused, custom_ar_pool(pool):
        pass
    assert _CustomAllreduce(group=None, device="cuda:0").max_size == AITER_POOL_BYTES
    assert AG_MAX_ENV not in os.environ


def test_unset_pool_changes_nothing(env):
    with custom_ar_pool(None):
        assert (
            _CustomAllreduce(group=None, device="cuda:0").max_size == AITER_POOL_BYTES
        )
    assert AG_MAX_ENV not in os.environ


class _PyNccl:
    def __init__(self, disabled=False):
        self.disabled = disabled
        self.calls = []

    def broadcast(self, tensor, src, stream=None):
        self.calls.append(("pynccl", src))


class _Group:
    def __init__(self, world_size, pynccl):
        self.world_size = world_size
        self.device_communicator = type("Comm", (), {"pynccl_comm": pynccl})()
        self.calls = []

    def broadcast(self, tensor, src=0):
        self.calls.append(("torch", src))
        return tensor


def test_broadcast_runs_on_the_groups_pynccl_communicator():
    pynccl = _PyNccl()
    group = _Group(8, pynccl)
    tensor = object()
    assert comm_memory.group_broadcast(group, tensor) is tensor
    assert pynccl.calls == [("pynccl", 0)]
    assert group.calls == []


@pytest.mark.parametrize("pynccl", [None, _PyNccl(disabled=True)])
def test_broadcast_without_a_pynccl_communicator_uses_the_process_group(pynccl):
    group = _Group(8, pynccl)
    tensor = object()
    assert comm_memory.group_broadcast(group, tensor) is tensor
    assert group.calls == [("torch", 0)]


def test_broadcast_on_one_rank_does_nothing():
    pynccl = _PyNccl()
    group = _Group(1, pynccl)
    comm_memory.group_broadcast(group, object())
    assert pynccl.calls == [] and group.calls == []


def test_host_barrier_waits_for_the_device_then_the_gloo_group(monkeypatch):
    import torch

    calls = []
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(
        torch.distributed,
        "barrier",
        lambda group=None: calls.append(("barrier", group)),
    )
    cpu_group = object()
    comm_memory.host_barrier(cpu_group)
    assert calls == ["sync", ("barrier", cpu_group)]


def test_torch_group_communicator_built_by_one_all_to_all(monkeypatch):
    import torch

    calls = []

    def all_to_all_single(out, inp, group=None):
        calls.append((tuple(inp.shape), inp.dtype, group))

    monkeypatch.setattr(torch.distributed, "all_to_all_single", all_to_all_single)
    device_group = object()
    group = type(
        "G", (), {"world_size": 8, "device": "cpu", "device_group": device_group}
    )()
    comm_memory.build_torch_group_communicator(group)
    assert calls == [((8,), torch.int32, device_group)]
    group.world_size = 1
    comm_memory.build_torch_group_communicator(group)
    assert len(calls) == 1


def _stub_dcp(monkeypatch, tp_device_group, dcp_device_group, dcp_size):
    """aiter's parallel state, stubbed (CI has no aiter): TP and DCP groups over
    the same 8 ranks, as separate coordinators (no communicator sharing)."""
    import sys
    import types

    from atom.distributed import dcp_utils

    def group(device_group):
        return types.SimpleNamespace(
            ranks=list(range(8)), world_size=8, device="cpu", device_group=device_group
        )

    tp, dcp = group(tp_device_group), group(dcp_device_group)
    parallel_state = types.ModuleType("aiter.dist.parallel_state")
    parallel_state.get_tp_group = lambda: tp
    parallel_state.get_dcp_group = lambda: dcp
    for name, module in (
        ("aiter", types.ModuleType("aiter")),
        ("aiter.dist", types.ModuleType("aiter.dist")),
        ("aiter.dist.parallel_state", parallel_state),
    ):
        module.__path__ = getattr(module, "__path__", [])
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(dcp_utils, "get_dcp_world_size", lambda: dcp_size)
    dcp_utils.get_dcp_group.cache_clear()
    return dcp_utils


@pytest.mark.parametrize("dcp_size, built", [(8, ["tp"]), (1, [])])
def test_dcp_communicator_is_the_one_the_combine_uses(monkeypatch, dcp_size, built):
    # DCP = TP without communicator sharing: the combine's all-to-all runs on
    # the TP group (dcp_utils.get_dcp_group), so that is the one warmup builds.
    import torch

    calls = []
    monkeypatch.setattr(
        torch.distributed,
        "all_to_all_single",
        lambda out, inp, group=None: calls.append(group),
    )
    dcp_utils = _stub_dcp(monkeypatch, "tp", "dcp", dcp_size)
    try:
        comm_memory.build_dcp_communicator()
    finally:
        dcp_utils.get_dcp_group.cache_clear()
    assert calls == built
