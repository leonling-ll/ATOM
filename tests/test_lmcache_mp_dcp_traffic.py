# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""LMCache MP under DCP: bounded saves, lookups only for requests that may load.

Under DCP every rank stores and looks up its own shard, so one save is
``dcp_size`` server stores and one lookup ``dcp_size`` key rows a chunk. Saves
past what the server stores queue there until one outlives the transfer
deadline, which stops the engine; a lookup whose answer cannot load anything
only refreshes the tier's LRU. At DCP 1 nothing changes.
"""

from types import SimpleNamespace

import pytest
from test_lmcache_mp import _config, _m3_pro
from test_lmcache_mp_async_lookup import CHUNK, _Adapter, _Client

from atom.kv_transfer.offload.chunked_scheduler import ChunkedOffloadSchedulerBase
from atom.kv_transfer.offload.mp import lookup as mp_lookup
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from atom.kv_transfer.offload.mp.native_state_scheduler import (
    NativeStateLMCacheMPConnectorScheduler,
)
from tests.lmcache_mp_support import bare_mp_scheduler

MIN_LOAD = 32


def _built(monkeypatch, config):
    """The scheduler's own `__init__` over the async-capable fake adapter."""
    monkeypatch.setattr(
        mp_scheduler,
        "_make_scheduler_adapter",
        lambda _config, **_kwargs: _Adapter(_Client(0)),
    )
    return mp_scheduler.LMCacheMPConnectorScheduler(config)


def _scheduler(*, dcp, cap=None, resident=None, lookup_done=False):
    """A bare MP scheduler as `__init__` leaves it at `dcp`, with a block
    manager whose HBM prefix walk answers `resident` (tokens by request id)."""
    adapter = _Adapter(_Client(8, lookup_done=lookup_done))
    client = mp_lookup._MPLookupClient(
        adapter, config=_config(), timeout=1.0, poll_interval=0.01
    )
    sched = bare_mp_scheduler()
    sched._mp_adapter = adapter
    ChunkedOffloadSchedulerBase.__init__(
        sched, _config(), chunk_size=CHUNK, lookup_client=client
    )
    sched._min_load_tokens = MIN_LOAD
    if dcp > 1:
        sched._dcp_lookups_for_loads = True
        sched._max_pending_saves = cap
    resident = resident or {}
    sched._block_manager = SimpleNamespace(
        resident_prefix_blocks=lambda seq: resident.get(seq.id, 0) // CHUNK,
        hash_block_size=CHUNK,
    )
    return sched, client, adapter


def _seq(sid=7, n=64):
    return SimpleNamespace(
        id=sid,
        num_tokens=n,
        num_prompt_tokens=n,
        num_cached_tokens=0,
        token_ids=list(range(n)),
        block_table=list(range(100, 100 + n)),
        cache_seed=-1,
        offload_loaded=False,
        offload_load_failed=False,
    )


def _looked_up(adapter):
    return [key[1] for key, _tp in adapter._client.lookups]


def _rid(client, sid):
    return mp_lookup._mp_session_id(client._config, str(sid))


def _saves(meta):
    return [req for req in meta.requests if req.save_spec is not None]


# ── configuration ────────────────────────────────────────────────────────


def test_dcp1_keeps_every_lookup_and_unbounded_saves(monkeypatch):
    monkeypatch.setenv("OFFLOAD_MAX_PENDING_SAVES", "4")
    sched = _built(monkeypatch, _config())
    assert sched._dcp_lookups_for_loads is False
    assert sched._max_pending_saves is None
    sched._save_inflight.update({str(i): object() for i in range(64)})
    assert sched._may_emit_save()


def test_dcp8_bounds_saves_at_16_by_default(monkeypatch):
    monkeypatch.delenv("OFFLOAD_MAX_PENDING_SAVES", raising=False)
    sched = _built(monkeypatch, _m3_pro(tp=8, dcp=8))
    assert sched._dcp_lookups_for_loads is True
    assert sched._max_pending_saves == mp_scheduler._DCP_MAX_PENDING_SAVES == 16
    sched._save_inflight.update({str(i): object() for i in range(15)})
    assert sched._may_emit_save()
    sched._save_inflight["15"] = object()
    assert not sched._may_emit_save()


def test_dcp8_bound_follows_the_env_then_the_connector_config(monkeypatch):
    monkeypatch.setenv("OFFLOAD_MAX_PENDING_SAVES", "32")
    assert _built(monkeypatch, _m3_pro(tp=8, dcp=8))._max_pending_saves == 32
    config = _m3_pro(tp=8, dcp=8)
    config.kv_transfer_config["kv_connector_extra_config"]["max_pending_saves"] = 4
    assert _built(monkeypatch, config)._max_pending_saves == 4


def test_native_state_scheduler_keeps_its_own_bounds_and_lookups():
    assert NativeStateLMCacheMPConnectorScheduler._bounds_dcp_traffic is False
    assert mp_scheduler.LMCacheMPConnectorScheduler._bounds_dcp_traffic is True


# ── saves ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(("dcp", "cap", "first"), [(1, None, 3), (8, 2, 2)])
def test_save_emission_stops_at_the_bound_and_resumes(dcp, cap, first):
    sched, _, _ = _scheduler(dcp=dcp, cap=cap)
    for sid in (7, 8, 9):
        seq = _seq(sid, n=16)
        seq.num_cached_tokens = 8
        sched.update_state_after_alloc(seq)

    assert len(_saves(sched.build_connector_meta())) == first
    if cap is None:
        return
    assert sched._perf_counters["save_cap_block_steps"] == 1
    sid, operation = next(iter(sched._save_inflight.items()))
    sched.save_finished(operation)
    assert sid not in sched._save_inflight
    assert len(_saves(sched.build_connector_meta())) == 1


# ── lookups ──────────────────────────────────────────────────────────────


def test_dcp8_prefetches_only_requests_the_hbm_prefix_leaves_a_load_for():
    covered, partial, cold, short = _seq(7), _seq(8), _seq(9), _seq(10, n=16)
    sched, client, adapter = _scheduler(
        dcp=8, cap=16, resident={7: 64 - MIN_LOAD + 4, 8: 64 - MIN_LOAD}
    )
    # The scheduler passes an iterator over the waiting queue.
    sched.prefetch_lookups(iter([covered, partial, cold, short]))
    # A short prompt without an HBM hit is not detached at admission (no hit),
    # so its lookup stays off the admission path as before.
    assert _looked_up(adapter) == [_rid(client, sid) for sid in (8, 9, 10)]


def test_dcp1_prefetches_every_request():
    sched, client, adapter = _scheduler(dcp=1, resident={7: 64})
    sched.prefetch_lookups([_seq(7)])
    assert _looked_up(adapter) == [_rid(client, 7)]


def test_dcp8_detach_admits_without_a_lookup_and_keeps_the_hbm_floor():
    sched, client, adapter = _scheduler(dcp=8, cap=16, lookup_done=True)
    sched._do_save = True
    seq = _seq()

    assert sched.detach_lookup(seq, 62) is True
    assert adapter._client.lookups == []
    assert "7" not in client.pending_ids()
    assert sched._detached_lookups == {}
    assert sched._hit_save_floors["7"] == 60
    assert sched._perf_counters["lookup_detached_unsent"] == 1

    seq.num_cached_tokens = 62
    sched.update_state_after_alloc(seq)
    # No answer is awaited, so nothing resets the floor to 0.
    sched._end_finished_sessions()
    assert sched._save_tracker["7"][1] == 60


def test_dcp8_detach_keeps_a_lookup_a_prefetch_sent():
    # The walk saw a gap; by admission the HBM prefix covers the prompt.
    sched, client, adapter = _scheduler(dcp=8, cap=16, resident={7: 16})
    seq = _seq()
    sched.prefetch_lookups([seq])
    assert sched.detach_lookup(seq, 60) is True
    assert len(adapter._client.lookups) == 1
    assert "7" in client._orphans
    assert "7" in sched._detached_lookups


def test_dcp8_detach_refuses_a_gap_the_tier_could_fill():
    sched, _, adapter = _scheduler(dcp=8, cap=16)
    assert sched.detach_lookup(_seq(), 64 - MIN_LOAD) is False
    assert adapter._client.lookups == []


@pytest.mark.parametrize(("dcp", "offered"), [(1, 1), (8, 0)])
def test_lookups_are_offered_on_arrival_only_at_dcp1(dcp, offered):
    sched, _, _ = _scheduler(dcp=dcp, cap=16)
    sent = []
    sched._lookup_client = SimpleNamespace(
        driving=True, offer=lambda sid, _tokens, **_kw: sent.append(sid)
    )
    sched.offer_lookups([_seq()])
    assert len(sent) == offered
