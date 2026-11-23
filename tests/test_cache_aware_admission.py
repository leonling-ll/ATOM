"""`ATOM_CACHE_AWARE_ADMISSION`: waiting requests with a resident prefix are admitted first."""

import time

from conftest import MockConfig

from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler


def _scheduler(monkeypatch, enabled=True, max_num_seqs=4):
    monkeypatch.setenv("ATOM_CACHE_AWARE_ADMISSION", "1" if enabled else "0")
    return Scheduler(
        MockConfig(
            enable_prefix_caching=True,
            kv_cache_block_size=4,
            num_kvcache_blocks=20,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=256,
        )
    )


def _warm(sched, seq_factory):
    """Run a first turn to completion so its two full blocks stay indexed."""
    seq = seq_factory([1, 2, 3, 4, 5, 6, 7, 8, 9])
    sched.add(seq)
    batch, _ = sched.schedule()
    seq.append_token(2)
    sched.postprocess(
        list(sched.running),
        ScheduledBatchOutput(
            req_ids=[seq.id],
            token_ids=[(2,)],
            num_rejected=None,
            num_bonus=None,
            draft_token_ids=None,
        ),
        batch=batch,
    )


def _queue(sched, seq_factory, cold_age=0.0):
    """A cold request, then the warm session's next turn behind it."""
    now = time.time()
    cold = seq_factory([50, 51, 52, 53, 54, 55, 56, 57, 58])
    warm = seq_factory([1, 2, 3, 4, 5, 6, 7, 8, 10, 11])
    cold.arrive_time, warm.arrive_time = now - cold_age, now
    sched.add(cold)
    sched.add(warm)
    return cold, warm


def test_resident_prefix_goes_first(monkeypatch, seq_factory):
    sched = _scheduler(monkeypatch)
    _warm(sched, seq_factory)
    cold, warm = _queue(sched, seq_factory)
    assert sched.block_manager.resident_prefix_blocks(warm) == 2
    assert sched.block_manager.resident_prefix_blocks(cold) == 0
    sched._cache_aware_reorder()
    assert list(sched.waiting) == [warm, cold]


def test_request_past_max_wait_keeps_its_place(monkeypatch, seq_factory):
    monkeypatch.setenv("ATOM_CACHE_AWARE_MAX_WAIT_S", "30")
    sched = _scheduler(monkeypatch)
    _warm(sched, seq_factory)
    cold, warm = _queue(sched, seq_factory, cold_age=60.0)
    sched._cache_aware_reorder()
    assert list(sched.waiting) == [cold, warm]


def test_schedule_admits_the_warm_turn_first(monkeypatch, seq_factory):
    sched = _scheduler(monkeypatch, max_num_seqs=1)
    _warm(sched, seq_factory)
    cold, _ = _queue(sched, seq_factory)
    batch, _ = sched.schedule()
    assert batch.num_scheduled_tokens == [2]  # the warm turn, 8 of 10 tokens from cache
    assert list(sched.waiting) == [cold]


def test_off_by_default_keeps_fifo(monkeypatch, seq_factory):
    monkeypatch.delenv("ATOM_CACHE_AWARE_ADMISSION", raising=False)
    sched = Scheduler(
        MockConfig(
            enable_prefix_caching=True,
            kv_cache_block_size=4,
            num_kvcache_blocks=20,
            max_num_seqs=1,
            max_num_batched_tokens=256,
        )
    )
    _warm(sched, seq_factory)
    _, warm = _queue(sched, seq_factory)
    batch, _ = sched.schedule()
    assert batch.num_scheduled_tokens == [9]
    assert list(sched.waiting) == [warm]
