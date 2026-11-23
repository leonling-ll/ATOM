from aiter_stub import stubbed_aiter

# model_runner imports aiter at module level, which a CPU runner does not have.
with stubbed_aiter():
    from atom.model_engine.model_runner import warmup_token_sweep


def test_off_by_default():
    assert warmup_token_sweep("", 16384) == []


def test_auto_ladder_covers_the_tuned_buckets():
    got = warmup_token_sweep("auto", 16384)
    # MiniMax-M3.1-Pro's tuned MXFP8 GEMM M values and MoE token buckets
    tuned = [
        1,
        2,
        4,
        8,
        16,
        32,
        48,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        176,
        192,
        208,
        224,
        240,
        256,
        288,
        320,
        352,
        384,
        416,
        448,
        480,
        512,
        768,
        1024,
        2048,
        4096,
        8192,
        16384,
    ]
    assert set(tuned) <= set(got)
    assert got == sorted(set(got)) and max(got) == 16384


def test_limit_and_list():
    assert max(warmup_token_sweep("auto", 4096)) == 4096
    assert warmup_token_sweep("64, 8,0,99999", 16384) == [8, 64]
