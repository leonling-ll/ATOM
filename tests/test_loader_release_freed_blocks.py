# SPDX-License-Identifier: MIT
"""Post-load processing hands freed blocks back to the driver between steps.

Processing a module (online quantization, then the quant method's shuffle)
frees temporaries and allocates the processed weights. Left cached, a freed
temporary's block is where the caching allocator puts the next persistent
copy, and the rest of that segment is stranded for the life of the weight:
on MiniMax-M3.1-Pro about 40 GB per GPU, missing from the KV pool. The loader
therefore empties the cache after each processing step that left more than a
little cached-but-free memory behind.
"""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from torch import nn

GB = 1 << 30


@pytest.fixture(scope="module")
def loader():
    with atom_without_aiter() as load:
        return load("atom.model_loader.loader")


class FakeAllocator:
    """Reserved and allocated bytes, and a log of what touched them."""

    def __init__(self):
        self.reserved = 200 * GB
        self.allocated = 200 * GB
        self.log = []

    def free_temporary(self, nbytes):
        self.allocated -= nbytes  # freed but still cached

    def empty_cache(self):
        self.log.append("empty_cache")
        self.reserved = self.allocated


@pytest.fixture
def allocator(loader, monkeypatch):
    a = FakeAllocator()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *_: a.reserved)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: a.allocated)
    monkeypatch.setattr(torch.cuda, "empty_cache", a.empty_cache)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: a.allocated)
    monkeypatch.setattr(loader, "load_weights_into_model", lambda **_: set())
    monkeypatch.setattr(
        loader.OnlineQuantStreamer, "maybe_create", classmethod(lambda *_: None)
    )
    return a


def _quant_method(loader, allocator, name, frees):
    class Method(loader.QuantizeMethodBase):
        def create_weights(self, *args, **kwargs):
            raise NotImplementedError

        def apply(self, *args, **kwargs):
            raise NotImplementedError

        def process_weights_after_loading(self, layer):
            allocator.log.append(f"{name}.shuffle")
            allocator.free_temporary(frees)

    return Method()


class Layer(nn.Module):
    """Online quantization in its own hook, then its quant method's shuffle."""

    def __init__(self, loader, allocator, name, frees):
        super().__init__()
        self.allocator = allocator
        self.name = name
        self.frees = frees
        self.quant_method = _quant_method(loader, allocator, name, frees)

    def process_weights_after_loading(self):
        self.allocator.log.append(f"{self.name}.quantize")
        self.allocator.free_temporary(self.frees)


def _load(loader, *layers):
    model = nn.Sequential(*layers)
    loader.load_model(model, "unused", SimpleNamespace(), load_dummy=None)


def test_each_processing_step_starts_on_an_emptied_cache(loader, allocator):
    _load(
        loader,
        Layer(loader, allocator, "l0", 2 * GB),
        Layer(loader, allocator, "l1", 2 * GB),
    )

    # Every step that freed a temporary is followed by a release, so the next
    # step's persistent allocations cannot land in that temporary's block.
    assert allocator.log == [
        "l0.quantize",
        "empty_cache",
        "l0.shuffle",
        "empty_cache",
        "l1.quantize",
        "empty_cache",
        "l1.shuffle",
        "empty_cache",
    ]
    assert allocator.reserved == allocator.allocated


def test_small_slack_is_left_cached(loader, allocator):
    # A few MB of slack is not worth a device sync per module.
    _load(loader, Layer(loader, allocator, "l0", 8 << 20))

    assert "empty_cache" not in allocator.log
