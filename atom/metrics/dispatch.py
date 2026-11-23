"""How each real forward ran: replayed from a captured CUDA graph or eagerly.

The runner decides per step (`ForwardMode.decide` for the target, the draft
pass's own recordings for the drafter); this counts the outcome, so "no eager
decode steps" is read off the server's /metrics instead of inferred from a
batch-size histogram and the capture ladder. One TP rank records, so a count is
steps, not steps times ranks.
"""

from prometheus_client import Counter

_MODELS = ("target", "draft")
_PHASES = ("prefill", "decode")
_GRAPH = ("true", "false")


class ForwardDispatchMetrics:
    def __init__(self, *, dp_rank=0, engine_role="default", registry=None):
        self.steps = Counter(
            "atom:forward_steps",
            "Real forward passes by model (target, or the drafter's declared "
            "pass), step phase and whether they replayed a captured CUDA graph; "
            "excludes dummy DP steps.",
            ["dp_rank", "engine_role", "model", "phase", "cudagraph"],
            registry=registry,
        )
        # Every series exists from startup, so a zero reads as zero, not absent.
        self._series = {
            (model, phase, graph): self.steps.labels(
                str(dp_rank), engine_role, model, phase, graph
            )
            for model in _MODELS
            for phase in _PHASES
            for graph in _GRAPH
        }

    def record(self, model: str, is_prefill: bool, cudagraph: bool) -> None:
        self._series[
            (
                model,
                "prefill" if is_prefill else "decode",
                "true" if cudagraph else "false",
            )
        ].inc()
