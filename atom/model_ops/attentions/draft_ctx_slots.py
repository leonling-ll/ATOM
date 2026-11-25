# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The context slots of a draft that owns a KV pool outside the DCP group.

Such a draft (``kv_pool_spec`` set) stores every token of the target forward at
its pool's entry size, ``block_size * dcp``. Under DCP the target's
``slot_mapping`` is DCP-filtered and in ``block_size`` units, so it cannot
address that pool: the target's builder stages the draft's own slots as
``forward_vars["draft_ctx_slots"]``, which ``DSparkProposer.compute_draft_kv``
writes the context K/V through. At DCP 1 the draft writes through the target's
slots and none are staged.

A builder calls ``_declare_draft_ctx_slots`` once from its ``__init__``, adds
``_draft_ctx_h2d`` to its ``h2d_group_members`` and calls
``_stage_draft_ctx_slots`` after its prefill and decode staging.
"""

import numpy as np
import torch

from atom.utils import CpuGpuBuffer

from .token_layout.slots import slot_mapping


class DraftCtxSlotsMixin:
    def _declare_draft_ctx_slots(self, model_runner) -> None:
        """A draft that declares a pool of its own stores every token of the
        target forward at that pool's entry size. Under DCP that is not what
        the target's slots address (DCP-filtered, `block_size` tokens), so it
        gets slots of its own; at DCP 1 it writes through the target's."""
        drafter = getattr(model_runner, "drafter", None)
        spec = getattr(getattr(drafter, "model", None), "kv_pool_spec", None)
        if spec is not None and spec.needs_own_ctx_slots(model_runner.config):
            model_runner.forward_vars["draft_ctx_slots"] = CpuGpuBuffer(
                self.max_num_batched_tokens,
                dtype=torch.int64,
                device=self.device,
                publication_group="draft_ctx_slots",
            )

    def _draft_ctx_h2d(self, members: dict) -> dict:
        """The draft's slots go up with either phase's attention metadata
        (`stage_prefill` / `stage_decode`)."""
        if "draft_ctx_slots" in self.model_runner.forward_vars:
            for phase in ("prefill_metadata", "decode_metadata"):
                members[phase] = members.get(phase, ()) + ("draft_ctx_slots",)
        return members

    def _stage_draft_ctx_slots(
        self,
        batch,
        seqlens_q: np.ndarray,
        rows: int,
        group,
        prefix: str = "",
        first: int = 0,
    ) -> None:
        """The draft's own slot for each scheduled token, `-1` out to `rows`,
        the width the target's `slot_mapping` goes up at, counted into
        ``group``.

        Runs after the target's staging, whose host positions and packed
        block table (the ``prefix`` set) it reads. No DCP filter, as the draft
        stores every token, and `entry_tokens` per block id. ``first``: the
        step's row these tokens start at (a mixed step's decode half follows
        its prefill rows); ``rows`` counts from the step's first row.
        """
        var = self.model_runner.forward_vars
        buf = var.get("draft_ctx_slots")
        if buf is None or batch.is_dummy_run:
            return
        n = int(seqlens_q.sum())
        buf.np[first + n : rows] = -1
        if batch.block_tables:
            slot_mapping(
                var[f"{prefix}positions"].np[:n],
                seqlens_q,
                var[f"{prefix}block_tables"].np,
                self.model_runner.draft_kv_builder.kv_pool.entry_tokens,
                out=buf.np[first : first + n],
                scratch=self.token_axis_scratch,
            )
        else:
            buf.np[first : first + n] = -1
        group.set_count(buf, rows)
