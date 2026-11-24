# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3.1-Pro mono decode, phase 1: one persistent kernel per MoE layer
from o_proj through the FFN return, at TP8 and up to ``config.MAX_ROWS`` rows.
Off unless ``ProVariants.mono_ffn_rows`` is set (``config.latch_rows``)."""
