# MiniMax-M3.1-Pro: switches that trade numerics for speed

Every optimization on `yuhua/m3.1-pro` that can change Pro's outputs has its own switch, so an
accuracy regression can be bisected by turning them off one at a time. This page lists each one:
what it changes numerically, how far that goes, the accuracy measured with it, and what it buys.
Switches that are bit-identical by construction are listed at the end.

Switches named `x` are `ATOM_M3PRO_VARIANTS` fields (`'{"x": ...}'`); the rest are environment
variables. "Recipe" is the value in [`Agentic-MiniMax-M3.1-Pro.md`](Agentic-MiniMax-M3.1-Pro.md).

Accuracy numbers are GSM8K 64-shot flexible-extract, one run, DSpark k=7 with real acceptance,
1319 questions. One run's standard error is about ±0.0066, so differences under ~0.013 between
single runs are noise. The gate is ≥ 0.933. Retrieval is the 32k / 64k key-value retrieval
(20 questions each), which only passes when the indexer selects the block that holds the key.

## Switches that change numerics

| Switch | Recipe | Off | What changes numerically | Range of the change | Accuracy measured | Gain |
|---|---|---|---|---|---|---|
| `index_compress` | `true` | `false` | The index cache stores one bf16 centroid per 4 tokens, pooled once at write time from bf16 keys, instead of one fp8 key per token that is pooled again at every score. The scoring formula (`pool4` + max over heads) is unchanged. | Keys rounded to bf16 instead of fp8, centroid stored in bf16. 122-123.5 of the 128 selected blocks match today's fp8 path (~96%); the rest are near-ties. | GSM8K **0.9462**; retrieval 32k 20/20, 64k 20/20 | Index bytes per token halved: **+17.7% KV blocks** (160,613 → 189,090). Indexer scoring ~4× faster (c64 verify 209.8 → 47.5 µs a layer at 32k; 16k prefill 5536 → 1276 µs). The pooling write adds ~35 µs a layer eager, less in a CUDA graph. AgentX c64 (stub template, 600 s): P90 30.22 → 34.43 (+13.9%), 11,254 → 11,569 tok/s/GPU (+2.8%). |
| `day0_decode_rows` (24) and `day0_decode_mxfp8` (true) | on when the `amd_kernels` package imports | `0` / `false` | Decode / verify batches from 24 rows attend with amd-kernels-day0's FlyDSL sparse decode, which accumulates in a different order from aiter's paged attention and writes `o_proj`'s MXFP8 input itself. | Same selected blocks and fp8 K. Differences are fp32 accumulation order plus where the MXFP8 rounding of the attention output happens; `o_proj` quantized its input to MXFP8 before this change too. | Batch 1 (with `moe_route_glue`, the MXFP8 GEMM rows and `index_split_decode`): GSM8K **0.9424** | Part of batch 1: c64 P90 21.51 → 30.22, 9562 → 11254 tok/s/GPU. Kernel 1.9× / 3.4× / 5.9× of Triton at 64 / 128 / 256 rows. |
| `moe_route_glue` | `true` | `false` | A MoE layer's all-reduce, RMSNorm (+ MXFP8 copy), router GEMM, sigmoid top-k and sort run as one aiter launch. The router logits are bf16 with a different summation order. | Router scores differ by bf16 rounding, so two experts whose scores tie to within that rounding can swap. The MXFP8 scale rule is unchanged. | Batch 1: GSM8K **0.9424** | Six launches fewer per MoE layer at decode; part of batch 1. |
| `GPTOSS_SWIGLU_MXFP4_BF16_BOUND` | `1040` | unset (aiter's 256) | Below the bound the routed MoE takes bf16 activations (a16w4); at or above, MXFP4 activations (a4w4). Prefill chunks stay a4w4. | Activations quantized to MXFP4 above the bound. With aiter's 256, DSpark verify steps (sequences × 8 rows) cross it. | 1040: **0.9393**. 256 (5 runs): 0.926-0.937, mean ~0.930. Prefill in bf16 too (bound 1e6): 0.9484, within noise of 1040. | 1040 is the accuracy fix; a16w4 prefill would be 2.6-3.8× slower. |
| `k_plain_fp8` | `true` | `false` | The fp8 K cache is a plain clamped cast (scale 1.0) instead of per-token scaled. | The normed K is O(1), inside e4m3's range; rounding is e4m3's at unit scale rather than at the row's own scale. | With `k_only` and the aiter prep: GSM8K 0.9393-0.9462 over runs | Needed by day0's prefill and decode kernels. |
| `attn_fp8_compute` | `true` | `false` | aiter's paged attention runs Q·K and P·V in fp8 on an fp8 cache. | Q and P rounded to fp8 inside the attention. Only reached by rows that don't take the day0 decode. | In every recipe GSM8K above; ruled out as the cause of the DSpark drop | Faster aiter decode below 24 rows. |
| `ar_lamport_rows` | `16` | `0` | Steps of ≤16 rows run the all-reduce + norm as a Lamport all-reduce plus a separate norm. | At most 1 bf16 ulp from the fused kernel (rank-order sum, separate rounding). | GSM8K 0.9378 (baselines 0.9416 / 0.9447) | bs1 decode step −0.18 ms (−1.4%). No effect at c64, where steps are >16 rows. |
| aiter MXFP8 GEMM rows (the c64 retune merged with day0's) | built into aiter `yuhua/m3.1-pro` | none (config rows) | Picks a different kernel / tile / split-K per shape. | fp32 accumulation order only. | Batch 1: 0.9424 | Part of batch 1. |

## Bit-identical switches

These produce the same bits as their off position by construction (each has a CPU test), so they
don't need an accuracy run:

- `index_split` and `index_split_decode`: each TP rank scores and selects a share of the rows; the
  selections are all-gathered. Every row's selection is computed exactly as before.
- `score_tq`, `score_tq_causal`: the same scores, more rows per program and only causal tiles.
- `topk: "split"`: the same picks as `select_logical`.
- `fuse_mxfp8_quant`: the norms write the same MXFP8 bytes `quantize_fp8` would.
- `AITER_REUSE_IDENTICAL_COMM_GROUPS`, `ATOM_ONLINE_QUANT_STREAMING` (weights hash-identical),
  `--text-only`, `ATOM_CACHE_AWARE_ADMISSION`, LMCache: no numerics change.
- DSpark `num_speculative_tokens`: rejection sampling keeps the target distribution; k changes
  speed only. The forced acceptance length (`--spec-decode-acceptance-length`) replaces output
  text and is for throughput runs only.

## Not merged yet

These are switchable on their branches and will be listed above once they are merged:

- `ATOM_SHARDED_SAMPLING` (vocab-sharded sampling): same distribution, greedy bit-identical.
- `--enable-tbo` with F1 split collectives: changes all-reduce grouping in prefill.
- amd-kernels shuffle16 decode GEMMs: cosine ≥ 0.9999999, relative L2 ≤ 4.2e-5 per GEMM.
- F4, routed + shared expert outputs written in one pass: heavier error tail at the atomic stage-2 rows.
- day0 `wtshare` prefill attention.

Two have been tried and are kept out:

- In-place atomic MoE accumulate: 3-run GSM8K mean 0.9353.
- Shared-expert quantization (FSE): breaks the InferenceX rule that the shared expert stays MXFP8.
