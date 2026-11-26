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
| `day0_decode_rows` (24) and `day0_decode_mxfp8` (true) | on when the aiter build has `minimax_m3_fp8_sparse_decode` | `0` / `false` | Decode / verify batches from 24 rows attend with amd-kernels-day0's FlyDSL sparse decode, which accumulates in a different order from aiter's paged attention and writes `o_proj`'s MXFP8 input itself. | Same selected blocks and fp8 K. Differences are fp32 accumulation order plus where the MXFP8 rounding of the attention output happens; `o_proj` quantized its input to MXFP8 before this change too. | Batch 1 (with `moe_route_glue`, the MXFP8 GEMM rows and `index_split_decode`): GSM8K **0.9424** | Part of batch 1: c64 P90 21.51 → 30.22, 9562 → 11254 tok/s/GPU. Kernel 1.9× / 3.4× / 5.9× of Triton at 64 / 128 / 256 rows. |
| `moe_route_glue` | `true` | `false` | A MoE layer's all-reduce, RMSNorm (+ MXFP8 copy), router GEMM, sigmoid top-k and sort run as one aiter launch. The router logits are bf16 with a different summation order. | Router scores differ by bf16 rounding, so two experts whose scores tie to within that rounding can swap. The MXFP8 scale rule is unchanged. | Batch 1: GSM8K **0.9424** | Six launches fewer per MoE layer at decode; part of batch 1. |
| `GPTOSS_SWIGLU_MXFP4_BF16_BOUND` | `1040` | unset (aiter's 256) | Below the bound the routed MoE takes bf16 activations (a16w4); at or above, MXFP4 activations (a4w4). Prefill chunks stay a4w4. | Activations quantized to MXFP4 above the bound. With aiter's 256, DSpark verify steps (sequences × 8 rows) cross it. | 1040: **0.9393**. 256 (5 runs): 0.926-0.937, mean ~0.930. Prefill in bf16 too (bound 1e6): 0.9484, within noise of 1040. | 1040 is the accuracy fix; a16w4 prefill would be 2.6-3.8× slower. |
| `k_plain_fp8` | `true` | `false` | The fp8 K cache is a plain clamped cast (scale 1.0) instead of per-token scaled. | The normed K is O(1), inside e4m3's range; rounding is e4m3's at unit scale rather than at the row's own scale. | With `k_only` and the aiter prep: GSM8K 0.9393-0.9462 over runs | Needed by day0's prefill and decode kernels. |
| `attn_fp8_compute` | `true` | `false` | aiter's paged attention runs Q·K and P·V in fp8 on an fp8 cache. | Q and P rounded to fp8 inside the attention. Only reached by rows that don't take the day0 decode. | In every recipe GSM8K above; ruled out as the cause of the DSpark drop | Faster aiter decode below 24 rows. |
| `ar_lamport_rows` | `16` | `0` | Steps of ≤16 rows run the all-reduce + norm as a Lamport all-reduce plus a separate norm. | At most 1 bf16 ulp from the fused kernel (rank-order sum, separate rounding). | GSM8K 0.9378 (baselines 0.9416 / 0.9447) | bs1 decode step −0.18 ms (−1.4%). No effect at c64, where steps are >16 rows. |
| `mono_ffn_rows` | `16` (the default) | `0` | Decode / verify forwards of ≤ 16 rows run each MoE layer's `o_proj`, all-reduce, RMSNorm (+ MXFP8 copy and the lookahead norm), router, routed and shared experts as one persistent kernel. Routed experts are A16W4 as in production below 1040 rows (bf16 activations and mids against the same MXFP4 weights); the router keeps fp32 logits as `moe_route_glue` does; the all-reduce sums in the Lamport kernel's rank order. | fp32 accumulation order inside the GEMMs. 8-GPU layer test against the torch reference: worst FFN rel 0.0142, bound 2^-5. | Mono on, Agentic section 1 variants, no spec, concurrency 16 (every decode forward on mono): GSM8K 64-shot 0.9333 / 0.9424 / 0.9371, mean 0.9376 (mono off, the #226 tree `9a6937a6` with DSpark k7: 0.9325, one run; below 0.94, accepted by the owner under #7's near-miss rule, #191). Retrieval 32k 20/20, 64k 19/20; the 64k miss repeats with mono off (#171). AIME26 avg@16 ×2: pending | c1 TPOT 8.19 → 6.05 ms, c4 9.15 → 7.48 ms (1k/1k, no spec); c1 DSpark k7 at AL 2.94 3.66 → 3.25 ms. No effect on forwards of more than 16 rows. |
| aiter MXFP8 GEMM rows (the c64 retune merged with day0's) | built into aiter `yuhua/m3.1-pro` | none (config rows) | Picks a different kernel / tile / split-K per shape. | fp32 accumulation order only. | Batch 1: 0.9424 | Part of batch 1. |
| `index_day0` | `true` in the CONC 64 launch | `false` | A split prefill's index scores and top-128 come from aiter-M3's FlyDSL `minimax_m3_index_top128` (amd-kernels-day0's `index_top128`, vLLM's prefill indexer, aiter-M3#24) instead of the Triton kernels. The scores are fp16 and the top-128 runs on 16-bit score keys, as in vLLM; the Triton path scores in fp32. | A block whose score ties the 128th within fp16 rounding can be swapped. The port matches amd-kernels-day0 bit for bit on 11 of 11 inputs. | GSM8K 0.9431 64-shot, 0.9356 5-shot (one run); AIME 2026 avg@16 0.9625 ± 0.0029. With day0's own kernel: 0.9439 / 0.9353 (3 runs). | CONC 64, 3600 s on one node: 16,360 → 16,798 tok/s/GPU (+2.7%), p90 43.30 → 44.65 (+3.1%). 600 s runs do not resolve it. |

## DCP 8

GSM8K here is 64-shot flexible-extract, DSpark k=5 at real acceptance, 3 runs, with the fp8 index
cache (`index_compress` off) and `dcp_topk_merge`. "Recipe" is the
[CONC 64 with DCP=8](Agentic-MiniMax-M3.1-Pro.md#conc-64-with-dcp8) launch. McNemar p-values are
over the 9 run pairs between two arms; the repeats inside one arm give 0.04-1.00.

- `--decode-context-parallel-size 8` (with DSpark): decode attends with a log-sum-exp merge of the
  eight ranks' partial outputs, and each rank scores its own blocks (within fp32 rounding of the
  single-rank scores, so a near-tie can pick another block). The draft's numerics are unchanged.
  GSM8K 0.9439 / 0.9401 / 0.9401, mean 0.9414, against DCP 1 0.9507 / 0.9378 / 0.9431, mean 0.9439,
  McNemar p 0.10-1.00: no shift beyond the repeats. The two arms are not a like-for-like pair: the
  DCP 1 run had `index_compress` on and ran on another tree (`8169708a`).
  - `dcp_mailbox` (rows; recipe 768): the merge in aiter's mailbox combine for captured steps up to
    that many rows, the same weighted sum in another order (bf16 output within one rounding).
  - `dcp_q_fp8` (recipe on): the decode query all-gathered as a plain e4m3 cast, as DCP1's day0
    decode reads it (q rounded to fp8; the bf16 DCP decode does not round it).
  - `dcp_fold` (recipe on): the selection stores the sort keys it all-gathers, the attention the
    LSE as the all-to-all merge reads it, and that merge the layer's output: the same bits.
  - The three above together: GSM8K 0.9363 / 0.9386 / 0.9424, mean 0.9391, against 0.9414 without
    them, McNemar p 0.25-0.90: no shift.
  - `dcp_day0_decode` (recipe on): decode and verify steps attend with aiter-M3's FlyDSL
    `minimax_m3_fp8_sparse_decode_dcp` instead of the Triton attention. q rounds to e4m3 (as with
    `dcp_q_fp8` and DCP 1's day0 decode) and P stays bf16 as in the Triton attention; the
    accumulation order differs. Op check against the reference: rel L2 1.9e-3, LSE within 4e-5, the
    Triton attention's level. GSM8K 0.9454 / 0.9363 / 0.9409, mean 0.9409, against 0.9414 without
    it, McNemar p 0.25-1.00: no shift. Gain: one rank's attention 4-6× faster (23-45 against
    90-238 µs a layer at 256-768 rows); part of the DCP 8 CONC 64 row's +6.5%.
  - `index_compress` + `dcp_index_day0` (both off by default): the bf16 centroid index cache under
    DCP (each rank pools the blocks it holds), and a prefill's split indexer runs `index_day0` over
    the centroids the prefill gathers: the same kernel on the same pages as DCP 1's `index_day0`,
    so the same picks (checked on all 8 ranks). GSM8K with both on top of the recipe's variants:
    0.9393 / 0.9416 / 0.9348, mean 0.9386, against 0.9409 with `dcp_day0_decode`, McNemar p
    0.11-1.00 (0.25-1.00 against the three switches above, 0.17-1.00 against plain DCP 8): no
    shift. Gain: one rank's prefill indexer 6.2-7.7× faster at 16K-row steps (311 against 2412 µs
    a layer at 16K rows, 124K context).

## Bit-identical switches

These produce the same bits as their off position by construction (each has a CPU test), so they
don't need an accuracy run:

- `index_split` and `index_split_decode`: each TP rank scores and selects a share of the rows; the
  selections are all-gathered. Every row's selection is computed exactly as before.
- `score_tq`, `score_tq_causal`: the same scores, more rows per program and only causal tiles.
- `topk: "split"`: the same picks as `select_logical`.
- `dcp_topk_merge` (DCP only): each rank's best `topk` blocks merged as sort keys, the same picks
  as `select_logical` over the all-gathered scores.
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
