# MiniMax-M3.1-Pro c32 agentic profile (TP8, CPU KV offload, DSpark k=7)

Where the GPU time of one c32 decode step and one prefill chunk goes on 8x MI355X, at the
fastest known c32 configuration, with achieved TFLOPS and HBM bandwidth per GEMM and MoE
shape and a breakdown of per-GPU HBM at startup. Later c32 tasks rank against these tables.
It ran the image's aiter, not the aiter-M3 build of the c32 nightly, so three paths run their
fallbacks (below); the shares of those paths are not the nightly's.

## Configuration

The server is the launch of [`Agentic-MiniMax-M3.1-Pro.md`](../../recipes/Agentic-MiniMax-M3.1-Pro.md)
at `CONC=32` (`--max-num-seqs 64`) with `--text-only` and `AITER_REUSE_IDENTICAL_COMM_GROUPS=1`:

| Item | Value |
|---|---|
| ATOM | `hatwu/m3.1-pro-infx` `5666260b` plus `057e95e6` from `yuhua/m3.1-pro` (`--text-only`) |
| Image, aiter | `rocm/atom-dev:nightly_202611201604`, aiter `572232d23`; no `amd_kernels` package |
| `ATOM_M3PRO_VARIANTS` | `{"kernels": true, "attn": "aiter", "topk": "split", "prep": "aiter", "index_split": true, "score_tq": 8, "ar_lamport_rows": 16, "k_only": true, "k_plain_fp8": true, "prefill_attn": "day0"}` |
| Environment | `ATOM_WARMUP_TOKENS=auto`, `ATOM_ONLINE_QUANT_STREAMING=1`, `ATOM_CACHE_AWARE_ADMISSION=1`, `AITER_REUSE_IDENTICAL_COMM_GROUPS=1`, `AITER_CUSTOM_AR_DISABLE_LAMPORT` unset, quick all-reduce off |
| CPU offload | `ATOM_KV_OFFLOAD=lmcache`, `LMCACHE_MAX_LOCAL_CPU_SIZE=184`, chunk 256, `PYTHONHASHSEED=0` |
| Server flags | TP8, `--kv_cache_dtype fp8 --index-cache-dtype fp8 --block-size 16 --level 0`, `--max-model-len 132096 --max-num-batched-tokens 16384 --gpu-memory-utilization 0.95`, `--online_quant_config` experts to MXFP4, `--method dspark --num-speculative-tokens 7 --spec-decode-acceptance-length 4.0`, `--text-only` |
| Load | `scripts/m31pro/proxy_bench.sh`'s aiperf replay at concurrency 32 (seed 42, 64 sessions capped at 128k) |

The c32 nightly of 2026-11-23 (7511 tok/s/GPU) ran this launch on aiter-M3 `dc53d8b5`. On this
image's aiter the server falls back on three paths:

- Decode attention is Triton's (`triton attention, K-only` in the log): on the K-only cache
  `MiniMaxM3ProAttention` takes aiter's paged attention only when `aiter_value_rows_available()`,
  and this aiter's `pa_decode_gluon` has no `value_rows`.
- The routed MoE runs FlyDSL's heuristic configs (`no tuned FlyDSL config ... using heuristic
  FlyDSL fallback` for every token count from 1 to 16384).
- The MXFP8 activation quantization is not fused into the norms and the all-reduce epilogue
  (`aiter MXFP8 norms False, all-reduce MXFP8 epilogue False`).

`k_plain_fp8` keeps the Triton prep, and without `amd_kernels` the prefill attention is Triton's.
The draft's acceptance is forced to 4.0 tokens a step, not Pro's measured 2.82 at k=7: that moves
how fast each request advances and so the batch mix, not the cost of a step at a given batch.

Two torch-profiler windows of 5 s (`/start_profile`, `/stop_profile`, `ATOM_TORCH_PROFILER_DIR`,
`ATOM_PROFILER_RECORD_SHAPES=1`, `ATOM_ENABLE_DETAILED_ANNOTATION=1`) were taken 180 s and 340 s
into aiperf's profiling phase, rank 0. Kernels are attributed to forwards by their launch's
`ATOM::forward` range (`scripts/m31pro/kernel_roofline.py`'s method), and step time is the wall
time from one forward's first kernel to the next forward's first kernel, so the categories below
sum to 100% and include GPU idle inside the step. Rates use the MI355X spec peaks: HBM 8.0 TB/s,
bf16 2.5, fp8 5.0, MXFP4 10.1 PFLOPS dense.

## Decode step (named result)

136 pure-decode forwards: **40.3 ms a step** (median 40.5 ms), 83.8 rows a step on average
(10.5 sequences x 8 verify rows), 60.6k context tokens a sequence. Every decode step replayed a
CUDA graph; GPU idle between forwards is under 10 us. The windows ran at 10.5 sequences a step
against 6.46 over the profiling phase (histogram below).

| Component | us a step | Share |
|---|---:|---:|
| MoE routed experts (FlyDSL a16w4 gemm1 + gemm2) | 11,604 | **28.8%** |
| Indexer score + top-k (`_score_qtile_kernel`, `_select_split_kernel`) | 11,041 | **27.4%** |
| Attention GEMMs (q/k, index q/k, o_proj; MXFP8) | 3,886 | **9.6%** |
| All-reduce (reduce-scatter, Lamport, 2-stage, RMSNorm tail) | 3,259 | 8.1% |
| Sparse attention (Triton `_attn_kernel` + combine) | 2,723 | 6.8% |
| MoE shared expert (GEMMs, SwiGLU, residual add) | 1,740 | 4.3% |
| Norms + activation quantization | 1,504 | 3.7% |
| MoE router + sort | 1,454 | 3.6% |
| DSpark draft (`propose_dspark` 1,292, `draft_kv` 341) | 1,633 | 4.1% |
| Attention prep (norm, rope, cache write) | 592 | 1.5% |
| Sampling and bookkeeping after the forward | 341 | 0.8% |
| lm_head + logits all-gather | 194 | 0.5% |
| Dense MLP (layers 0-3) | 158 | 0.4% |
| GPU idle inside the step | 157 | 0.4% |
| Copies and elementwise | 48 | 0.1% |

Step time by running sequences. A decode step replays the smallest captured graph that holds its
sequences (`[1..8, 16, 32, 48, 64]`), and the padding rows of that graph cost score and attention
time as real rows do; the routed MoE follows the real rows:

| Sequences | Graph | Steps | Step ms | Score ms | Select ms | `_attn_kernel` ms | Routed MoE ms | Context a sequence |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7 | 7 | 17 | 28.8 | 3.74 | 1.47 | 1.54 | 10.04 | 63.4k |
| 8 | 8 | 31 | 31.8 | 5.29 | 1.62 | 1.57 | 10.81 | 61.3k |
| 9 | 16 | 39 | 40.8 | 9.28 | 1.98 | 2.63 | 10.95 | 56.2k |
| 10 | 16 | 8 | 42.6 | 10.41 | 2.01 | 2.64 | 11.68 | 59.4k |
| 12 | 16 | 5 | 44.4 | 10.59 | 2.04 | 2.68 | 12.89 | 63.8k |
| 16 | 16 | 19 | 47.1 | 11.86 | 2.17 | 2.75 | 13.90 | 63.2k |
| 17 | 32 | 8 | 68.3 | 22.83 | 3.08 | 4.88 | 13.10 | 64.6k |

From 8 to 9 sequences the step grows 9.0 ms, 4.0 of it in the score, for one more sequence.
Steps with 11, 13 and 15 sequences (9 steps) are left out of the table.

Running sequences per decode step:

| Sequences | 1 | 2 | 3-4 | 5-8 | 9-12 | 13-16 | 17-24 | mean | steps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| profiling phase, first 376 s (`atom:decode_batch_size`) | 2.6% | 7.5% | 25.6% | 42.1% | 15.1% | 3.9% | 3.1% | 6.46 | 8,295 |
| the two profiler windows | 0 | 0 | 0 | 35.3% | 39.0% | 19.9% | 5.9% | 10.5 | 136 |

No request waited (`atom:requests_waiting` 0, 8 to 14 running) and the KV pool was 16-29% used:
at c32 the batch is set by the replay's inter-turn gaps, not by KV capacity.

## Achieved rates per decode shape

At 64 rows a step (8 sequences x 8 verify rows, 61k context each, the 8-sequence graph, no
padding); rows are the GEMM's M. The routed experts' bytes depend on how many distinct experts a
step touches, which the trace does not show, so their bandwidth is not given.

| Kernel | Shape (N x K) | us a call | us a step | TFLOPS | % of peak | TB/s | % of HBM |
|---|---|---:|---:|---:|---:|---:|---:|
| routed gemm1 (a16w4) | 384 x (1280 x 7168) MXFP4 weights, bf16 activations, top-4 | 112.7 | 7,213 | 42 | 1.7% bf16 | | |
| routed gemm2 (a16w4) | 384 x (7168 x 640) MXFP4 weights, bf16 activations, top-4 | 56.2 | 3,594 | 42 | 1.7% bf16 | | |
| index score | 16 heads x 128, fp8 index K | 77.8 | 5,292 | 206 | 4.1% fp8 | 0.85 | 11% |
| sparse attention (`_attn_kernel`) | 16 local heads x 2048 selected tokens | 23.0 | 1,567 | 82 | 1.6% fp8 | 0.23 | 3% |
| qk_proj | 4352 x 7168 MXFP8 | 20.1 | 1,364 | 199 | 4.0% fp8 | 1.66 | 21% |
| index_qk_proj | 2176 x 7168 MXFP8 | 9.9 | 671 | 202 | 4.0% fp8 | 1.71 | 21% |
| shared gate_up | 1280 x 7168 MXFP8 | 9.7 | 621 | 121 | 2.4% fp8 | 1.04 | 13% |
| o_proj | 7168 x 3072 MXFP8 | 8.2 | 559 | 343 | 6.8% fp8 | 2.90 | 36% |
| router | 384 x 7168 bf16 | 5.8 | 373 | 60 | 2.4% bf16 | 1.11 | 14% |
| shared down | 7168 x 640 MXFP8 | 4.7 | 298 | 126 | 2.5% fp8 | 1.22 | 15% |
| dense gate_up | 5120 x 7168 MXFP8 | 20.1 | 80 | 234 | 4.6% fp8 | 1.94 | 24% |
| lm_head | 25008 x 7168 bf16 | 68.4 | 68 | 335 | 13.3% bf16 | 5.30 | 66% |
| dense down | 7168 x 2560 MXFP8 | 7.4 | 29 | 319 | 6.3% fp8 | 2.72 | 34% |

In the 16-sequence graph (128 rows, whatever the real count) the MXFP8 GEMMs run other kernels and
take longer a call (qk_proj 30.0 us, o_proj 14.8 us), and the index score takes 136.5 us a call at
9 real sequences and 174.5 us at 16.

## One prefill chunk

The largest chunk in the windows: 16,384 new tokens of one sequence at 49,152 context, **822.6 ms**.
The six prefill forwards in the windows took 2.5 s of GPU time against 5.5 s for the decode steps.

| Component | ms | Share |
|---|---:|---:|
| Sparse attention (Triton `_attn_kernel`) | 258.5 | 31.4% |
| All-reduce (142 launches) and index-split all-gather (68) | 167.9 | 20.4% |
| MoE routed experts (afp4 x wfp4 `mfma_moe1` / `mfma_moe2`) | 105.8 | 12.9% |
| Attention GEMMs (q/k, index q/k, o_proj) | 91.2 | 11.1% |
| Indexer score + top-k | 73.3 | 8.9% |
| Norms + activation quantization | 47.6 | 5.8% |
| MoE shared expert (with SwiGLU) | 32.3 | 3.9% |
| MoE router (hipBLASLt bf16 GEMM) + sort | 12.8 | 1.6% |
| Attention prep | 12.0 | 1.5% |
| DSpark draft (`draft_kv`) | 11.0 | 1.3% |
| Dense MLP, copies, GPU idle | 10.1 | 1.2% |

| Kernel | Shape (N x K), M = 16384 | us a call | TFLOPS | % of peak |
|---|---|---:|---:|---:|
| sparse attention | 16 local heads x 2048 selected tokens | 3,801 | 127 | 2.5% fp8 |
| index score (2,048 of the rows, index split) | 16 heads x 128 over the causal context | 1,003 | 342 | 6.8% fp8 |
| routed gemm1 | 384 x (1280 x 7168) MXFP4, top-4 | 865 | 1,391 | 13.8% MXFP4 |
| routed gemm2 | 384 x (7168 x 640) MXFP4, top-4 | 789 | 762 | 7.6% MXFP4 |
| qk_proj | 4352 x 7168 MXFP8 | 642 | 1,592 | 31.6% fp8 |
| index_qk_proj | 2176 x 7168 MXFP8 | 359 | 1,424 | 28.3% fp8 |
| o_proj | 7168 x 3072 MXFP8 | 340 | 2,120 | 42.1% fp8 |
| shared gate_up | 1280 x 7168 MXFP8 | 231 | 1,300 | 25.8% fp8 |
| shared down | 7168 x 640 MXFP8 | 115 | 1,309 | 26.0% fp8 |
| dense gate_up | 5120 x 7168 MXFP8 | 697 | 1,726 | 34.3% fp8 |
| dense down | 7168 x 2560 MXFP8 | 299 | 2,013 | 40.0% fp8 |
| router | 384 x 7168 bf16 | 107 | 845 | 33.6% bf16 |

With `"index_split": true` each rank scores and selects 2,048 of the chunk's 16,384 rows and the
selections are all-gathered (`[8, 2048, 128]`, one a layer, about 59 us). The all-reduces of the
chunk's 16,384 x 7168 bf16 rows (235 MB) take about 1.15 ms each: 136 fused with the RMSNorm and 6
plain.

## Per-GPU HBM at startup

Rank 0, GiB (ATOM's `GB`), from the `After weight loading`, `Memory budget` and `Post-init memory`
lines of the same server:

| Item | GiB | Note |
|---|---:|---|
| HBM | 287.98 | |
| Weights after streaming load | 175.90 | estimate from the shapes: routed experts 167.3, attention 4.5, shared 0.8, draft 1.0, embedding + lm_head 0.7, router 0.3, dense 0.2 |
| Post-load processing | +1.38 | 177.28 allocated after it |
| Other torch buffers at KV sizing | +1.40 | 178.68 allocated, 178.90 reserved |
| Warmup transient (`peak_torch` - allocated) | 2.10 | `peak_torch` 180.78 |
| `non_torch` | 9.16-10.01 | across ranks; sizing charges the largest |
| CUDA graph estimate / actual pool | 0.42 / 0.64 | 12 graphs, batch 1-64 |
| Safety margin | 5.76 | |
| Outside the 0.95 budget | 14.40 | |
| **KV pool** | **76.62** | 177,260 blocks of 16 tokens (2.84M tokens), 29,008 B a token |

After graph capture rank 0 uses 267.9-269.2 GiB (93.0-93.5% of HBM): 4.4-5.7 GiB of the 0.95 target
and the 14.4 GiB outside it stay unused. #37 owns KV sizing and the warmup peak, #79 the growth from
post-load processing (1.38 GiB in this configuration, against 23.6 GB on the non-streaming load
#79 measured), and #80 the non-torch memory (9.2-10.0 GiB here, against 12.8 GB before comm-group
reuse).

## Where the c32 step can move

- The index score is 11% of HBM bandwidth at 8 sequences and a sixth to a quarter of a decode step;
  with the top-k select it is the largest gap from roofline at every width.
- Graph padding: 9 sequences replay the 16-sequence graph and 17 the 32-sequence one, and the padded
  rows cost score and attention time (8 to 9 sequences: +9.0 ms). Graph sizes between 8, 16 and 32
  sequences would cut it (#99).
- The routed experts are the largest single share; their achieved bandwidth needs the distinct
  experts a step, which this profile does not measure.
- Prefill is bound by the Triton sparse attention (2.5% of fp8 peak) and the all-reduces of 16k-row
  chunks; amd-kernels-day0's prefill attention was not in this image.

## Reproduce

Run the server and replay above inside the image under the node lock, then
`curl -X POST localhost:8300/start_profile`, wait 5 s, `curl -X POST localhost:8300/stop_profile`.
Decode tables: `python3 scripts/m31pro/kernel_roofline.py <rank_0 traces> --config <config.json>
--out <dir>`, after adding the `mfma_moe1_` / `mfma_moe2_` (routed) and `moe_sort` (sort) kernel
names to its rules; its `Cijk_` rule files the prefill router GEMM as `lm_head` (#100).
