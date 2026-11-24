# MiniMax-M3.1-Pro small-concurrency decode: kernel roofline and gap list

Every kernel of a Pro decode step on MI355X (TP8) against the floor its bytes, FLOPs and
launch allow, the gaps ranked by time lost per step, and the same for the mono v1 MoE-layer
kernel. The tool is `scripts/m31pro/kernel_roofline.py`; this page is its reading.

## Status

| input | state | where |
|---|---|---|
| c1, c4, c8, c16 DSpark k7 traces on #33's variants (agentic proxy replay, golden AL 4.0, prefix caching) | MEASURED (node 257, 2026-11-23: c1 / c4 / c8 18:38-18:56 UTC, item `30_prof17`; c16 23:14-23:24 UTC, item `72_roofline_traces`, max-num-seqs 32). ATOM `ecb1ac9a` of `/home/zhiyan/pro_opt/atom` (branch `opt/autok`, auto-k off), the tree #17's launcher serves, not `hatwu/m3.1-pro-infx` | traces `/mnt/m2m_nobackup/zhiyan/prof17/prof17/c{1,4,8}/` and `/mnt/m2m_nobackup/zhiyan/roofline/traces_roofline_c16/c16/` (`w*/rank_0/`); CSVs `/home/zhiyan/pro_opt/roofline/results/k7_c1-c16_measpeaks/` |
| decode traces at 1, 8 and 16 rows, plain decode, about 1100 tokens a sequence, mono off and on | MEASURED (node 257, 2026-11-23 12:13 UTC, run `v1model_1123T1213`) | traces `/mnt/m2m_nobackup/zhiyan/mono_v1/prof_v1model_1123T1213_{off,on}/rank_0/`; CSVs `/home/zhiyan/pro_opt/roofline/results/v1model_1123T1213_{off,on}_measpeaks/` |
| HBM read bandwidth, empty-kernel time, bf16 / fp8 / MXFP4 GEMM peaks, one GPU | MEASURED (node 257, 2026-11-23 22:58 UTC, item `70_roofline_peaks`) | `/home/zhiyan/pro_opt/roofline/results/peaks_1123T2258.peaks.json`; the analysis reads `peaks_1123T2258_trace_tmin.json` (the same without the empty-kernel time, see the cautions below) |
| TP8 all-reduce latency, 1 to 128 rows of 7168 bf16, Lamport | MEASURED (node 257, 2026-11-23 07:59 UTC) | `/home/zhiyan/pro_opt/mono/spike_ar/results/w1_1123T0759_bench_ar_B.txt`, as a table in the peaks JSON |
| mono v1 per-stage time at 1, 8, 16 rows (timeline build, standalone, near-uniform routing) | MEASURED (node 257, 2026-11-23 22:58 UTC, item `71_roofline_mono_tl`; worktree `pro_opt/wt/mono-v1` at `c467bcf9`) | `/home/zhiyan/pro_opt/roofline/results/mono_tl_1123T2258.mono/mono_timeline.json` |
| mono v1 kernel time at 1, 8, 16 rows, standalone | MEASURED (node 257, `v1perf_1123T1153`) | `/home/zhiyan/pro_opt/mono/v1/results/v1perf_1123T1153.log` |

## Measured peaks (one MI355X)

| | measured | spec sheet |
|---|---:|---:|
| HBM read, 4 GB (Triton grid-stride read, CUDA graph) | 6.23 TB/s | 8.0 TB/s |
| HBM read at 1 GB / 128 MB / 32 MB / 16 MB | 6.10 / 5.37 / 3.46 / 2.29 TB/s | |
| bf16 GEMM 8192^3 (`torch.matmul`) | 1287 TFLOPS | 2516.6 |
| fp8 e4m3 GEMM 8192^3 (`torch._scaled_mm`) | 2664 TFLOPS | 5033.2 |
| MXFP4 GEMM 8192^3 (aiter `gemm_a4w4`) | 4412 TFLOPS | 10066.3 |
| empty Triton kernel in a graph: profiler duration / graph slope | 4.08 us / 1.71 us | |
| TP8 all-reduce, Lamport, 1 / 8 / 16 / 32 / 64 / 128 rows | 4.2 / 5.9 / 6.3 / 7.4 / 9.7 / 14.2 us | |

The analysis uses the measured values. Two cautions:

- A single read of 16 to 32 MB (one layer's qk_proj, o_proj or shared-expert weights) reaches
  only 2.3 to 3.5 TB/s in this test, against 6.23 at 4 GB (a fit over 256 MB to 4 GB gives 5.6 us
  plus 6.26 TB/s). The floor is the classical roofline, `max(t_min, bytes / BW)`, so for those
  kernels it is below what one launch can reach and their gaps are upper bounds.
- The empty kernel's profiler duration (4.08 us) is longer than real kernels in the decode
  traces (0.1th percentile 2.44 us; `_masked_embedding_kernel` 2.55 to 2.76 us), so it is not
  used as t_min. Item 72's own run of the analysis used it and reads 16 to 38% within 80% of
  roofline without the routed experts; the numbers below take t_min from the traces.

## How the floor is computed

For each kernel launch, `floor = max(t_min, bytes / BW, FLOPs / F)`; for an all-reduce,
`floor = AR(rows)` from the measured table (linear between measured rows). Percent of roofline
is floor over measured time; the gap is time minus floor.

- **bytes** is the compulsory HBM traffic at Pro's shapes on one rank: every weight once,
  activations in and out once, each sequence's KV and index pages once. The shapes come from
  the checkpoint's `config.json` and the forward's rows, sequences and context (`sk` in the run
  label when the server ran with `ATOM_ENABLE_DETAILED_ANNOTATION=1`, else `--ctx`).
- **Routed experts** read the weights of the distinct experts the step picks. The trace does
  not say how many that is, so the model takes the expected count under uniform routing (4 at
  1 row, 30.9 at 8, 59.2 at 16). Real routing touches fewer, so the routed percent is an upper
  bound. This is why every share below is given twice: without the routed experts (and the mono
  layer, whose floor uses the same count), which the measurement supports, and with them, an
  upper bound.
- **t_min** is the 0.1th-percentile kernel duration of the traces (2.44 us in the k7 traces,
  2.88 / 2.64 us in the plain-decode ones), not the shortest kernel, so one outlier cannot set it;
  a kernel can still run below it and read above 100%.
- **F** is the measured peak of the kernel's math. No decode kernel at these widths comes near
  it, so the bound is memory, latency or communication.
- The MXFP8 GEMMs share kernel names across roles; the tool labels them by order between two
  indexer `_prep_kernel` launches (o_proj, then the FFN, then the next layer's qk_proj and
  index_qk_proj).
- The draft's GEMMs are one group against the draft's weight bytes. Their FLOPs are taken as 2
  a weight byte a row, right for the MXFP8 weights and twice the real count for the bf16 `fc`
  and `lm_head`; it would matter only if a draft group turned compute-bound.
- **The DSpark Markov head** (`_markov_stage1`, one launch a block position, 7 a k7 draft step)
  reads all of `markov_w2` (200064 x 256 bf16, 102.4 MB, replicated on every rank) plus the
  sequences' all-gathered base logits: about 103 MB a launch at one sequence. PR #86 shards the
  head by vocabulary, which cuts what each rank reads 8x; the kernel name stays the same, so a
  trace of that build needs the model changed to the shard.
- The all-reduce table is Lamport's. #33's variants send all-reduces of more than 16 rows to
  another route (`ar_lamport_rows: 16`), so at 24 rows the floor is from a path the step did
  not take (`allreduce_fusion_kernel_2stage` at c16).

## Decode step at c1, c4, c8 and c16, DSpark k7 (MEASURED, measured peaks)

One step is a target verify of 8 rows a sequence plus the draft (`draft_kv`, `propose_dspark`).
In the profiled windows of the agentic replay, the pure-decode forwards hold 1 sequence at c1,
1 or 2 at c4 and c8, and 1 to 3 at c16; the other sessions are between turns or prefilling.
Prefill-only forwards are 2 to 11% of the forward time and are left out. Per concurrency,
weighted by forwards (`steps.csv` has each width):

| conc | forwards at 8 / 16 / 24 rows | context a sequence | step us | within 80% of roofline: without the routed experts / all kernels (upper bound) | latency-bound launches / us | three largest gaps (us a step) |
|---:|---|---:|---:|---:|---:|---|
| c1 | 388 / - / - | 55.9k | 13542 | 8.4% / 29.1% | 1277 / 7481 | top-k selection `_select_split_kernel` 886; index scores `_score_qtile_kernel` 472; `_prep_kernel` 414 |
| c4 | 381 / 82 / - | 71.0k | 14254 | 8.3% / 29.6% | 1265 / 7683 | top-k selection 994; index scores 736; `_prep_kernel` 415 |
| c8 | 228 / 264 / - | 40.8k | 15009 | 8.1% / 33.7% | 1277 / 7821 | top-k selection 823; index scores 754; `_attn_kernel` 452 |
| c16 | 73 / 99 / 110 | 42.8k | 16197 | 1.2% / 29.2% | 1200 / 7372 | top-k selection 905; index scores 786; `_prep_kernel` 413 |

Kernels with a model cover 99.7 to 99.8% of the kernel time in every bucket.

What it says:

1. **Launch count is the lever.** 1140 to 1280 latency-bound launches (their bytes would
   stream in less than t_min) take 6.9 to 8.5 ms of a 13.1 to 18.5 ms step. The modelled floor
   of the whole step is 8.0 ms at 8 rows and 12.8 ms at 24.
2. **At agentic context the indexer is the largest gap.** At 40k to 71k tokens the top-k
   selection takes 15 us a layer at 8 rows (1052 us a step at c1) and the index scores 9 us (638
   us), both against the 2.4 us latency floor: the selection's bytes (the scores of 3500 blocks
   a row) and the scores' (7 MB of index keys a layer) stream in about 1 us. Together they are 1.4
   to 1.7 ms a step above their floors.
3. **The routed experts read 99% of roofline at 8 rows under uniform routing**: 6.2 TB/s, the
   measured peak, if a step touched 30.9 distinct experts. It is the upper bound; how far they
   really are waits for the distinct-expert count (brief B4).
4. **The draft is 7% of the step** (961 us with a model at c1). Its largest kernel, the Markov
   head, runs at 94% of roofline (122 us a step against 116): it is near its floor, and what
   would shrink it is reading less (the vocabulary-sharded head of PR #86).
5. **The within-80% share is a threshold count.** At c16 it falls to 1.2% because the
   all-reduces, 1130 us a step at 8 rows against 937 at c8 for the same row count, drop from 88%
   to 73% of their floor; nothing else moved across the line.

## Plain decode at short context, mono off and on (MEASURED, measured peaks)

From #89's traces (`v1model_1123T1213`), about 1100 tokens a sequence, no draft:

| rows | mono | step us | within 80%: without routed / all (upper bound) | latency-bound launches / us | three largest gaps (us a step) |
|---:|---|---:|---:|---:|---|
| 1 | off | 9262 | 0.7% / 0.7% | 1273 / 6410 | routed gemm1 771; `_prep_kernel` 341; `_attn_kernel` 322 |
| 8 | off | 11052 | 9.6% / 34.2% | 1209 / 6212 | `_prep_kernel` 379; `_attn_kernel` 348; index_qk_proj 305 |
| 16 | off | 12534 | 8.9% / 39.5% | 1209 / 6459 | `_attn_kernel` 481; `_prep_kernel` 383; index_qk_proj 322 |
| 1 | on | 6141 | 6.8% / 6.8% | 508 / 2783 | mono layer 1335; `_prep_kernel` 341; `_attn_kernel` 333 |
| 8 | on | 8480 | 6.5% / 6.5% | 508 / 2893 | mono layer 1209; `_prep_kernel` 373; `_attn_kernel` 341 |
| 16 | on | 9906 | 5.9% / 60.7% | 508 / 3112 | `_attn_kernel` 478; `_prep_kernel` 382; index_qk_proj 316 |

Mono v1 cuts the launches from about 1620 to 730 a step (latency-bound: 1209-1273 to 508) and
shortens the step by 3.1 / 2.6 / 2.6 ms at 1 / 8 / 16 rows.

## Gap list

Ranked by gap at c1, DSpark k7, 8 rows, 55.9k tokens a sequence, measured peaks
(`k7_c1-c16_measpeaks`). "Saving" is the gap itself: the most the item can save if its kernels
reach their floor (ESTIMATED, an upper bound); items that remove launches can save up to the
whole launch time instead.

| # | item | kernels | c1 k7: us / floor / gap a step | saving a step (ESTIMATED) | notes |
|---:|---|---|---|---|---|
| 1 | launch count | every small kernel | 1277 latency-bound launches, 7481 us | mono v1 MEASURED on plain decode: -3.1 / -2.6 / -2.6 ms at 1 / 8 / 16 rows | items 2 to 10 are where the remaining launches are |
| 2 | indexer top-k selection | `_select_split_kernel` | 1052 / 166 / 886 | up to 0.89 ms; 0.82 to 0.99 ms at c4 to c16 | 15.5 us a layer for the scores of 3500 blocks a row; grows with context |
| 3 | router path | hgemm_bf16, topk_gating, 2 x moe_sorting | 1256 / 624 / 631 | up to 0.63 ms, up to 1.26 ms if fused into one launch | four launches a layer for 5.5 MB of weights; mono v1 does it in-kernel |
| 4 | shared expert | gate_up, SwiGLU, down, add | 1212 / 637 / 576 | up to 0.58 ms | four launches for 14 MB; mono v1 runs it in-kernel |
| 5 | norms and quantization | add_rmsnorm_quant x 2, fused_qk_rmsnorm_group_quant, quantize_fp8 | 1220 / 659 / 561 | up to 0.56 ms, up to 1.22 ms when fused away | each a separate launch of about 4.5 us for under 0.3 MB |
| 6 | qk_proj + index_qk_proj | two MXFP8 GEMMs a layer, 48 MB together | 1076 / 529 / 547 | up to 0.55 ms | index_qk_proj runs at 2.2 TB/s, qk_proj at 3.9; both read the same K = 7168 |
| 7 | sparse attention | `_attn_kernel` + `_combine_kernel` | 846 / 332 / 514 | up to 0.51 ms | 128 selected blocks a row whatever the context |
| 8 | indexer scores | `_score_qtile_kernel` | 638 / 166 / 472 | up to 0.47 ms; 0.74 to 0.79 ms at c4 to c16 | 9.4 us a layer for 7 MB of fp8 index keys; PR #87 changes the scorer |
| 9 | attention prep | `_prep_kernel` | 580 / 166 / 414 | up to 0.41 ms | 8.5 us for well under 1 MB: latency-bound, a fusion candidate (into the qk GEMM's epilogue) |
| 10 | DSpark draft | draft_kv, propose_dspark | 961 / 616 / 344 | up to 0.34 ms | the Markov head is 122 us of it at 94% (PR #86 shards it); the GEMM groups run at 59 to 68% |
| 11 | o_proj | MXFP8 GEMM, 22.7 MB | 368 / 249 / 119 | up to 0.12 ms | 4.2 TB/s |
| 12 | all-reduce | Lamport one-shot / two-shot, 141 a step | 950 / 838 / 112 | up to 0.11 ms; 0.34 ms at c16 | the floor is the standalone latency; the excess in the model is rank skew |
| 13 | routed experts | gemm1_a16w4, gemm2_a16w4 | 2807 / 2784 / 23 | unknown until B4 | at the floor only under uniform routing; at 1 row of plain decode gemm1 runs at 24% |
| 14 | routed intermediate pad 640 -> 768 | gemm1 / gemm2 | 17% of the routed bytes | about 0.47 ms at c1 if the routed kernels stream at their current rate | PR #88 removes the pad (rounding-level change); 13-15% of the routed time at large M (yuhzhu `opt/best.md` 8.1) |

## Mono v1 (#55, #70): the MoE layer in one kernel

What it runs on one rank a MoE layer: o_proj (MXFP8), AR #1 + residual + Gemma RMSNorm +
MXFP8 x + the lookahead norm of layer L+2, the router split over the ranks (48 experts a rank,
logits all-gathered), the routed experts in W4A4 without the 640 -> 768 pad, the shared expert
in MXFP8. AR #2 stays on production's Lamport path.

Floors from the shapes and the measured peaks (`--mono-floor 1,8,16`,
`results/mono_floor_measured.csv`); times from item 71's timeline build (every CTA stamps the
kernel's points; a stage is the latest CTA past its end point minus the latest past its start,
median of 31 launches, slowest rank). Its random weights route near-uniformly, so the routed
floor applies as computed. The timeline build's totals are within 5% of the plain kernel's
(33.7 / 74.1 / 114.3 us, `v1perf_1123T1153`).

| stage (timeline points) | floor 1 / 8 / 16 rows, us | measured 1 / 8 / 16 rows, us | % of floor |
|---|---|---|---|
| o_proj: core quant, o GEMM (22.7 MB), push (0 -> 2) | 3.65 / 3.67 / 3.69 | 3.71 / 7.21 / 9.99 | 98 / 51 / 37 |
| AR #1 + norm + MXFP8 + lookahead norm (2 -> 5) | 4.20 / 5.90 / 6.30 (AR table) | 4.38 / 7.97 / 12.67 | 96 / 74 / 50 |
| router (0.7-0.9 MB bf16 a rank) | 0.11 / 0.13 / 0.15 | router CTAs' median 9.88 / 17.76 / 25.28, concurrent with the shared gate_up | - |
| shared MXFP8 (14.2 MB) | 2.28 / 2.31 / 2.33 | gate_up CTAs' median 8.60 / 13.56 / 19.00; its down runs with the routed down | - |
| routed W4A4 (U x 7.31 MB) | 4.69 / 36.21 / 69.52 | union + x MXFP4 2.28 / 2.96 / 3.15; gate_up 7.00 / 28.26 / 48.38; down + shared down 8.60 / 17.70 / 29.35 | - |
| router + shared + routed together (5 -> 9) | 7.08 / 38.65 / 72.00 | 23.92 / 56.59 / 89.01 | 30 / 68 / 81 |
| AR #2, outside the kernel (mean Lamport call in the mono-on trace) | 4.20 / 5.90 / 6.30 | 4.84 / 6.66 / 7.08 | 87 / 89 / 89 |
| **kernel (all but AR #2)** | **14.93 / 48.22 / 81.99** | **32.05 / 71.81 / 111.71** | **47 / 67 / 73** |

The router, shared and routed stages overlap inside 5 -> 9 (different CTAs), so the timeline
splits that span by point, not by stage; the CTA medians are not additive.

Where the time above the floor is:

- **1 row: the FFN span**, 16.8 of the 17.1 us above the floor. The routed gate_up (7.0 us for
  four experts, 19 MB with gate and up) and the down + shared down (8.6 us) each take a few
  times their streaming time: at this size the stages are latency-bound, and the router and
  shared gate_up CTAs (8.6 to 9.9 us) are the critical path into the routed stage.
- **16 rows: o_proj and AR #1** grow from 8 to 23 us with the rows while their floors stay at 10:
  12.7 us above the floor, against 17 in the FFN span.
- In the model (`v1model_1123T1213_on`, plain decode, real routing) the kernel takes 35.8 /
  67.1 / 84.8 us a MoE layer, against production's FFN segment of 82.7 / 106.1 / 122.9.

### Methods that apply, with sources

| method | what it would change in v1 | source | MI355X |
|---|---|---|---|
| prefetch the next task's weights while the current task computes | routed stage: the README lists "no prefetch of the next task's weights" | persistent megakernels: Hazy Research, "Look Ma, No Bubbles!" (hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles); Mirage Persistent Kernel, arXiv:2512.22219 (cross-operator software pipelining) | applies: 256 CUs, 160 KB LDS a CU; v1 already holds every CTA resident |
| balance the routed down over all CTAs | 448 down tasks over 256 CTAs leave a 1.75-wave tail | Stream-K (Osama et al., PPoPP 2023, arXiv:2301.03598): split the MAC iterations evenly over a fixed grid | applies; aiter's FlyDSL GEMMs already split K (the `sk4` / `sk7` kernels in the trace) |
| plan the expert schedule in-kernel from the routing | today round-robin tasks | aiter `aiter/ops/flydsl/mega_moe_tp.py` (MegaMoE TP: one persistent launch a TP MoE layer, dynamic schedule up to 256 tokens, AR or AG/RS comm in-kernel) | gfx950 kernel; Pro's 640 intermediate needs tuned rows (`configs/mega_moe_tp_tuned.csv`) |
| in-kernel all-reduce with the norm and quant fused | v1 does this for AR #1; AR #2 is still a separate Lamport launch | aiter `aiter/dist/device_communicators/custom_all_reduce.py` (Lamport one-shot / two-shot with fused RMSNorm); FlashDMoE, arXiv:2506.04667 (device-initiated communication inside one MoE kernel) | applies: peer memory over XGMI via hipIpc, as v1 and MegaMoE TP use |
| move the layer boundary to swallow AR #2 | one launch fewer a layer, 4.8 to 7.1 us a layer today | MPK and Hazy Research (whole-layer kernels) | applies; needs the attention side in the same kernel or a peer handshake between kernels |
| skip the zero pad | v1 already reads 640, production 768 | yuhzhu `opt/best.md` 8.1 (p640 13-15% faster at 256-1024 rows) | applies |
| small-M fused MoE paths | the production comparison point | aiter `aiter/fused_moe.py` (the a16w4 "port" kernels in the trace) | in use |

## Ready-to-file briefs (not filed)

Check ownership first for anything touching the indexer: #3 and #12 / #72 own it.

**B1. Mono v1: the FFN span at small rows, o_proj and AR #1 at 16.** Goal: the standalone
mono v1 kernel at 25 / 60 / 100 us at 1 / 8 / 16 rows (60 / 80 / 82% of its floor; 32.1 /
71.8 / 111.7 today). File set: `atom/models/minimax_m3/pro_mono/kernels/` (the FFN and the
o_proj / AR #1 stages), their tests. Exit: `perf_layer.py` at 1 / 8 / 16 rows meets the target,
the 8-GPU layer test passes, the timeline shows which stages shrank. Estimate: 200 code, 80
test lines. GPU: 30 min.

**B2. qk_proj and index_qk_proj as one launch.** Goal: one MXFP8 GEMM launch a layer for both
projections (48 MB) at 5 TB/s or more: about 10 us a layer instead of 15.8 at c1 k7, 0.4 ms a
step (ESTIMATED). File set: the aiter-M3 GEMM (a grouped two-A call), ATOM `pro.py`'s caller.
Exit: a kernel microbench at 1 / 8 / 16 / 32 / 128 rows against the two launches, numerics
bit-equal, the trace showing 68 launches fewer. Estimate: 150 code, 80 test lines (aiter side
first). GPU: 30 min.

**B3. Attention prep into the qk GEMM epilogue, combine into the attention.** Goal: 2 launches
fewer a layer (136 a step; the two kernels take 0.89 ms a step at c1 k7 today). File set:
`atom/model_ops/minimax_m3/pro_kernels.py`. Exit: the per-layer attention-side time on a c1 k7
trace below today's, the 32k / 64k retrieval check 40 / 40, GSM8K. Estimate: 250 code, 100
test lines. GPU: 1 h.

**B4. Count the distinct experts a verify step touches.** Goal: replace the uniform-routing
assumption in the routed floor with the measured count at c1 to c16 (k7), which decides gap
item 13. File set: a profile-only counter next to the router (off by default) and
`kernel_roofline.py`'s routed model reading it. Exit: the routed percent below 100% at every
width; the count per width in the PR. Estimate: 60 code, 40 test lines. GPU: 20 min.

**B5. Top-k block selection at agentic context (check #12 / #72 first).** Goal:
`_select_split_kernel` at 56k tokens and 8 rows from 15.5 to 6 us a layer, 0.6 ms a step at c1
k7 (ESTIMATED). File set: the selection kernel in `atom/model_ops/minimax_m3/pro_kernels.py`.
Exit: the same tables bit for bit, a kernel microbench at 16k / 64k / 128k context and 8 / 16 /
24 rows, the c1 k7 trace showing the drop. Estimate: 150 code, 80 test lines. GPU: 30 min.

## Reproduce

```bash
# traces of one rank under c<N>/ directories, the checkpoint's config, measured peaks
python3 scripts/m31pro/kernel_roofline.py /mnt/m2m_nobackup/zhiyan/prof17/prof17 \
  /mnt/m2m_nobackup/zhiyan/roofline/traces_roofline_c16 \
  --config /mnt/m2m_nobackup/zhiyan/models/m3-pro/config.json \
  --peaks /home/zhiyan/pro_opt/roofline/results/peaks_1123T2258_trace_tmin.json --out <dir>
# traces without sk labels (the plain-decode runs): add --ctx 1100
# the mono stage floors
python3 scripts/m31pro/kernel_roofline.py --config <config.json> --peaks <peaks> --mono-floor 1,8,16
```

The GPU items (`/home/zhiyan/pro_opt/roofline/roofline_item.sh peaks | mono_tl | traces |
analyze`) take the node lock and write to `/home/zhiyan/pro_opt/roofline/results/` and
`/mnt/m2m_nobackup/zhiyan/roofline/`.
