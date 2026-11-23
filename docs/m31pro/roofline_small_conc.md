# MiniMax-M3.1-Pro small-concurrency decode: kernel roofline and gap list

Every kernel of a Pro decode step on MI355X (TP8) against the floor its bytes, FLOPs and
launch allow, the gaps ranked by time lost per step, and the same for the mono v1 MoE-layer
kernel. The tool is `scripts/m31pro/kernel_roofline.py`; this page is its reading.

## Status

| input | state | where |
|---|---|---|
| decode traces at 1, 8 and 16 rows, plain decode, context about 1100 tokens a sequence, mono off and mono on | MEASURED (node 257, 2026-11-23 12:13 UTC run `v1model_1123T1213`) | traces `/mnt/m2m_nobackup/zhiyan/mono_v1/prof_v1model_1123T1213_{off,on}/rank_0/`; CSVs `/mnt/m2m_nobackup/zhiyan/roofline/v1model_1123T1213_{off,on}/` (copies in `/home/zhiyan/pro_opt/roofline/results/`) |
| TP8 all-reduce latency, 1 to 128 rows of 7168 bf16 | MEASURED (node 257, 2026-11-23 07:59 UTC) | `/home/zhiyan/pro_opt/mono/spike_ar/results/w1_1123T0759_bench_ar_B.txt`, as a table in `/home/zhiyan/pro_opt/roofline/peaks_ar_only.json` |
| mono v1 kernel time at 1, 8 and 16 rows, standalone, uniform routing | MEASURED (node 257, `v1perf_1123T1153`) | `/home/zhiyan/pro_opt/mono/v1/results/v1perf_1123T1153.log` |
| HBM read bandwidth, empty-kernel time, bf16 / fp8 / MXFP4 GEMM peaks | NOT RUN, queued as `q257/pending/70_roofline_peaks.sh` | spec-sheet values stand in (below) |
| mono v1 per-stage time (timeline build) | NOT RUN, queued as `71_roofline_mono_tl.sh` | |
| c1, c4, c8, c16 DSpark k7 traces on #33's config | NOT RUN: c1 / c4 / c8 come from `30_prof17.sh` (#17), c16 from `72_roofline_traces.sh`, which also re-runs this analysis | |

Until 70 and 72 land, every percent below is against the spec-sheet bandwidth (8.0 TB/s), so
it understates how close a memory-bound kernel is to what the GPU can reach; and the measured
widths are plain decode, not DSpark verify. The verify width of c1 at k7 is 8 rows and of c2 is
16 rows, so the 8- and 16-row columns show those kernels at those row counts, but not the
indexer and attention at agentic context lengths (100k tokens and more), nor the draft.

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
  1 row, 30.9 at 8, 59.2 at 16). Real text routes to fewer, so the routed percent is an upper
  bound: above 100% means the step touched fewer experts than uniform routing would.
- **t_min** is the shortest kernel the trace holds (2.88 us mono off, 2.64 us mono on; a memcpy
  or the embedding). Item 70 measures an empty kernel instead.
- **F** is the dense spec peak of the kernel's math: bf16 2516.6, fp8 5033.2, fp4 10066.3
  TFLOPS. At 16 rows no decode kernel comes near it, so the bound is memory or latency.
- The MXFP8 GEMMs share kernel names across roles; the tool labels them by order between two
  indexer `_prep_kernel` launches (o_proj, then the FFN, then the next layer's qk_proj and
  index_qk_proj).
- The draft's GEMMs are one group against the draft's weight bytes (traces 72 and 30 hold it;
  the plain-decode traces here have none).

## Where a decode step goes (MEASURED, mono off)

From `steps.csv`; "latency-bound" counts the launches whose floor is t_min, that is whose
bytes would stream in less than one launch's time.

| rows | step us | floor us | within 80% of roofline | latency-bound launches / us | three largest gaps (us a step) |
|---:|---:|---:|---:|---:|---|
| 1 | 9262 | 4977 | 0.1% | 1345 / 6781 | routed gemm1 824; `_prep_kernel` 341; `_attn_kernel` 322 |
| 8 | 11052 | 7009 | 17.2% | 1281 / 6601 | `_prep_kernel` 379; routed gemm1 367; `_attn_kernel` 348 |
| 16 | 12534 | 9060 | 39.0% | 1281 / 6854 | `_attn_kernel` 481; `_prep_kernel` 383; qk_proj 345 |

Kernels with a bytes / FLOPs model cover 99.8 to 99.9% of the kernel time at every width (the
rest is the logits all-gather, which has no measured latency, and the lm_head-side NCCL
launch). Launch gaps inside the forward are 42 to 44 us a step and idle between forwards 14 us:
the graph replays back to back.

What the table says:

1. **About 1620 launches a step, 1281 to 1345 of them latency-bound.** A forward runs about
   24 launches a layer; the latency-bound ones, at 4 to 8 us each, cost 6.6 to 6.9 ms of a 9.3
   to 12.5 ms step, while their bytes would stream in a fraction of that. At t_min each they
   would still take 3.7 ms. This, not bandwidth, is what limits small-concurrency decode.
2. **The memory-bound part is the routed experts and the big projections.** At 8 rows the
   routed gemm1 + gemm2 run at 80% (uniform-routing bound); at 1 row gemm1 runs at 19%
   (15.8 us a layer for 23 MB). qk_proj (32 MB) and index_qk_proj (16 MB) run at 3.5 to 4 TB/s.
3. **Mono v1 cuts the launches from about 1620 to 730 a step** (latency-bound: 1281 to 512)
   and shortens the step by 3.1 / 2.6 / 2.6 ms at 1 / 8 / 16 rows (MEASURED; mono on: 6141 /
   8480 / 9906 us a step).

Module view at 8 rows (the c1 k7 verify width), mono off:

| module | us a step | share | floor us | % roofline |
|---|---:|---:|---:|---:|
| moe.routed (gemm1, gemm2) | 2711 | 24.5% | 2168 | 80.0 (bound) |
| moe.router (router GEMM, gating, 2 x sorting) | 1279 | 11.6% | 737 | 57.6 |
| norm_quant (add_rmsnorm_quant x 2, lookahead norm, quantize_fp8) | 1166 | 10.5% | 777 | 66.7 |
| moe.shared (gate_up, SwiGLU, down, add) | 1158 | 10.5% | 737 | 63.6 |
| attn.qk_proj (qk_proj, index_qk_proj) | 1077 | 9.7% | 470 | 43.7 |
| allreduce | 987 | 8.9% | 808 | 81.9 |
| attn.sparse_attn (`_attn_kernel`, `_combine_kernel`) | 841 | 7.6% | 392 | 46.5 |
| attn.prep | 575 | 5.2% | 196 | 34.0 |
| attn.o_proj | 369 | 3.3% | 196 | 53.0 |
| attn.index_score | 324 | 2.9% | 196 | 60.4 |
| attn.index_topk | 288 | 2.6% | 196 | 68.0 |
| lm_head, dense MLP, sampling, misc, the logits all-reduce | 218 | 2.0% | 136 | 62 |

The per-kernel tables for every width and both arms are in `summary.md` and `kernels.csv` next
to the CSVs named in Status.

## Gap list

Ranked by gap at 8 rows (mono off), the c1 k7 verify width. "Saving" is the gap itself: the
most the item can save if the kernels reach their floor (ESTIMATED, an upper bound); items
that remove launches save up to the whole launch time instead. 1 and 16 rows are given where
the order differs.

| # | item | kernels | 8 rows: us / floor / gap | saving a step (ESTIMATED) | notes |
|---:|---|---|---|---|---|
| 1 | launch count: about 1620 launches a step, 1281 latency-bound | every small kernel | 6601 us in latency-bound launches | mono v1 MEASURED: -2.57 ms at 8 rows (-3.12 at 1, -2.63 at 16) | the lever at c1 to c16; items 2 to 9 are where the remaining launches are |
| 2 | routed experts | gemm1_a16w4, gemm2_a16w4 | 2711 / 2168 / 542 | up to 0.54 ms at 8 rows, 1.04 ms at 1 row | 1 row: gemm1 at 19% (15.8 us for 23 MB). The weights carry the 640 -> 768 pad: 17% of the routed bytes |
| 3 | qk_proj + index_qk_proj | two MXFP8 GEMMs a layer, 48 MB together | 1077 / 470 / 607 | up to 0.61 ms | both read the same K = 7168; one launch would also drop 68 launches |
| 4 | router path | hgemm_bf16, topk_gating, 2 x moe_sorting | 1279 / 737 / 542 | up to 0.54 ms, or 1.28 ms if fused into one launch | four launches a layer for 5.5 MB of weights; mono v1 does it in-kernel |
| 5 | sparse attention | `_attn_kernel` + `_combine_kernel` | 841 / 392 / 450 | up to 0.45 ms; 0.58 ms at 16 rows | at about 1100 tokens of context; long agentic contexts wait for traces 30 / 72 |
| 6 | shared expert | gate_up, SwiGLU, down, add | 1175 / 750 / 425 | up to 0.43 ms | four launches for 14 MB; mono v1 runs it in-kernel |
| 7 | attention prep | `_prep_kernel` | 575 / 196 / 379 | up to 0.38 ms | 8.5 us for well under 1 MB at 8 rows: latency-bound, a fusion candidate (into the qk GEMM's epilogue) |
| 8 | norms and quantization | add_rmsnorm_quant x 2, fused_qk_rmsnorm_group_quant, quantize_fp8 | 1166 / 777 / 388 | up to 0.39 ms, up to 1.17 ms when fused away | each a separate launch of about 4.3 us for under 0.3 MB |
| 9 | all-reduce | Lamport one-shot / two-shot, 138 a step | 1000 / 814 / 185 | up to 0.19 ms | the floor is the standalone latency; the excess in the model is rank skew |
| 10 | o_proj | MXFP8 GEMM, 22.7 MB | 369 / 196 / 174 | up to 0.17 ms | 5.4 us vs 2.84 us at 8 TB/s |
| 11 | indexer scoring and top-k | `_score_tile_kernel`, `_select_split_kernel` | 612 / 392 / 221 | up to 0.22 ms at 1100 tokens | grows with context; the agentic number needs traces with `sk` (30 / 72). Open PR #87 changes the scorer |
| 12 | routed intermediate pad 640 -> 768 | gemm1 / gemm2 | 17% of the routed bytes | 13-15% of the routed time at large M (yuhzhu `opt/best.md` 8.1), about 0.35 ms at 8 rows if it holds at small M | open PR #88 removes the pad (rounding-level change) |
| 13 | DSpark draft at k7 | draft_kv, propose_dspark | no trace yet | - | the tool models it (weights-only group); traces 30 / 72. Open PR #86 measures the draft cost |

## Mono v1 (#55, #70): the MoE layer in one kernel

What it runs on one rank a MoE layer: o_proj (MXFP8), AR #1 + residual + Gemma RMSNorm +
MXFP8 x + the lookahead norm of layer L+2, the router split over the ranks (48 experts a rank,
logits all-gathered), the routed experts in W4A4 without the 640 -> 768 pad, the shared expert
in MXFP8. AR #2 stays on production's Lamport path.

### Floor by stage (ESTIMATED from the shapes, `--mono-floor 1,8,16`, 8.0 TB/s spec)

| stage | 1 row | 8 rows | 16 rows | bound |
|---|---:|---:|---:|---|
| o_proj (22.7 MB MXFP8) | 2.84 | 2.86 | 2.87 | memory |
| AR #1 + norm + quant (measured AR latency) | 4.20 | 5.90 | 6.30 | communication |
| router (0.7-0.9 MB bf16 a rank) | 0.09 | 0.10 | 0.11 | memory |
| routed W4A4 (U x 7.31 MB, U = 4 / 30.9 / 59.2) | 3.66 | 28.20 | 54.14 | memory |
| shared MXFP8 (14.2 MB) | 1.78 | 1.80 | 1.82 | memory |
| AR #2 (outside the kernel) | 4.20 | 5.90 | 6.30 | communication |
| **kernel floor (all but AR #2)** | **12.6** | **38.9** | **65.2** | |

### Measured

| | 1 row | 8 rows | 16 rows |
|---|---:|---:|---:|
| standalone kernel, uniform routing (`v1perf_1123T1153`, slope) | 33.7 | 74.1 | 114.3 |
| % of floor | 37% | 52% | 57% |
| in the model, plain decode (`v1model_1123T1213_on`, mean a MoE layer) | 35.8 | 67.1 | 84.8 |
| production's FFN segment a MoE layer, same run (`v1perf_1123T1153_off`) | 82.7 | 106.1 | 122.9 |
| per-stage times | NOT RUN (71) | NOT RUN (71) | NOT RUN (71) |

In the model the routing is real and touches fewer experts than uniform, so the in-model times
sit below the standalone ones at 8 and 16 rows and cannot be put against the uniform floor.
Against the standalone run, the kernel spends 21 us above its floor at 1 row, 35 us at 8 and
49 us at 16, and the floor's growth from 1 to 16 rows (52 us) is the routed weights. So the
routed stage is where the time above the floor should be; 71's timeline will say how it splits
between union, routed gate_up and routed down + shared down.

### Methods that apply, with sources

| method | what it would change in v1 | source | MI355X |
|---|---|---|---|
| prefetch the next task's weights while the current task computes | routed stage: the README lists "no prefetch of the next task's weights" | persistent megakernels: Hazy Research, "Look Ma, No Bubbles!" (hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles); Mirage Persistent Kernel, arXiv:2512.22219 (cross-operator software pipelining) | applies: 256 CUs, 160 KB LDS a CU; v1 already holds every CTA resident |
| balance the routed down over all CTAs | 448 down tasks over 256 CTAs leave a 1.75-wave tail | Stream-K (Osama et al., PPoPP 2023, arXiv:2301.03598): split the MAC iterations evenly over a fixed grid | applies; aiter's FlyDSL GEMMs already split K (the `sk4` / `sk7` kernels in the trace) |
| plan the expert schedule in-kernel from the routing | today round-robin tasks | aiter `aiter/ops/flydsl/mega_moe_tp.py` (MegaMoE TP: one persistent launch a TP MoE layer, dynamic schedule up to 256 tokens, AR or AG/RS comm in-kernel) | gfx950 kernel; Pro's 640 intermediate needs tuned rows (`configs/mega_moe_tp_tuned.csv`) |
| in-kernel all-reduce with the norm and quant fused | v1 does this for AR #1; AR #2 is still a separate Lamport launch | aiter `aiter/dist/device_communicators/custom_all_reduce.py` (Lamport one-shot / two-shot with fused RMSNorm); FlashDMoE, arXiv:2506.04667 (device-initiated communication inside one MoE kernel) | applies: peer memory over XGMI via hipIpc, as v1 and MegaMoE TP use |
| move the layer boundary to swallow AR #2 | one launch fewer a layer, 4.2 to 6.3 us a layer at the floor | MPK and Hazy Research (whole-layer kernels) | applies; needs the attention side in the same kernel or a peer handshake between kernels |
| skip the zero pad | v1 already reads 640, production 768 | yuhzhu `opt/best.md` 8.1 (p640 13-15% faster at 256-1024 rows) | applies |
| small-M fused MoE paths | the production comparison point | aiter `aiter/fused_moe.py` (the a16w4 "port" kernels in the trace) | in use |

## Ready-to-file briefs (not filed)

Check ownership first: `pro_opt/wt/mono-v1-perf` exists, so someone may already be tuning mono
v1; #3 and #12 / #72 own the indexer.

**B1. Mono v1 routed stage: weight prefetch and a balanced down.** Goal: the standalone mono
v1 kernel at 8 and 16 rows within 75% of its floor (52 and 87 us against 74.1 and 114.3 today).
File set: `atom/models/minimax_m3/pro_mono/kernels/ffn.py`, its tests. Exit: `perf_layer.py`
at 1 / 8 / 16 rows meets the target, the 8-GPU FFN test passes, 71's timeline shows the
routed stages shrinking. Estimate: 200 code, 80 test lines. GPU: 30 min.

**B2. qk_proj and index_qk_proj as one launch.** Goal: one MXFP8 GEMM launch a layer for both
projections (48 MB) at 6 TB/s or more: about 8 us a layer instead of 15.9 at 8 rows, 0.5 ms a
step (ESTIMATED). File set: the aiter-M3 GEMM (a grouped two-A call), ATOM `pro.py`'s caller.
Exit: a kernel microbench at 1 / 8 / 16 / 32 / 128 rows against the two launches, numerics
bit-equal, the trace showing 68 launches fewer. Estimate: 150 code, 80 test lines (aiter side
first). GPU: 30 min.

**B3. Attention prep into the qk GEMM epilogue, combine into the attention.** Goal: 2 launches
fewer a layer (136 a step; the two kernels take 0.87 ms a step at 8 rows today). File set:
`atom/model_ops/minimax_m3/pro_kernels.py`. Exit: the per-layer attention-side time on a c1 k7
trace below today's 36 us, the 32k / 64k retrieval check 40 / 40, GSM8K. Estimate: 250 code,
100 test lines. GPU: 1 h.

**B4. Count the distinct experts a verify step touches.** Goal: replace the uniform-routing
assumption in the routed floor with the measured count at c1 to c16 (k7). File set: a
profile-only counter next to the router (off by default) and `kernel_roofline.py`'s routed
model reading it. Exit: the routed percent below 100% at every width; the count per width in
the PR. Estimate: 60 code, 40 test lines. GPU: 20 min.

## Reproduce

```bash
# traces of one rank under c<N>/ directories, the checkpoint's config, measured peaks
python3 scripts/m31pro/kernel_roofline.py /scratch/prof17/prof17 \
  --config /mnt/m2m_nobackup/zhiyan/models/m3-pro/config.json \
  --peaks /home/zhiyan/pro_opt/roofline/results/<peaks run>.peaks.json --out <dir>
# traces without sk labels (the plain-decode runs above): add --ctx 1100
# the mono stage floors
python3 scripts/m31pro/kernel_roofline.py --config <config.json> --peaks <peaks> --mono-floor 1,8,16
```

The GPU items (`/home/zhiyan/pro_opt/roofline/roofline_item.sh peaks | mono_tl | traces |
analyze`) take the node lock and write to `/home/zhiyan/pro_opt/roofline/results/` and
`/mnt/m2m_nobackup/zhiyan/roofline/`.
