# MiniMax-M3.1-Pro AgentX Recipe on MI355X

This recipe runs the SemiAnalysis/Weka AgentX replay workload against MiniMax-M3.1-Pro
with ATOM on 8×AMD MI355X GPUs. It is tuned for **CONC 64**: sections 1 and 2 are the launch the
CONC 64 proxy measured, and the current best CONC 64 launch on the full workload (1M context) is in
[CONC 64 on the full workload](#conc-64-on-the-full-workload); the same workload at DCP=8 is in
[CONC 64 with DCP=8](#conc-64-with-dcp8). The current best **CONC 32** launch is in [CONC 32](#conc-32),
and the CONC 1-16 launch is in [Small concurrency](#small-concurrency-conc-1-16).
The configuration of sections 1 and 2 uses:

- MiniMax-M3.1-Pro (MXFP8 attention/dense, NVFP4 routed experts converted to MXFP4 at load)
- TP8, DCP=1
- FP8 K-only KV cache and FP8 index cache, `--block-size 16`
- native GPU prefix caching plus LMCache CPU offload
- cache-aware admission in the scheduler
- DSpark speculative decoding (the checkpoint's draft), 7 tokens, synthetic acceptance at Pro's
  golden AL, 2.94 (thinking enabled; the CONC 64 table below ran at 4.0)
- the SemiAnalysis Weka AgentX workload (AIPerf scenario `inferencex-agentx-mvp`)

Related guides: [`MiniMax-M3-Pro.md`](MiniMax-M3-Pro.md) (generic serving, kernels, GSM8K and
AIME), [`DSpark.md`](DSpark.md) (DSpark draft), [`Agentic-Kimi-K3.md`](Agentic-Kimi-K3.md)
(the same workload on Kimi-K3).

## Why the KV pool decides CONC 64

Pro has one KV head, so at TP8 every GPU holds the full KV of every token (about 29 KB a token a
GPU: the 256-wide K, the 128-wide index K and the draft's KV). The whole server caches only what
one GPU can. AgentX sessions come back to their prefix after other sessions have run, so a
small pool evicts them and the server re-prefills most prompts: at the old defaults the CONC 64
prefix hit was 1.5% and about two requests ran while 50-70 waited. A replay with an unbounded
cache gives about 97% hits in the measured phase, and about 5M cached tokens get within a few
points of that. Every setting below is there to enlarge the cache or to keep it warm.

| Setting | What it does |
|---|---|
| `ATOM_ONLINE_QUANT_STREAMING=1` | Converts each NVFP4 expert shard straight into its MXFP4 target while loading. Without it the per-layer conversion leaves ~40 GB of holes in the allocator that no util setting gets back. The converted weights are bit-identical. |
| MoE `pad_align` 128 (automatic for Pro) | Keeps the 640-wide TP8 expert slice unpadded instead of padding it to 768: 32.7 GB less weight a GPU. Needs aiter's inter-640 MoE rows. |
| `--gpu-memory-utilization 0.95` | Possible once the two above are in; 0.87 was the old ceiling. |
| `ATOM_CACHE_AWARE_ADMISSION=1` | Among the first 128 waiting requests, admits those whose prefix is still in HBM first; a request waiting longer than `ATOM_CACHE_AWARE_MAX_WAIT_S` (120 s) goes first regardless. |
| LMCache, 184 GiB a rank | Keeps evicted prefixes in host memory (~7.5M tokens for the server). |
| `--text-only` | Builds and loads no vision tower, projector or patch merger (1.8 GiB of weights a GPU); image requests get a 400. AgentX is text only. |
| `AITER_REUSE_IDENTICAL_COMM_GROUPS=1` | At TP8 aiter's `ep` group has the same ranks as `tp`; this makes it reuse `tp`'s custom all-reduce instead of building a second one (~2.5 GiB a GPU). |
| `GPTOSS_SWIGLU_MXFP4_BF16_BOUND=1040` | Keeps the routed MoE's activations in bf16 (a16w4) for steps below 1040 rows. aiter's default of 256 switches DSpark verify steps (sequences × 8 rows) to MXFP4 activations, which costs about 0.01 GSM8K (see Accuracy). Prefill chunks stay a4w4. |
| `ATOM_TOKEN_GRAPH_SIZES=256,512,1024,2048` | Pro's prefills of up to 2048 tokens pad to the smallest bucket and replay a recorded token graph instead of issuing ~2170 launches from the host (attention stays eager). Measured on the #33 launch and variants (no `moe_route_glue` / `index_compress`, util 0.87, no LMCache), not yet on this recipe's variants: a warm turn on a cached 55k / 94k prefix at c1, paired medians of 8: 16-128 new tokens TTFT 160-191 ms -> 83-125 ms (prefill forward 104 ms -> 24-39 ms); ~1050 new tokens (padded to 2048) -18 to -29 ms; ~2820 new tokens padded to the default's 3072 bucket +5 ms, so 3072 is left out. The CONC 32 launch adds 3072 and 4096 ([Token graphs at CONC 32](#token-graphs-at-conc-32)). |
| `"index_compress": true` | Stores the index cache as one bf16 centroid per 4 tokens (vLLM's compressed index): +17.7% KV blocks, indexer scoring ~4× faster, GSM8K 0.9462; c64 P90 30.22 → 34.43, 11,254 → 11,569 tok/s/GPU. Every switch that changes numerics, with its measured accuracy, is in [`MiniMax-M3.1-Pro-Accuracy-Switches.md`](MiniMax-M3.1-Pro-Accuracy-Switches.md). |
| `ATOM_CUSTOM_AR_POOL_BYTES=134217728` | Sizes aiter's custom all-reduce pool to 128 MiB (routing unchanged: twice the 64 MiB AR cutoff), and token-id broadcasts and post-KV barriers no longer make torch build its own RCCL communicator (~2.4 GiB, one of them after KV sizing). Non-torch memory 9.44 -> 5.31 GB, +7.3% KV blocks. |

## Measured (CONC 64 proxy)

Each row adds one change to the row above. These are **proxy runs, not submissions**: 600 s
profiling instead of 3600 s, the same seed-42 subset of 64 sessions capped at a 128k context,
otherwise the `inferencex-agentx-mvp` scenario, one node, one run per row. Hit ratio is the
server's over the whole run, warmup included; with LMCache it counts GPU and CPU hits.

| Configuration | KV a GPU | Cached tokens | Prefix hit | P90 E2E-norm interactivity | tok/s/GPU | Requests |
|---|---|---|---|---|---|---|
| ATOM `d24ac3a3`, util 0.87 | 11.7 GiB | 434k | 1.5% | 0.83 | 2815 | 273 |
| + streaming quant, util 0.95 | 34.7 GiB | 1.29M | 11.1% | 1.25 | 3873 | 361 |
| + pad_align 128 | 67.4 GiB | 2.50M | 30.9% | 4.22 | 6654 | 621 |
| + cache-aware admission | 67.4 GiB | 2.50M | 73.1% | 4.56 | 6862 | 638 |
| + LMCache 184 GiB a rank | 67.4 GiB | 2.50M + CPU | 77.4% | **30.57** | **10915** | 995 |
| pad_align 128 row + `--text-only` + comm-group reuse (no admission, no LMCache) | 74.8 GiB | 2.77M | 43.3% | 11.44 | 8585 | 778 |
| All of the above together, **3600 s** | 74.8 GiB | 2.77M + CPU | 78.1% | **28.88** | **11047** | 5505 |
| All of the above + `GPTOSS_SWIGLU_MXFP4_BF16_BOUND=1040`, 600 s | 74.8 GiB | 2.83M + CPU | 79.4% | 27.84 | 10910 | 983 |

The 3600 s run held a 77-81% hit ratio and almost no waiting requests in every 10-minute slice,
with 50.8M tokens loaded back from LMCache and no failed load; one request failed on a client-side
disconnect. The bound-1040 row is within run-to-run noise of the others (the same configuration
gave 30.57 / 10915 in one 600 s run and 28.88 / 11047 over 3600 s).

No request failed in any row. In the LMCache row LMCache loaded 10.3M tokens back with no failed
load, and nothing was preempted. The last row branches off the pad_align 128 row; the launch
below turns everything on at once, a combination not yet measured as a whole. GSM8K (64-shot,
one run, real DSpark acceptance) with streaming quant and pad_align 128 is 0.9424.

The rows above force acceptance 4.0. At the golden AL (2.94 for k=7, thinking enabled), on the
same node and setup, 600 s:

| Configuration | Prefix hit | P90 E2E-norm interactivity | tok/s/GPU | Requests | GSM8K |
|---|---|---|---|---|---|
| All of the above + bound 1040 | 78.7% | 21.51 | 9562 | 869 | 0.9393 |
| + decode kernels: FlyDSL sparse decode with MXFP8 out, merged MXFP8 GEMM rows, MoE route glue, `index_split_decode` | 79.7% | **30.22** | **11254** | 1022 | 0.9424 |
| + `index_compress` (compressed bf16 index cache, +17.7% KV blocks) | 83.2% measured phase (batch 1: 82.5%) | **34.43** | **11569** | 1055 | 0.9462 |

With the full 237eada chat template and `--timeout-keep-alive 600` (same node, 600 s, AL 2.94):

| Configuration | Prefix hit (measured phase) | P90 E2E-norm interactivity | tok/s/GPU | Requests |
|---|---|---|---|---|
| Batch 1 (807d48cd) | 69.8% | 22.37 | 11485 | 938 |
| + `index_compress` | 74.6% | **31.34** | **12340** | 1024 |
| + 128 MiB AR pool and shared communicator (a22ad7cb) | 75.6% | 32.49 | 12310 | – |
| + #177 / aiter-M3#19 sync, `--cudagraph-capture-sizes`, prefill prewarm before the client (**this recipe**: ATOM 2239b0cc, aiter 6b9b5e32) | 76.3% | **33.11** | **12826** | 1058 |

The same last two rows on a second node (LMCache 140 GiB): 24.64 / 11,329 -> **27.00 / 12,066**. The last row's server got a short prefill prewarm (35 prompts of 1k-16k tokens, 32 concurrent) before the client started, so no kernel compiled inside the measured window; the rows above it had none, so part of the last step may be first-use compiles they paid.

Shared communicator and 128 MiB AR pool (`ATOM_CUSTOM_AR_POOL_BYTES`), same-node A/B on a second node (LMCache 140 GiB, Pro template, 600 s): KV blocks 203,438 -> 218,199, P90 21.25 -> **24.71** (+16%), 11,479 -> **11,861** tok/s/GPU (+3.3%), 0 failed, no VRAM step after KV sizing.

`index_split_decode` alone gives 29.10 / 10864; the decode kernels add the rest.

An LMCache smoke on this configuration (util 0.73 so the pool is small; an 18.6k-token prompt is
pushed out of HBM by 14 fillers and asked again) gets the same answers both times, with 18432
tokens loaded from the CPU tier and the re-ask's TTFT down from 2.85 s to 0.20 s.

## Validated configuration

| Item | Value |
|---|---|
| Hardware | 8×MI355X (`gfx950`) |
| ATOM | `yuhua/m3.1-pro` (this recipe's commit) |
| aiter | aiter-M3 `yuhua/m3.1-pro` with aiter-M3#19 merged: the inter-640 MoE rows (`[Config] [Tuning] [gfx950] MiniMax-M3.1-Pro MXFP4 MoE rows at inter 640`), the a16w4 rows at 256/512/1024 tokens, the MXFP8 GEMM rows and `moe_route_glue`, and from `hatwu/m3.1-pro-infx` the FlyDSL fp8 sparse prefill (`a26613a8a`) and the op the `ar_fused_rows` / `moe_tail_ar` variants need (`b85ce1a27`, off here). Those rows are built in: do not list `best/tune3/new_rows.csv` or other copies in `AITER_CONFIG_FMOE` (duplicate rows abort startup) |
| Parallelism | TP8, DCP=1 |
| KV cache | FP8, K only, `--block-size 16`; index cache FP8 |
| Prefix cache | on |
| CPU offload | LMCache, 184 GiB a rank, chunk 256 |
| Speculative decoding | DSpark, 7 tokens, synthetic acceptance length 2.94 (golden AL, thinking enabled; the CONC 64 table ran at 4.0) |
| Prefill | 16384-token chunks; prefill attention on aiter's FlyDSL `minimax_m3_fp8_sparse_prefill` (amd-kernels-day0's kernel, ported into aiter-M3 `hatwu/m3.1-pro-infx` at `a26613a8a`) when the aiter build has it, otherwise the Triton kernel. `amd_kernels` is never installed. |
| Decode attention | from 24 rows, aiter's FlyDSL `minimax_m3_fp8_sparse_decode` (amd-kernels-day0's kernel, ported into aiter-M3 `hatwu/m3.1-pro-infx` at `3c55cbcc2`) with an MXFP8 output for `o_proj`, when the aiter build has it; otherwise aiter's paged attention |
| Decode front | up to 16 rows (`ar_lamport_rows`), a layer's q, k, index q and index k come from one MXFP8 GEMM over `[x; lookahead input]`; above it, two GEMMs |
| MoE layers at ≤ 16 rows | mono decode (`mono_ffn_rows` 16, the default): a decode or verify forward of at most 16 rows runs each MoE layer's `o_proj` through its FFN partial as one persistent FlyDSL kernel, with A16W4 routed experts (bf16 activations, the same MXFP4 weights). Needs FlyDSL, the Lamport all-reduce (`ar_lamport_rows` 16) and the day0 decode starting above 16 rows (`day0_decode_rows` 24); without them production serves those layers. Mono sums its all-reduce in the Lamport kernel's order, so a launch with `ar_fused_rows` above 0 runs with mono off (`mono FFN off: ar_fused_rows N overlaps 1 .. 16`): the CONC 32 launch (`"ar_fused_rows": 128`) is one. The small-concurrency launch (`RECIPE=agentic_sc`, Small concurrency below) keeps `ar_fused_rows` at 0 (unset), as these variants do; from CONC 8 it turns mono off (`mono_ffn_rows` 0) |
| `max-num-seqs` | 2 × CONC (128) |

## 0. Build aiter-M3 into the container

The launch below needs aiter-M3 `yuhua/m3.1-pro` with aiter-M3#19 merged (`zhuyuhua-v/aiter-M3`; `hatwu/m3.1-pro-infx` at `a26613a8a` lacks `moe_route_glue`): on the image's own aiter
the server falls back on three paths, each named in its log:

| Fallback | Log line |
|---|---|
| Triton decode attention on the K-only cache (no `value_rows` in `pa_decode_gluon`) | `triton attention, K-only` |
| FlyDSL heuristic MoE configs at inter 640 | `no tuned FlyDSL config ... using heuristic FlyDSL fallback` |
| MXFP8 activation quantization not fused | `aiter MXFP8 norms False, all-reduce MXFP8 epilogue False` |

Build it inside the run container, from the image's aiter checkout, and keep that container for
the runs (the build lives in its filesystem):

```bash
# a clone or bundle of zhuyuhua-v/aiter-M3 is in /w (the node has no access to the private repo)
cd /app/aiter-test
base=$(git rev-parse HEAD)
git fetch /w/aiter-M3.bundle '+refs/heads/*:refs/m3/*'
git checkout --detach refs/m3/yuhua/m3.1-pro   # aiter-M3 yuhua/m3.1-pro, #19 merged
git log -1 --format='aiter %H'                  # record it with the results
```

Then rebuild the JIT modules whose sources, headers, `rocm_ops.hpp` macros or
`optCompilerConfig.json` entries differ between `$base` and the checkout, one at a time in one
process (`aiter.jit.core.rm_module`, `clear_build`, then `build_module` with
`get_args_of_build(module)`), so that eight TP ranks do not race the builds at startup.
Python-only changes need no rebuild. Skip the mha backward modules (`module_mha_bwd`,
`module_mha_varlen_bwd`, `libmha_bwd`): they are training kernels the server never calls, and
`module_mha_bwd` fails to compile on `rocm/atom-dev:nightly_202611201604`. Only remove the
HipKittens MLA modules (`module_hk_*`, MI3xx only). The first server log should contain none
of the log lines in the table above.

## 1. Start the ATOM server

Start a fresh server for each run.

```bash
export MODEL_PATH="${MODEL_PATH:-/data/models/m3-pro}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"     # set to the 237eada chat_template.jinja if the snapshot ships the 1 KB stub (see below)
export PORT="${PORT:-8000}"
export CONC="${CONC:-64}"

export AITER_LOG_LEVEL=WARNING
unset AITER_CUSTOM_AR_DISABLE_LAMPORT          # ar_lamport_rows needs the Lamport path
export ATOM_M3PRO_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "split", "prep": "aiter", "index_split": true, "score_tq": 8, "ar_lamport_rows": 16, "k_only": true, "k_plain_fp8": true, "prefill_attn": "day0", "index_split_decode": 64, "moe_route_glue": true, "index_compress": true}'
export ATOM_WARMUP_TOKENS=auto
export ATOM_ONLINE_QUANT_STREAMING=1
export ATOM_CACHE_AWARE_ADMISSION=1
export AITER_REUSE_IDENTICAL_COMM_GROUPS=1
export ATOM_CUSTOM_AR_POOL_BYTES=134217728   # 128 MiB custom all-reduce pool (non-torch 9.4 -> 5.3 GB)
export GPTOSS_SWIGLU_MXFP4_BF16_BOUND=1040     # bf16 MoE activations up to 128 seqs x 8 verify rows
export ATOM_TOKEN_GRAPH_SIZES=256,512,1024,2048  # small prefills replay token graphs; no 3072 bucket

# LMCache CPU offload
export ATOM_KV_OFFLOAD=lmcache
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=184
export LMCACHE_CHUNK_SIZE=256
export PYTHONHASHSEED=0

rm -rf ~/.cache/atom/*

python3 -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --server-port "${PORT}" \
  --kv_cache_dtype fp8 \
  --index-cache-dtype fp8 \
  --block-size 16 \
  --level 0 \
  --max-model-len 132096 \
  --max-num-seqs $((2 * CONC)) \
  --max-num-batched-tokens 16384 \
  --cudagraph-capture-sizes "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,48,64,128,256,512]" \
  --gpu-memory-utilization 0.95 \
  --online_quant_config '{"layer_quant_config": {"*.experts": "mxfp4"}}' \
  --method dspark \
  --num-speculative-tokens 7 \
  --spec-decode-acceptance-length 2.94 \
  --text-only \
  --timeout-keep-alive 600 \
  ${CHAT_TEMPLATE:+--chat-template "${CHAT_TEMPLATE}"} \
  2>&1 | tee "server-m31pro-agentx-c${CONC}.log"
```

**Chat template.** Serve with the checkpoint's full M3.1-Pro template (revision 237eada, the one vLLM uses; it has the `thinking_mode` switch and defaults to thinking on). Snapshot 9d6169d5 has the same weights, config and tokenizer but ships a 1,165-byte stub `chat_template.jinja` that renders only the user messages, so AgentX prompts come out about 21% short; with that snapshot set `CHAT_TEMPLATE` to the 237eada file. `--timeout-keep-alive 600` keeps the server from closing pooled connections that aiperf (300 s pool) still reuses: with the default 5 s a warmup request can hit a closed socket and abort the run.

Check the startup log:

- `Memory budget: ... available_for_kv=74.81GB ... num_kvcache_blocks=173069` (67.43 GB without
  `--text-only` and the comm-group reuse). A pool near 35 GB means pad_align 128 is not in; near
  11 GB, streaming quant is not on.
- One `Creating LMCacheEngine` line per rank with `'max_local_cpu_size': 184.0`.
- After the first request, one `MiniMax-M3.1-Pro mono FFN: forwards of 1 .. 16 rows, routed experts
  A16W4` line per rank. A `MiniMax-M3.1-Pro mono FFN off: <why>` line means mono is off and
  production serves every MoE layer.

### Host memory for LMCache

Each rank pins its whole LMCache pool (184 GiB, about 198 GB) from the NUMA node it runs on, so
each node has to supply four ranks' pools, about 790 GB, after the model has loaded. When a node
is short (page cache from the checkpoint and memory a previous server is still releasing both
count), the pinning slows down without bound: those ranks stop after `NUMA mapping None`, never
log `Created backend: LocalCPUBackend`, and the server never becomes ready, with no error. This
has been seen on two nodes, including a second start right after a successful one. Until the
launcher handles it:

- make sure the previous server's processes are gone and each NUMA node has more free memory
  than 4 × `LMCACHE_MAX_LOCAL_CPU_SIZE` (`numactl -H`) before starting;
- drop the checkpoint from the page cache before starting;
- lower `LMCACHE_MAX_LOCAL_CPU_SIZE` on hosts with less memory per node;
- if a start isn't ready within about 15 minutes, stop it and start again.

### Synthetic acceptance

`--spec-decode-acceptance-length` is the InferenceX synthetic acceptance override: the
DSpark draft still runs, and the flag decides how many draft tokens are committed. It is
**performance-only**; output text is replaced by `synthetic` tokens. Drop it for any accuracy run.

InferenceX takes the value from the model's golden AL curve for the served thinking mode and the
draft length. Pro's thinking-on curve (InferenceX's golden AL file for minimax-m3.1-pro) is 2.94 at
7 draft tokens; `scripts/m31pro/proxy_bench.sh` holds the whole curve and picks the value for its
`SPEC_TOKENS`. There is no thinking-off curve yet.

### Decode graph sizes

A decode step replays the smallest captured graph that holds its sequences, and the padded rows
of that graph cost index score and top-k time as real rows do. The default ladder
(`EngineArgs.cudagraph_capture_sizes`) has nothing between 8, 16 and 32 sequences, so a step with
9 sequences replays the 16-sequence graph and one with 17 the 32-sequence graph. The
`--cudagraph-capture-sizes` ladder above adds every size from 9 to 16 and every second size from
18 to 32; the startup cut to `--max-num-seqs` drops what CONC cannot reach.
`scripts/m31pro/proxy_bench.sh` with `RECIPE=agentic` launches with the same ladder.

At CONC 32 the cut leaves 26 graphs against the default's 12: capture takes 280 s instead of
114 s and the graph pool 1.32 GiB instead of 0.64 GiB. KV sizing reserves the same graph estimate
for either ladder, so the KV pool does not shrink; the extra pool comes out of the HBM left unused
under the 0.95 target. Step times and the c32 proxy before and after are in #99.

## 2. Run the AgentX profile

Same client as [`Agentic-Kimi-K3.md`](Agentic-Kimi-K3.md); only the model and the concurrency
change. `--model` must be the served model name, which is the `--model` path above unless
`--served-model-name` is set.

```bash
export CONC="${CONC:-64}"
export PORT="${PORT:-8000}"
export MODEL_PATH="${MODEL_PATH:-/data/models/m3-pro}"
export OUTPUT_DIR="${OUTPUT_DIR:-results/m31pro-agentx-c${CONC}}"

export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000

mkdir -p "${OUTPUT_DIR}"

aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url "http://127.0.0.1:${PORT}" \
  --endpoint /v1/chat/completions \
  --endpoint-type chat \
  --streaming \
  --model "${MODEL_PATH}" \
  --concurrency "${CONC}" \
  --benchmark-duration 3600 \
  --stats-interval 30 \
  --random-seed 42 \
  --failed-request-threshold 0.10 \
  --trajectory-start-min-ratio 0.25 \
  --trajectory-start-max-ratio 0.75 \
  --warmup-requests-per-lane 10 \
  --trace-idle-gap-cap-seconds 300 \
  --warmup-grace-period 1800 \
  --use-server-token-count \
  --no-gpu-telemetry \
  --tokenizer "${MODEL_PATH}" \
  --tokenizer-trust-remote-code \
  --apply-chat-template \
  --num-dataset-entries 393 \
  --slice-duration 1.0 \
  --output-artifact-dir "${OUTPUT_DIR}" \
  --public-dataset semianalysis_cc_traces_weka_062126 \
  --server-metrics "http://127.0.0.1:${PORT}/metrics" \
  2>&1 | tee "${OUTPUT_DIR}/aiperf.log"
```

The server's `/metrics` has what explains a run: `atom:prefix_cache_hit_ratio`,
`atom:lmcache_loaded_tokens_total`, `atom:requests_running` / `atom:requests_waiting` and
`atom:kv_cache_usage_ratio`.

## CONC 64 on the full workload

The current best CONC 64 launch on the uncapped AgentX corpus, whose sessions grow to 1M tokens. It
was measured with the official InferenceX client (`benchmarks/srt_agentic.sh` at InferenceX
`e1999621`, agentx-harness 1.0.14). It changes five things in section 1's launch:

- **One LMCache server stores the KV once.** Pro has one KV head, so the eight TP ranks hold the
  same KV and section 1's in-process LMCache pins eight identical copies. In LMCache's multiprocess
  mode with `lmcache.mp.tp_rank_collapse` `"auto"`, one rank stores each block and all eight load
  it, so the 2,240 GiB host pool holds a single copy (#76).
- **1M context.** At `--max-model-len 132096` the server cannot schedule the sessions that grow
  past 132k tokens.
- **DSpark 5 draft tokens at AL 2.89**, the golden AL for 5 tokens with thinking on, as in vLLM's
  launch (#225).
- **The decode fusions and the LMCache MP transfer settings** in the block below (#279, #280).
- **vLLM's prefill indexer** for the split prefills (`index_day0`, #286), on aiter-M3's port of
  amd-kernels-day0's kernel.

The aiter is aiter-M3 `yuhua/m3.1-pro` at `882741bd5`: section 0's aiter-M3#19 plus aiter-M3#24. `amd_kernels` is not
needed.

```bash
export MODEL_PATH="${MODEL_PATH:-/data/models/m3-pro}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"     # the 237eada chat_template.jinja, as in section 1
export PORT="${PORT:-8000}"
export CONC="${CONC:-64}"
export LMCACHE_PORT="${LMCACHE_PORT:-5555}"

# One LMCache server for the host, started before ATOM; its L1 pool grows lazily up to 2,240 GiB.
lmcache server --host 127.0.0.1 --port "${LMCACHE_PORT}" --chunk-size 256 --null-block-id -1 \
  --separate-object-groups --supported-transfer-mode lmcache_driven \
  --l1-size-gb 2240 --l1-use-lazy --l1-read-ttl-seconds 900 \
  --eviction-policy LRU --eviction-trigger-watermark 0.98 --max-cpu-workers 4 \
  > lmcache-server.log 2>&1 &
until (exec 3<>"/dev/tcp/127.0.0.1/${LMCACHE_PORT}") 2>/dev/null; do sleep 2; done

export AITER_LOG_LEVEL=WARNING
unset AITER_CUSTOM_AR_DISABLE_LAMPORT          # ar_lamport_rows needs the Lamport path
export ATOM_M3PRO_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "split", "prep": "aiter", "index_split": true, "score_tq": 8, "ar_lamport_rows": 16, "k_only": true, "k_plain_fp8": true, "prefill_attn": "day0", "index_split_decode": 64, "moe_route_glue": true, "index_compress": true, "score_tq_causal": true, "ar_fused_rows": 128, "moe_tail_ar": true, "moe_glue_fold": true, "index_compress_fused": true, "day0_decode_rows": 48, "index_day0": true}'
export ATOM_WARMUP_TOKENS=auto
export ATOM_ONLINE_QUANT_STREAMING=1
export ATOM_CACHE_AWARE_ADMISSION=1
export AITER_REUSE_IDENTICAL_COMM_GROUPS=1
export ATOM_CUSTOM_AR_POOL_BYTES=134217728
export GPTOSS_SWIGLU_MXFP4_BF16_BOUND=1040
export ATOM_TOKEN_GRAPH_SIZES=256,512,1024,2048,3072,4096

# KV offload to the LMCache server
export ATOM_KV_OFFLOAD=lmcache_mp
export ATOM_KV_OFFLOAD_EXTRA_CONFIG="{\"lmcache.mp.host\": \"tcp://127.0.0.1\", \"lmcache.mp.port\": ${LMCACHE_PORT}, \"lmcache.mp.tp_rank_collapse\": \"auto\", \"lmcache.mp.transfer_deadline_s\": 120}"
export LMCACHE_CHUNK_SIZE=256
export OFFLOAD_MAX_PENDING_SAVES=32
export OFFLOAD_MP_SAVE_SAME_STEP=1
export OFFLOAD_MIN_LOAD_TOKENS=2048
export OFFLOAD_MP_TRANSFER_THREAD=1
export PYTHONHASHSEED=0

rm -rf ~/.cache/atom/*

python3 -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --served-model-name minimax-m3-pro \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --server-port "${PORT}" \
  --kv_cache_dtype fp8 \
  --index-cache-dtype fp8 \
  --block-size 16 \
  --level 0 \
  --max-model-len 1048576 \
  --max-num-seqs $((2 * CONC)) \
  --max-num-batched-tokens 16384 \
  --cudagraph-capture-sizes "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,48,64,128,256,512]" \
  --gpu-memory-utilization 0.95 \
  --online_quant_config '{"layer_quant_config": {"*.experts": "mxfp4"}}' \
  --method dspark \
  --num-speculative-tokens 5 \
  --spec-decode-acceptance-length 2.89 \
  --text-only \
  --timeout-keep-alive 900 \
  ${CHAT_TEMPLATE:+--chat-template "${CHAT_TEMPLATE}"} \
  2>&1 | tee "server-m31pro-agentx-full-c${CONC}.log"
```

What the settings beyond section 1 do:

| Setting | What it does |
|---|---|
| `score_tq_causal` | The index score kernel leaves the blocks past each row's context unwritten instead of writing -inf (`362bd6bd`). A verify graph's score table is max-model-len / 16 blocks wide, 65,536 at 1M. The selected blocks are the same bit for bit. |
| `ar_fused_rows` 128, `moe_tail_ar` | One-launch Lamport all-reduce + RMSNorm + MXFP8 up to 128 rows, and the shared-expert add in the next all-reduce (#127). |
| `day0_decode_rows` 48, `index_compress_fused`, `moe_glue_fold` | Decode attention on aiter-M3's FlyDSL fp8 sparse decode from 48 rows (#208); the compressed index insert in one launch (#216); the MoE glue's shared add in the next Lamport all-reduce and the lookahead norm from the route op (#220). |
| `ATOM_TOKEN_GRAPH_SIZES` up to 4096 | Token graphs for prefills up to 4096 tokens (#217). |
| `index_day0` | A split prefill's index scores and top-128 come from aiter-M3's FlyDSL `minimax_m3_index_top128`, amd-kernels-day0's `index_top128` (vLLM's prefill indexer) ported in aiter-M3#24, instead of the Triton kernels (#286). Its scores are fp16, as in vLLM, so a near-tie can pick another block ([`MiniMax-M3.1-Pro-Accuracy-Switches.md`](MiniMax-M3.1-Pro-Accuracy-Switches.md)). `false` turns it off. An aiter without the op logs `M3-Pro index_day0: Triton indexer (this aiter has no minimax_m3_index_top128)` and runs the Triton kernels. |
| `OFFLOAD_MAX_PENDING_SAVES=32` | Lets 32 saves run or wait at once; unset, the bound is twice the copy workers. |
| `OFFLOAD_MP_SAVE_SAME_STEP=1` | Submits a prefill chunk's save once the step that computed it has finished its forward, instead of at the next step. |
| `OFFLOAD_MIN_LOAD_TOKENS=2048` | Loads CPU-tier hits from 2048 tokens; shorter ones are recomputed. `lmcache_mp`'s default is 8192. |
| `OFFLOAD_MP_TRANSFER_THREAD=1` | The TP workers submit transfers from a thread instead of the forward thread. |
| `lmcache.mp.transfer_deadline_s` 120 | ATOM stops the engine when an LMCache MP load or save stays open past this deadline, 30 s by default. At CONC 64 a 65k-118k-token retrieve occasionally takes 13-23 s instead of tens of ms, and loads queued behind it passed 30 s in three runs. |

Check the startup log: `num_kvcache_blocks` near 217,600 (`available_for_kv` near 80 GB) on every
rank, and the `ATOM_KV_OFFLOAD` line with `"lmcache.mp.tp_rank_collapse": "auto"`. Once the first
split prefill has run, the log has no `M3-Pro index_day0: Triton indexer` line. The client needs
the served name `minimax-m3-pro`; with InferenceX's `srt_agentic.sh` set `SERVED_MODEL_NAME`
to it and `KV_OFFLOADING=dram`, `KV_OFFLOAD_BACKEND=lmcache_mp`, `TOTAL_CPU_DRAM_GB=2405`.
The host needs 2,240 GiB free for the pool once it fills, on top of the server and the page cache.

### Measured (full workload, CONC 64)

The official client's summary over its profiling window, one node, one run per row. Every row ran
the uncapped corpus at 1M context and DSpark k=5 at AL 2.89, and no request failed.

| Launch | Duration | tok/s/GPU | p90 interactivity | TTFT p50 / p90 |
|---|---|---|---|---|
| Section 1's launch (in-process LMCache, 184 GiB a rank) | 3600 s | 3,698 | 7.53 | 322 / 405 s |
| One LMCache server, 2,240 GiB, section 1's variants + `score_tq_causal` (*) | 3600 s | 15,241 | 40.90 | 3.0 / 99.6 s |
| vLLM, #225's launch (2.4 TB CPU offload) | 3600 s | 12,360 | 35.88 | 64 / 100 s |
| This launch without `index_day0` (#282) | 3600 s | 16,360 | 43.30 | 2.6 / 91.7 s |
| This launch | 3600 s | **16,798** | **44.65** | 2.7 / 84.1 s |

Single steps, 600 s each on one node. 600 s runs of one launch differ by up to about 10% run to run.

| Launch | tok/s/GPU | p90 interactivity | TTFT p50 / p90 |
|---|---|---|---|
| One LMCache server, 2,240 GiB (*) | 13,237 | 39.35 | 0.3 / 85.4 s |
| + the decode fusions (*) | 15,631 | 45.19 | 0.5 / 76.2 s |
| + the decode fusions, without `index_day0` | 14,584 | 42.22 | 0.3 / 79.1 s |
| + the LMCache MP transfer settings, no fusions (*, transfer profiling on) | 14,610 | 46.66 | 0.5 / 72.8 s |
| This launch without `index_day0` (#282) | 14,664 | 44.24 | 0.5 / 69.9 s |
| This launch | 15,047 | 45.96 | 0.5 / 85.0 s |
| This launch with `"index_day0": false`, right after the row above | 15,478 | 51.62 | 0.4 / 78.2 s |

(*) These runs had `index_day0` on amd-kernels-day0's own kernel, before its aiter-M3 port; the
third row is the second without it. The last two rows ran back to back on one node: 600 s does not
resolve the hour's 2.7%.

Over the hour, the one-server launch recomputed 4.6% of its prompt tokens and vLLM 9.4%: ATOM
served about 30% from HBM and 66% from the CPU tier, vLLM 7% and 84%. The in-process run
recomputed most of its prompts and queued: its median TTFT was 322 s.

GSM8K for the second row's launch (DSpark k=5 at real acceptance, 3 runs): 0.9439 64-shot,
0.9353 on vLLM's 5-shot protocol, where vLLM scores 0.9174 with DSpark. For this launch, one run:
0.9431 ± 0.0064 64-shot and 0.9356 ± 0.0068 5-shot, acceptance length 3.85 (without `index_day0`:
0.9416 and 0.9249, 3.84). AIME 2026 for this launch: avg@16 0.9625 ± 0.0029, pass@16 0.9667.

## CONC 64 with DCP=8

The full-workload CONC 64 launch with decode context parallelism 8 (`--decode-context-parallel-size 8`,
see [`MiniMax-M3-Pro.md`](MiniMax-M3-Pro.md#decode-context-parallelism-dcp-8)). Each GPU holds every
eighth block of the target's K and index K, so the KV pool holds about 4.9× the tokens (the DSpark
draft's KV is not sharded), and at CONC 64 nothing queues: the GPU prefix hit is 0.90 against 0.24 at
DCP=1. It changes these things in the DCP=1 launch above:

- **`--decode-context-parallel-size 8`** and **DSpark 3 draft tokens at AL 2.53**, the golden AL for
  3 tokens with thinking on, as vLLM's TP8 + DCP8 launch runs it.
- **The DCP variants**: `dcp_topk_merge`, `dcp_mailbox` 768, `dcp_fold`, `dcp_q_fp8` and
  `dcp_day0_decode`, which attends decode and verify steps with aiter-M3's FlyDSL
  `minimax_m3_fp8_sparse_decode_dcp` instead of the Triton attention. `index_compress` is off: the
  measured runs used the fp8 index cache. `index_day0` is in the JSON as it ran, but it needs
  `index_compress` and DCP=1, so the split prefills run the Triton indexer. `index_compress_fused`
  and `day0_decode_rows` are not set.
- **Decode protection**: `ATOM_PREFILL_DECODE_INTERVAL=4` runs at least 4 scheduler passes for decode
  after each prefill forward. Without it a DP=1 server is prefill first, and at DCP=8 the pool has room
  for every waiting prefill, so prefill steps stop all decodes (one run without it had two minutes with
  no decode step). `ATOM_PREFILL_DELAYER_TARGET_FILL=0.01` and `ATOM_PREFILL_DELAYER_STALL_TICKS=1`
  let a held prefill go as soon as the interval allows instead of waiting to fill a chunk.
- **LMCache MP under DCP**: `lmcache.mp.tp_rank_collapse` is `false`, since each rank's pages are its
  own shard (ATOM refuses collapse under DCP); 16 saves in flight; 8 GPU workers on the LMCache
  server. The DCP=1 launch's `OFFLOAD_MP_SAVE_SAME_STEP`, `OFFLOAD_MIN_LOAD_TOKENS` and
  `OFFLOAD_MP_TRANSFER_THREAD` are not set, as in the measured runs.

The aiter is aiter-M3 `yuhua/m3.1-pro` at `d76a4587f` (aiter-M3#36), which adds
`minimax_m3_fp8_sparse_decode_dcp`, built as in section 0. On an aiter without it the server's
`M3-Pro attention:` line says `triton` where it should say `DCP decode day0 attention`, and decode
runs the Triton attention. `amd_kernels` is not needed.

```bash
export MODEL_PATH="${MODEL_PATH:-/data/models/m3-pro}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"     # the 237eada chat_template.jinja, as in section 1
export PORT="${PORT:-8000}"
export CONC="${CONC:-64}"
export LMCACHE_PORT="${LMCACHE_PORT:-5555}"

# One LMCache server for the host, started before ATOM; its L1 pool grows lazily up to 2,240 GiB.
lmcache server --host 127.0.0.1 --port "${LMCACHE_PORT}" --chunk-size 256 --null-block-id -1 \
  --separate-object-groups --supported-transfer-mode lmcache_driven \
  --l1-size-gb 2240 --l1-use-lazy --l1-read-ttl-seconds 900 \
  --eviction-policy LRU --eviction-trigger-watermark 0.98 --max-cpu-workers 4 --max-gpu-workers 8 \
  > lmcache-server.log 2>&1 &
until (exec 3<>"/dev/tcp/127.0.0.1/${LMCACHE_PORT}") 2>/dev/null; do sleep 2; done

export AITER_LOG_LEVEL=WARNING
unset AITER_CUSTOM_AR_DISABLE_LAMPORT          # ar_lamport_rows needs the Lamport path
export ATOM_M3PRO_VARIANTS='{"kernels":true,"attn":"aiter","topk":"split","prep":"aiter","index_split":true,"score_tq":8,"ar_lamport_rows":16,"k_only":true,"k_plain_fp8":true,"prefill_attn":"day0","index_split_decode":64,"moe_route_glue":true,"index_compress":false,"score_tq_causal":true,"index_day0":true,"dcp_topk_merge":true,"ar_fused_rows":128,"moe_tail_ar":true,"moe_glue_fold":true,"dcp_mailbox":768,"dcp_fold":true,"dcp_q_fp8":true,"dcp_day0_decode":true}'
export ATOM_WARMUP_TOKENS=auto
export ATOM_ONLINE_QUANT_STREAMING=1
export ATOM_CACHE_AWARE_ADMISSION=1
export AITER_REUSE_IDENTICAL_COMM_GROUPS=1
export ATOM_CUSTOM_AR_POOL_BYTES=134217728
export GPTOSS_SWIGLU_MXFP4_BF16_BOUND=1040
export ATOM_TOKEN_GRAPH_SIZES=256,512,1024,2048,3072,4096

# decode protection
export ATOM_PREFILL_DECODE_INTERVAL=4
export ATOM_PREFILL_DELAYER_TARGET_FILL=0.01
export ATOM_PREFILL_DELAYER_STALL_TICKS=1

# KV offload to the LMCache server; under DCP every rank stores its own shard
export ATOM_KV_OFFLOAD=lmcache_mp
export ATOM_KV_OFFLOAD_EXTRA_CONFIG="{\"lmcache.mp.host\": \"tcp://127.0.0.1\", \"lmcache.mp.port\": ${LMCACHE_PORT}, \"lmcache.mp.tp_rank_collapse\": false, \"lmcache.mp.transfer_deadline_s\": 120}"
export LMCACHE_CHUNK_SIZE=256
export OFFLOAD_MAX_PENDING_SAVES=16
export PYTHONHASHSEED=0

rm -rf ~/.cache/atom/*

python3 -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --served-model-name minimax-m3-pro \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --decode-context-parallel-size 8 \
  --server-port "${PORT}" \
  --kv_cache_dtype fp8 \
  --index-cache-dtype fp8 \
  --block-size 16 \
  --level 0 \
  --max-model-len 1048576 \
  --max-num-seqs $((2 * CONC)) \
  --max-num-batched-tokens 16384 \
  --cudagraph-capture-sizes "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,48,64,128,256,512]" \
  --gpu-memory-utilization 0.95 \
  --online_quant_config '{"layer_quant_config": {"*.experts": "mxfp4"}}' \
  --method dspark \
  --num-speculative-tokens 3 \
  --spec-decode-acceptance-length 2.53 \
  --text-only \
  --timeout-keep-alive 900 \
  ${CHAT_TEMPLATE:+--chat-template "${CHAT_TEMPLATE}"} \
  2>&1 | tee "server-m31pro-agentx-dcp8-c${CONC}.log"
```

Check the startup log: the `ATOM_KV_OFFLOAD` line with `"lmcache.mp.tp_rank_collapse": false`,
`Forced speculative acceptance ON: mean acceptance length 2.5300 over 3 draft positions`, and an
`M3-Pro attention:` line a rank with `DCP decode day0 attention`. The client is the one of
[CONC 64 on the full workload](#conc-64-on-the-full-workload), with the served name
`minimax-m3-pro`. The measured runs also set `OFFLOAD_PROFILE=1` and
`ATOM_ENABLE_METRICS_DEVICE_TIMER=1`, which only add logging.

### Measured (full workload, CONC 64, DCP=8)

The official client (InferenceX `e1999621`, agentx-harness 1.0.14), uncapped corpus at 1M context,
one run per row, no request failed. The rows of this launch ran on ATOM `yuhzhu/attn9-dcp` `782d18cd`
(the DCP code of this recipe's commit before its merge) with aiter-M3's sync tree plus the DCP decode
op (`84822644e`, the op merged in aiter-M3#36); the last row ran on `7c636337` with the sync tree
alone, and the DCP=1 row is the full-workload launch above.

| Launch | Duration | tok/s/GPU | P90 / P50 interactivity | TTFT p50 / p90 | Requests | GPU prefix hit |
|---|---|---:|---:|---:|---:|---:|
| This launch (node B) | 3600 s | **19,681** | 17.74 / 23.75 | 0.6 / 2.5 s | 4,876 | 0.903 |
| DCP=1: the full-workload CONC 64 launch above (node A) | 3600 s | 16,798 | 44.65 / 64.6 | 2.7 / 84.1 s | 4,210 | 0.242 (+ 0.716 CPU) |
| This launch (node A) | 600 s | 18,469 | 19.9 / 29.2 | 0.48 / 2.38 s | 923 | 0.892 |
| This launch without decode protection, `dcp_mailbox`, `dcp_fold`, `dcp_q_fp8` and `dcp_day0_decode` (ATOM `7c636337`, node B) | 600 s | 17,344 | 18.2 / 28.4 | 0.34 / 2.00 s | 872 | 0.893 |

Over the hour DCP=8 gives 17% more tok/s/GPU than DCP=1 and trades per-user speed for no queueing:
P90 and P50 interactivity are 2.5× and 2.7× lower, but TTFT p90 is 2.5 s instead of 84 s and the
GPU serves 0.90 of the prompt tokens from HBM instead of 0.24. The two 3600 s rows ran on different
nodes. The 600 s rows are not 3600 s rows: 600 s runs of one launch differ by up to about 10%.
Against the last row, this launch's decode step costs 1.39 ms a sequence instead of 1.51 (−8%) and
prefill 51.5 µs a token instead of 55.0 (600 s each, different nodes).

GSM8K (64-shot, DSpark k=5 at real acceptance, 3 runs, DCP=8 with `dcp_topk_merge`, fp8 index
cache): 0.9414 without the DCP cuts, 0.9391 with `dcp_mailbox` 768, `dcp_fold` and `dcp_q_fp8`,
0.9409 with `dcp_day0_decode`, none apart beyond the repeats
([`MiniMax-M3.1-Pro-Accuracy-Switches.md`](MiniMax-M3.1-Pro-Accuracy-Switches.md)).

## CONC 32

The current best CONC 32 configuration, as `scripts/m31pro/proxy_bench.sh` launches it with
`RECIPE=agentic`. It differs from the CONC 64 launch above: DSpark runs 5 draft tokens, the
variants add the c32 kernels. Everything else is the section 1 launch: `RECIPE=agentic` launches
that block's env and flags (`tests/test_m31pro_proxy_bench.py` diffs the two), and the full chat
template comes from `CHAT_TEMPLATE`.

### aiter-M3 for CONC 32

aiter-M3 `hatwu/m3.1-pro-infx` at `5bc8d81ec`, built as in section 0. That commit has
aiter-M3#12 (FlyDSL decode indexer scorer, LDS ring), #13 (`minimax_m3_moe_experts_decode`),
#15 (a16w4 MoE stage-1 rows at 32 and 64 tokens), #16 (fp8 sparse prefill refreshed to
amd-kernels-day0 a9cd9a21), #18 (deep B pipeline for the a16w4 MoE kernels) and the sync
aiter-M3#17, which brings `yuhua/m3.1-pro`'s fused all-reduce + RMSNorm + router + top-k + sort.
`moe_route_glue` needs aiter-M3#17: on an older aiter the server logs
`M3-Pro moe_route_glue unavailable` and keeps the separate launches.

The numbers below ran on aiter-M3 `3c55cbcc2` with the route-glue files of `yuhua/m3.1-pro`
`6e7523d6e` copied in, not on `5bc8d81ec`; #12, #13, #15, #16 and #18 are not in them.

### Launch for CONC 32

```bash
export MODEL_PATH="${MODEL_PATH:-/data/models/m3-pro}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-/shared_nfs/huggingface_models/amd/MiniMax-M3-MXFP4/chat_template.jinja}"
export PORT="${PORT:-8000}"
export CONC=32

export AITER_LOG_LEVEL=WARNING
export AITER_QUICK_REDUCE_QUANTIZATION=NONE    # quick all-reduce is forbidden
unset AITER_CUSTOM_AR_DISABLE_LAMPORT          # ar_lamport_rows and ar_fused_rows need the Lamport path
export AITER_REUSE_IDENTICAL_COMM_GROUPS=1
export ATOM_M3PRO_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "split", "prep": "aiter", "index_split": true, "score_tq": 8, "ar_lamport_rows": 16, "k_only": true, "k_plain_fp8": true, "prefill_attn": "day0", "index_split_decode": 64, "moe_route_glue": true, "index": "fused", "ar_fused_rows": 128, "index_compress": true, "day0_decode_rows": 48}'
export ATOM_WARMUP_TOKENS=auto
export ATOM_ONLINE_QUANT_STREAMING=1
export ATOM_CACHE_AWARE_ADMISSION=1
export ATOM_CUSTOM_AR_POOL_BYTES=134217728
export GPTOSS_SWIGLU_MXFP4_BF16_BOUND=1040
export ATOM_TOKEN_GRAPH_SIZES=256,512,1024,2048,3072,4096  # prefills of up to 4096 tokens replay token graphs

# LMCache CPU offload
export ATOM_KV_OFFLOAD=lmcache
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=184
export LMCACHE_CHUNK_SIZE=256
export PYTHONHASHSEED=0

rm -rf ~/.cache/atom/*

python3 -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --server-port "${PORT}" \
  --kv_cache_dtype fp8 \
  --index-cache-dtype fp8 \
  --block-size 16 \
  --level 0 \
  --max-model-len 132096 \
  --max-num-seqs $((2 * CONC)) \
  --max-num-batched-tokens 16384 \
  --cudagraph-capture-sizes "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,48,64,128,256,512]" \
  --gpu-memory-utilization 0.95 \
  --online_quant_config '{"layer_quant_config": {"*.experts": "mxfp4"}}' \
  --method dspark \
  --num-speculative-tokens 5 \
  --spec-decode-acceptance-length 2.89 \
  --text-only \
  --timeout-keep-alive 600 \
  --chat-template "${CHAT_TEMPLATE}" \
  2>&1 | tee "server-m31pro-agentx-c${CONC}.log"
```

The proxy run starts the same server. `RECIPE=agentic` sets the section 1 env and flags itself and
merges `ATOM_M3PRO_VARIANTS` onto the section 1 variants, so only the c32 keys go there.
`ATOM_TOKEN_GRAPH_SIZES` replaces section 1's buckets when set (empty runs every prefill eagerly). It refuses
a checkpoint whose `chat_template.jinja` is the stub unless `CHAT_TEMPLATE` is set. The template
named here is the full one with the `thinking_mode` switch that `recipes/MiniMax-M3-Pro.md` serves
with and `scripts/m31pro/aime_gate.sh` uses; the 237eada file is not on `/shared_nfs`.

```bash
export RECIPE=agentic SPEC_TOKENS=5
export MODEL="${MODEL_PATH}"
export CHAT_TEMPLATE=/shared_nfs/huggingface_models/amd/MiniMax-M3-MXFP4/chat_template.jinja
export ATOM_M3PRO_VARIANTS='{"index": "fused", "ar_fused_rows": 128, "day0_decode_rows": 48}'
export ATOM_TOKEN_GRAPH_SIZES=256,512,1024,2048,3072,4096
flock /tmp/m31pro-gpu.lock bash scripts/m31pro/proxy_bench.sh 32 "${OUT_DIR}"
```

- **AL is forced.** k=5 at Pro's golden AL 2.89 (thinking on), as vLLM moo runs c32; on the
  full template k=5 matched k=3's throughput at a higher p90
  ([#7](https://github.com/zhuyuhua-v/ATOM-M3/issues/7), 2026-11-25). `proxy_bench.sh` takes the
  AL from its golden AL curve for `SPEC_TOKENS=5`. The server logs
  `Forced speculative acceptance ON: mean acceptance length 2.8900 over 5 draft positions`, and
  the run measures 2.89 (`atom:mtp_average_tokens_per_forward`). Output text is `synthetic`.
- No `amd_kernels`: day0's prefill and decode kernels run from aiter-M3. The day0 decode
  needs an aiter-M3 at `3c55cbcc2` or later; an older build keeps aiter's paged attention.
  `"day0_decode_rows": 48` (the default is 24): at 24 to 40 decode rows, most CONC 32 k=3
  steps, aiter's value_rows attention is faster; from 48 rows the FlyDSL decode is
  ([#208](https://github.com/zhuyuhua-v/ATOM-M3/pull/208), #207).
- `"prep": "aiter"` runs Triton's prep here: aiter's prep declines `k_plain_fp8` and
  `index_compress` (the server logs `triton prep`).
- With `index_compress` on, `"index": "fused"` scores the bf16 centroid cache on aiter-M3's FlyDSL
  scorer in graphs below `index_split_decode` (#183), where the aiter has the centroid format;
  an aiter without it logs `decode scores stay Triton's` once a rank and keeps Triton's
  BLOCK_CENTROID. Graphs of `index_split_decode` rows or more always score on Triton.
- The capture ladder is #99's. At `--max-num-seqs 64` and k=5 the startup cut keeps 26 graphs,
  1 to 64.
- With k=5 a verify step at CONC 32 is at most 192 rows, under aiter's default 256-row MXFP4
  activation bound, so at CONC 32 `GPTOSS_SWIGLU_MXFP4_BF16_BOUND=1040` changes only steps of
  256 to 1039 rows, which are prefill chunks. The measured rows below ran without it.

Startup log, measured without the AR pool: `available_for_kv=77.69GB ... num_kvcache_blocks=211452`, one
`M3-Pro moe_route_glue: fused all-reduce + norm + router + top-k + sort` line a rank, and one
`Creating LMCacheEngine` line a rank with `'max_local_cpu_size': 184.0`.

### What each landed change does at CONC 32

| Change | In this launch |
|---|---|
| #108 / #122 | `"index": "fused"`: decode and verify scores on aiter-M3's FlyDSL scorer; with `index_compress` on, only graphs below `index_split_decode` (above). |
| #119 / #124 | `"prefill_attn": "day0"`: aiter-M3's FlyDSL `minimax_m3_fp8_sparse_prefill`, no `amd_kernels` import. |
| #110 / #127 | `"ar_fused_rows": 128`: up to 128 rows, the all-reduce + residual + RMSNorm + MXFP8 sites run as one Lamport launch. `moe_tail_ar` stays off. |
| #99 / #155 | The capture ladder between 8, 16 and 32 sequences (Decode graph sizes, above). |
| #97 / #163 | `"moe_decode": "day0"` runs MoE calls of at most 64 rows on aiter-M3#13's `minimax_m3_moe_experts_decode`. Off (`"aiter"`, the default); not yet measured at c32. |
| #42 / #161 | `moe_two_stream_rows`: the shared expert on a side stream. Off (0, the default). |
| #101 / #120 | `proxy_bench.sh` gets the golden AL curve and the `RECIPE=agentic` launch; `accuracy_gate.sh` the recipe's utilization. |
| #152 / #156 | `accuracy_gate.sh` takes `GATE_VARIANTS` and `GATE_DISABLE_LAMPORT`, so GSM8K runs with these variants and Lamport on. |
| #173 / #174 | `scripts/m31pro/aime_gate.sh`: AIME 2025 through the chat endpoint, thinking on, provisional bar 0.90. |
| #128 (`yuhua/m3.1-pro`, via #176) | `moe_route_glue`, `index_split_decode` 64, the MXFP8 GEMM rows, the day0 decode guard. |
| #137 (`yuhua/m3.1-pro`, via #176) | `index_compress`: the bf16 centroid index cache. |

### Measured (CONC 32 proxy)

`proxy_bench.sh`, 900 s, the seed-42 subset of 64 sessions capped at a 128k context, one run per
row. Run details are in #7.

**These rows ran with the stub chat template.** The checkpoint's 1,165-byte `chat_template.jinja`
renders only the user messages, so every AgentX prompt is about 21% short (see the chat template
note in section 1). The rows compare with each other, not with the CONC 64 tables or with a run
on the full template. The launch above, as `proxy_bench.sh` now starts it, has not been measured yet.

| Run | ATOM | aiter | k | AL | tok/s/GPU | P90 E2E-norm interactivity | Failed |
|---|---|---|---|---|---:|---:|---|
| #99 baseline | `e7f85047` | image aiter `572232d23` | 7 | 2.94 | 5000.0 | 8.01 | 0/642 |
| First batch (#108, #119) | `e7f85047` + #122, #124 (PR heads) | aiter-M3 `c6a7d6d8` | 7 | 2.94 | 7149.3 | 42.59 | 0/917 |
| Second batch (+ #110, #112) | `03309580` (`843a79cb` + `--text-only`) | aiter-M3 `3c55cbcc2` | 3 | 2.53 | 7321.7 | 45.45 | 0/936 |
| + `yuhua/m3.1-pro`, `index_compress` off | `28356f19` (`78c55bcf` + `yuhua/m3.1-pro` `a22ad7cb`) | `3c55cbcc2` + route glue | 3 | 2.53 | 7482.2 | 44.59 | 0/952 |
| + `index_compress` on (this launch) | `28356f19` | `3c55cbcc2` + route glue | 3 | 2.53 | 7628.2 | 49.99 | 0/969 |

With `index_compress` on, KV blocks go from 179,728 to 211,452 and LMCache reloads 0.48M tokens
instead of 1.56M. From the second batch on, the draft length is 3, so those rows are not
like-for-like with the k=7 rows above them.

### Token graphs at CONC 32

`ATOM_TOKEN_GRAPH_SIZES` picked by three 15-minute c32 proxies on one node (m31pro-256), the full
chat template, DSpark k=5 at its golden AL 2.89 (measured 2.89), aiter-M3 `b7f589913`, ATOM
`29b7ed93` (`ATOM_TOKEN_GRAPH_SIZES=""` needs #205's `proxy_bench.sh`). KV blocks are 218,276 in
every arm: the 3072 and 4096 recordings come out of the margin, not the pool.

| `ATOM_TOKEN_GRAPH_SIZES` | tok/s/GPU (output) | P90 E2E-norm interactivity | Prefill forwards replayed / eager | Failed |
|---|---:|---:|---|---|
| `""` (eager) | 8322.8 (84.63) | 48.83 | 0 / 977 | 0/956 |
| `256,512,1024,2048` | 8483.6 (86.05) | 53.29 | 637 / 375 | 0/971 |
| `256,512,1024,2048,3072,4096` (this launch) | **8534.1 (86.44)** | **55.21** | 754 / 232 | 0/974 |

Prefill forward on TP rank 0 (device events around `ModelRunner.run_model`, median ms), eager
against this launch: up to 256 rows 90 -> 37; 257-512 90 -> 53; 513-1024 106 -> 78; 1025-2048
124 -> 106; 2049-3072 135 -> 136 and 3073-4096 158 -> 169, but their means drop 194 -> 146 and
228 -> 183 (the eager forwards of that size have a long host-bound tail). Prefill device time over
the window: 203 s eager, 163 s here.

### Accuracy at CONC 32

- GSM8K 64-shot, second batch (ATOM `03309580`, aiter-M3 `3c55cbcc2`, DSpark k=3 at real
  acceptance, fused index, day0 prefill, `ar_fused_rows` 128 with Lamport on, `index_compress`
  off): flexible-extract 0.9325 + 0.0069 = 0.9394, short of the bar in #7 by 0.0006. Accepted
  without a rerun by the owner's ruling
  ([#7, 2026-11-24](https://github.com/zhuyuhua-v/ATOM-M3/issues/7#issuecomment-7134786667)).
- No GSM8K run yet with `moe_route_glue` and `index_compress` together at k=3.
- AIME 2025 has not run yet; its gate is `scripts/m31pro/aime_gate.sh` (#174).

## Small concurrency (CONC 1-16)

The CONC 1, 4, 8 and 16 launch, as `scripts/m31pro/proxy_bench.sh` starts it with
`RECIPE=agentic_sc` (`tests/test_m31pro_proxy_bench.py` diffs the two). It is the section 1 launch
and aiter (section 0) with LMCache and admission off, the decode indexer split from 24 rows, and a smaller `--max-num-seqs` below CONC 8. It branches at CONC 8: CONC 1 and 4 keep mono and the fp8 index cache on the fused decode scorer; CONC 8 and 16 take the compressed index cache with mono off. CONC 8 then takes its own settings on top (the CONC 8 row), which replace the CONC 8 values of the rows above:

| Setting | Here | Why |
|---|---|---|
| LMCache offload, cache-aware admission | off | Up to CONC 16 the HBM pool holds every live session's prefix: at CONC 16 the prefix working set peaks at about 1.2M tokens, and util 0.95 with streaming quant gives 155,940 blocks (2.5M tokens). On that HBM-only launch, adding LMCache (184 GiB a rank) and admission left the prefix hit at 0.94 and cost 10.6% P90 E2E-normalized interactivity (77.38 -> 69.21) and 1.4% tok/s/GPU (3270.4 -> 3224.7). This was one 900 s run per arm at CONC 16, k=7, AL 2.94, measured in #133. |
| `index_compress` | CONC 1, 4: off. CONC 8, 16: on, with `index_compress_fused` | Up to 16 rows the centroid writes (two launches a layer, 576 us a step) cost more than the faster scoring saves: a decode step is 2.9% slower at CONC 1 (13,018 -> 13,398 us) and 2.0% slower at CONC 4 (#145, 8-row buckets). The CONC 8 and 16 tail decodes at 5-7 sequences (40-56 rows), where the centroid scores are faster, and compress also shortens the long prefill chunks (a 16k chunk at 300-600k context: 584-640 vs 1010-1140 ms). On the official CONC 8 run zhuyuhua-v's launch, which has `index_compress`, gave 105.67 P90 E2E-normalized interactivity against 84.95 for this launch without it; compress is the largest part of that gap by estimate ([#107](https://github.com/zhuyuhua-v/ATOM-M3/issues/107#issuecomment-7186675560)). `index_compress_fused` writes the pooled centroid and the ring in one launch, the same bytes. Measurement: [#304](https://github.com/zhuyuhua-v/ATOM-M3/issues/304). |
| `mono_ffn_rows` | CONC 1, 4: 16 (the default). CONC 8, 16: 0 (off) | Mono serves forwards of at most 16 rows, so it does not reach the CONC 8 tail at 40-56 rows; [#107](https://github.com/zhuyuhua-v/ATOM-M3/issues/107#issuecomment-7186675560) puts mono off at about 0 on CONC 8 P90 (estimated from the official-run comparison). This launch (CONC 16 variant set at `--max-model-len` 132096, DSpark k7 at real acceptance, tree 17bc447d): GSM8K 64-shot strict-match 0.9416 / 0.9340 / 0.9500, mean 0.9419 (flexible 0.9424 / 0.9348 / 0.9500), at num_concurrent 16, measured AL 4.11 on the GSM8K prompts; retrieval 32k 20/20, 64k 20/20 ([#313](https://github.com/zhuyuhua-v/ATOM-M3/pull/313)). |
| `index` | CONC 1, 4: `"fused"`. CONC 8, 16: `"triton"` (the default) | Decode and verify scores on aiter-M3's FlyDSL scorer over the fp8 index cache. Graphs of 32 to 56 rows take it at any `--max-model-len` (the 32-row rule): past 24 rows Triton's scores are not causal and fill the whole block table, 50.5 vs 28.0 us a layer at 4 x 8 rows, 64k context, 1M-wide table. Graphs below 32 rows take it only on a block table at least 12288 blocks wide, that is `--max-model-len` 196608 or more, such as the official runs' 1048576 (`MAX_MODEL_LEN=1048576` for `proxy_bench.sh`): 31.66 vs 21.98 us at 8 rows and 340k, but 8.56 vs 13.42 at 64k. At this block's 132096 only the 32-56-row graphs change. At 1048576, on the 11-25 official runs' decode steps (68 layers, weighted by steps), the index score costs 1.94 -> 1.43 ms a round at CONC 1, 1.99 -> 1.52 at CONC 4, 2.70 -> 1.88 at CONC 8 and 2.50 -> 1.56 at CONC 16 ([#230](https://github.com/zhuyuhua-v/ATOM-M3/issues/230), microbench, not end to end). Graphs of 24 rows or more take `index_split_decode`'s Triton scores (below). With `index_compress` (CONC 8, 16), the graphs below 24 rows stay on Triton's centroid scorer: aiter's FlyDSL centroid scorer is slower there (528 vs 303 us a round of 68 layers at 1 x 8 rows, 64k; one-GPU microbench, not end to end). |
| `index_split_decode` | 24 (section 1: 64) | From 3 sequences of k7 verify rows, each rank scores and selects one sequence and the rows are all-gathered (the same selection bit for bit), instead of every rank doing every row. Below 64 rows nothing split before, and the 24-56-row graphs carry the CONC 8 / 16 tail. On the 11-25 official steplogs (before #300) the 8-sequence round, which split, cost less than the 7-sequence one, which did not: 26.6 vs 35.6 ms at CONC 16, 22.5 vs 32.0 ms at CONC 8. Measurement: [#304](https://github.com/zhuyuhua-v/ATOM-M3/issues/304). |
| `ar_fused_rows` | 0 (unset) | Up to 16 rows the all-reduce already runs on the Lamport path (`ar_lamport_rows` 16), and the mono decode path (#55) refuses an `ar_fused_rows` above 0. At CONC 8 and 16, where mono is off, it has not been measured on this launch. |
| `--max-num-seqs` | 16 below CONC 8, else 2 × CONC | `proxy_bench.sh`'s rule for every small-concurrency point. |
| CONC 8 | variants `{"index": "fused", "mono_ffn_rows": 32, "ar_lamport_rows": 32, "day0_decode_rows": 24, "ar_fused_rows": 64, "ar_fused_min_rows": 16, "moe_glue_fold": true, "index_split_decode": 32, "select_long_rows": 2048, "mono_o_mx8": true, "qk_pair": true, "index_compress": true, "index_compress_fused": true, "moe_route_glue_small": true}` on top; `AITER_M3PRO_DECODE_SPLIT=auto`, `AITER_MOE_ROUTE_ONESHOT_CAP=1` | The CONC 8 launch that passed the CONC 8 GSM8K accuracy gate: mono and the Lamport all-reduce up to 32 rows, the fused all-reduce tail from 16 to 64 rows, the MoE glue fold, long-row selection on aiter's top-k, the MXFP8 o_proj in mono, the q/k pair, the decode index split from 32 rows, the fused index scorer with the compressed index cache, moe_route_glue's 128-row buffers, aiter's sparse decode split on auto and the route op's one-shot inbox cap. CONC 8 only; CONC 16 keeps the values above. |

The rest of section 1 stays, including `moe_route_glue`. In #145 it raised P90 E2E-normalized interactivity
by 14.8% at CONC 1, 9.6% at CONC 4 and 8.2% at CONC 16, on the #33 launch at util 0.87. The chat
template is the 237eada one, as in section 1.

```bash
export MODEL_PATH="${MODEL_PATH:-/data/models/m3-pro}"
export CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"     # the 237eada chat_template.jinja (section 1)
export PORT="${PORT:-8000}"
export CONC="${CONC:-16}"                     # 1 to 16

export AITER_LOG_LEVEL=WARNING
unset AITER_CUSTOM_AR_DISABLE_LAMPORT          # ar_lamport_rows needs the Lamport path
if [ "$CONC" -ge 8 ]; then
  export ATOM_M3PRO_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "split", "prep": "aiter", "index_split": true, "score_tq": 8, "ar_lamport_rows": 16, "k_only": true, "k_plain_fp8": true, "prefill_attn": "day0", "index_split_decode": 24, "moe_route_glue": true, "index_compress": true, "index_compress_fused": true, "index": "triton", "mono_ffn_rows": 0}'
else
  export ATOM_M3PRO_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "split", "prep": "aiter", "index_split": true, "score_tq": 8, "ar_lamport_rows": 16, "k_only": true, "k_plain_fp8": true, "prefill_attn": "day0", "index_split_decode": 24, "moe_route_glue": true, "index": "fused"}'
fi
export ATOM_WARMUP_TOKENS=auto
export ATOM_ONLINE_QUANT_STREAMING=1
export AITER_REUSE_IDENTICAL_COMM_GROUPS=1
export ATOM_CUSTOM_AR_POOL_BYTES=134217728   # 128 MiB custom all-reduce pool
export GPTOSS_SWIGLU_MXFP4_BF16_BOUND=1040
export ATOM_TOKEN_GRAPH_SIZES=256,512,1024,2048
# no LMCache, no cache-aware admission
# CONC 8: its own settings on top (the CONC 8 row)
if [ "$CONC" = 8 ]; then
  export ATOM_M3PRO_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "split", "prep": "aiter", "index_split": true, "score_tq": 8, "ar_lamport_rows": 32, "k_only": true, "k_plain_fp8": true, "prefill_attn": "day0", "index_split_decode": 32, "moe_route_glue": true, "index_compress": true, "index_compress_fused": true, "index": "fused", "mono_ffn_rows": 32, "day0_decode_rows": 24, "ar_fused_rows": 64, "ar_fused_min_rows": 16, "moe_glue_fold": true, "select_long_rows": 2048, "mono_o_mx8": true, "qk_pair": true, "moe_route_glue_small": true}'
  export AITER_M3PRO_DECODE_SPLIT=auto
  export AITER_MOE_ROUTE_ONESHOT_CAP=1
fi

rm -rf ~/.cache/atom/*

python3 -m atom.entrypoints.openai_server \
  --model "${MODEL_PATH}" \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --server-port "${PORT}" \
  --kv_cache_dtype fp8 \
  --index-cache-dtype fp8 \
  --block-size 16 \
  --level 0 \
  --max-model-len 132096 \
  --max-num-seqs $(( CONC < 8 ? 16 : 2 * CONC )) \
  --max-num-batched-tokens 16384 \
  --cudagraph-capture-sizes "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,48,64,128,256,512]" \
  --gpu-memory-utilization 0.95 \
  --online_quant_config '{"layer_quant_config": {"*.experts": "mxfp4"}}' \
  --method dspark \
  --num-speculative-tokens 7 \
  --spec-decode-acceptance-length 2.94 \
  --text-only \
  --timeout-keep-alive 600 \
  ${CHAT_TEMPLATE:+--chat-template "${CHAT_TEMPLATE}"} \
  2>&1 | tee "server-m31pro-agentx-c${CONC}.log"
```

The proxy run, one fresh server a point:

```bash
export RECIPE=agentic_sc MODEL="${MODEL_PATH}" CHAT_TEMPLATE=/path/to/237eada/chat_template.jinja
for CONC in 1 4 8 16; do
  flock /tmp/m31pro-gpu.lock bash scripts/m31pro/proxy_bench.sh "${CONC}" "results/m31pro-agentx-sc-c${CONC}"
done
```

Check the startup log: no `Creating LMCacheEngine` line, one `M3-Pro moe_route_glue` line a rank,
and `num_kvcache_blocks` above the 155,940 that util 0.95 with streaming quant gave without
`--text-only` (#133).

No row has been measured on this launch yet. The CONC 1, 4, 8 and 16 points at k=7, AL 2.94 are #135.

## Accuracy

Start the same server **without** `--spec-decode-acceptance-length` (keep `--method dspark`), then
run GSM8K as in [`MiniMax-M3-Pro.md`](MiniMax-M3-Pro.md):

```bash
lm_eval --model local-completions \
  --model_args "model=${MODEL_PATH},tokenizer=${MODEL_PATH},base_url=http://127.0.0.1:${PORT}/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 64
```

The gate is 0.94; one run has a standard error of about 0.0066, so a single run at 0.933 or above
is within noise.

### Why `GPTOSS_SWIGLU_MXFP4_BF16_BOUND=1040`

aiter runs the routed MoE with bf16 activations below 256 rows and MXFP4 activations from 256
rows up. A DSpark verify step is (sequences × 8) rows, so at 32 concurrent requests every verify
step crosses 256, while plain decode never does. GSM8K, one run each, DSpark k7 at real
acceptance unless noted:

| Configuration | GSM8K |
|---|---|
| DSpark off (4 runs) | 0.936-0.942 |
| DSpark on, default bound 256 (5 runs) | 0.926-0.937, mean ~0.930 |
| DSpark off, 256 concurrent, so plain decode reaches 256 rows | 0.9287 |
| DSpark on, bound 520 (verify in bf16, prefill a4w4) | 0.9462 |
| DSpark on, bound 1000000 (prefill in bf16 too) | 0.9484 |
| DSpark on, bound 1040 with the 256/512/1024-row a16w4 rows | 0.9393 |

The drop follows the MoE activation format, not DSpark: verify-row block selection, the
acceptance rule and attention precision (fp8 or bf16, aiter or Triton) were each ruled out.
Prefill stays a4w4: bf16 prefill gains nothing measurable and runs the MoE 2.6-3.8× slower.
The bound only changes activation precision and leaves the weights as shipped. At CONC 64 most
verify steps are 129-256 rows, where tuned a16w4 is within 3-5% of a4w4 per MoE layer.

## InferenceX rules this configuration keeps

- Online quantization touches only the routed experts (`*.experts`, NVFP4 to MXFP4). Attention,
  dense, shared experts and the DSpark draft run as shipped.
- No integer all-reduce: leave `AITER_QUICK_REDUCE_QUANTIZATION` unset.
- The KV cache is FP8, Pro's native format.
- Cache-aware admission and LMCache offload are engine features; an official run needs them in
  the pinned image, not as local patches.

## Not covered yet

- Other concurrencies: only CONC 64 has been measured with these settings, and CONC 32 with
  its own (CONC 32, above). CONC 1-16 has its launch (Small concurrency, above) but no measured row
  yet.
- CONC 32 with the full 237eada chat template (#182), and at aiter-M3 `5bc8d81ec`.
- A 3600 s run with `GPTOSS_SWIGLU_MXFP4_BF16_BOUND=1040`; the 3600 s row above is without it.
- The intermittent LMCache start hang (see Host memory for LMCache) is not fixed in the launcher.
- DCP=8 has its CONC 64 launch ([CONC 64 with DCP=8](#conc-64-with-dcp8)), measured on the
  pre-merge tree; a run on the merged commits, other concurrencies and `index_compress` at DCP=8
  are not measured yet.
