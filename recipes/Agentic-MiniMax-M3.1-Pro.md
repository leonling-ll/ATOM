# MiniMax-M3.1-Pro AgentX Recipe on MI355X

This recipe runs the SemiAnalysis/Weka AgentX replay workload against MiniMax-M3.1-Pro
with ATOM on 8×AMD MI355X GPUs. It is tuned for **CONC 64**, the point measured so far.
The configuration uses:

- MiniMax-M3.1-Pro (MXFP8 attention/dense, NVFP4 routed experts converted to MXFP4 at load)
- TP8, DCP=1
- FP8 K-only KV cache and FP8 index cache, `--block-size 16`
- native GPU prefix caching plus LMCache CPU offload
- cache-aware admission in the scheduler
- DSpark speculative decoding (the checkpoint's draft), 7 tokens, synthetic acceptance 2.94 (the golden AL, thinking enabled)
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
| `"index_compress": true` | Stores the index cache as one bf16 centroid per 4 tokens (vLLM's compressed index): +17.7% KV blocks, indexer scoring ~4× faster, GSM8K 0.9462; c64 P90 30.22 → 34.43, 11,254 → 11,569 tok/s/GPU. Every switch that changes numerics, with its measured accuracy, is in [`MiniMax-M3.1-Pro-Accuracy-Switches.md`](MiniMax-M3.1-Pro-Accuracy-Switches.md). |

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

`index_split_decode` alone gives 29.10 / 10864; the decode kernels add the rest.

An LMCache smoke on this configuration (util 0.73 so the pool is small; an 18.6k-token prompt is
pushed out of HBM by 14 fillers and asked again) gets the same answers both times, with 18432
tokens loaded from the CPU tier and the re-ask's TTFT down from 2.85 s to 0.20 s.

## Validated configuration

| Item | Value |
|---|---|
| Hardware | 8×MI355X (`gfx950`) |
| ATOM | `yuhua/m3.1-pro` (this recipe's commit) |
| aiter | `yuhua/m3.1-pro` with the inter-640 MoE rows (`[Config] [Tuning] [gfx950] MiniMax-M3.1-Pro MXFP4 MoE rows at inter 640`) and the a16w4 rows at 256/512/1024 tokens, the MXFP8 GEMM rows and `moe_route_glue`. Those rows are built in: do not list `best/tune3/new_rows.csv` or other copies in `AITER_CONFIG_FMOE` (duplicate rows abort startup) |
| Parallelism | TP8, DCP=1 |
| KV cache | FP8, K only, `--block-size 16`; index cache FP8 |
| Prefix cache | on |
| CPU offload | LMCache, 184 GiB a rank, chunk 256 |
| Speculative decoding | DSpark, 7 tokens, synthetic acceptance length 2.94 (golden AL, thinking enabled) |
| Prefill | 16384-token chunks; prefill attention from amd-kernels-day0 when its `amd_kernels` package imports, otherwise the Triton kernel |
| Decode attention | from 24 rows, amd-kernels-day0's FlyDSL sparse decode with an MXFP8 output for `o_proj`, when its `amd_kernels` package imports; otherwise aiter |
| `max-num-seqs` | 2 × CONC (128) |

## 1. Start the ATOM server

Start a fresh server for each run.

```bash
export MODEL_PATH="${MODEL_PATH:-/data/models/m3-pro}"
export PORT="${PORT:-8000}"
export CONC="${CONC:-64}"

export AITER_LOG_LEVEL=WARNING
unset AITER_CUSTOM_AR_DISABLE_LAMPORT          # ar_lamport_rows needs the Lamport path
export ATOM_M3PRO_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "split", "prep": "aiter", "index_split": true, "score_tq": 8, "ar_lamport_rows": 16, "k_only": true, "k_plain_fp8": true, "prefill_attn": "day0", "index_split_decode": 64, "moe_route_glue": true, "index_compress": true}'
export ATOM_WARMUP_TOKENS=auto
export ATOM_ONLINE_QUANT_STREAMING=1
export ATOM_CACHE_AWARE_ADMISSION=1
export AITER_REUSE_IDENTICAL_COMM_GROUPS=1
export GPTOSS_SWIGLU_MXFP4_BF16_BOUND=1040     # bf16 MoE activations up to 128 seqs x 8 verify rows

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
  --gpu-memory-utilization 0.95 \
  --online_quant_config '{"layer_quant_config": {"*.experts": "mxfp4"}}' \
  --method dspark \
  --num-speculative-tokens 7 \
  --spec-decode-acceptance-length 2.94 \
  --text-only \
  2>&1 | tee "server-m31pro-agentx-c${CONC}.log"
```

Check the startup log:

- `Memory budget: ... available_for_kv=74.81GB ... num_kvcache_blocks=173069` (67.43 GB without
  `--text-only` and the comm-group reuse). A pool near 35 GB means pad_align 128 is not in; near
  11 GB, streaming quant is not on.
- One `Creating LMCacheEngine` line per rank with `'max_local_cpu_size': 184.0`.

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

`--spec-decode-acceptance-length 2.94` is the InferenceX synthetic acceptance override: the
DSpark draft still runs, and the flag decides how many draft tokens are committed. It is
**performance-only**; output text is replaced by `synthetic` tokens. Drop it for any accuracy run.

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

- Other concurrencies: only CONC 64 has been measured with these settings.
- A 3600 s run with `GPTOSS_SWIGLU_MXFP4_BF16_BOUND=1040`; the 3600 s row above is without it.
- The intermittent LMCache start hang (see Host memory for LMCache) is not fixed in the launcher.
- DCP=8, which shards the target's K and index K over the eight GPUs (about 4.9× the cached
  tokens with the DSpark draft unsharded), is being brought up and is not part of this recipe.
