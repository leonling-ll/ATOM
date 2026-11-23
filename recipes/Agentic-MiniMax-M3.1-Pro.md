# MiniMax-M3.1-Pro AgentX Recipe on MI355X

This recipe runs the SemiAnalysis/Weka AgentX replay workload against MiniMax-M3.1-Pro
with ATOM on 8×AMD MI355X GPUs. It is tuned for **CONC 64**, the point measured so far.
The configuration uses:

- MiniMax-M3.1-Pro (MXFP8 attention/dense, NVFP4 routed experts converted to MXFP4 at load)
- TP8, DCP=1
- FP8 K-only KV cache and FP8 index cache, `--block-size 16`
- native GPU prefix caching plus LMCache CPU offload
- cache-aware admission in the scheduler
- DSpark speculative decoding (the checkpoint's draft), 7 tokens, synthetic acceptance 4.0
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

No request failed in any row. In the last row LMCache loaded 10.3M tokens back with no failed
load, and nothing was preempted. GSM8K (64-shot, one run, real DSpark acceptance) with streaming
quant and pad_align 128 is 0.9424.

## Validated configuration

| Item | Value |
|---|---|
| Hardware | 8×MI355X (`gfx950`) |
| ATOM | `yuhua/m3.1-pro` (this recipe's commit) |
| aiter | `yuhua/m3.1-pro` with the inter-640 MoE rows (`[Config] [Tuning] [gfx950] MiniMax-M3.1-Pro MXFP4 MoE rows at inter 640`) |
| Parallelism | TP8, DCP=1 |
| KV cache | FP8, K only, `--block-size 16`; index cache FP8 |
| Prefix cache | on |
| CPU offload | LMCache, 184 GiB a rank, chunk 256 |
| Speculative decoding | DSpark, 7 tokens, synthetic acceptance length 4.0 |
| Prefill | 16384-token chunks; prefill attention from amd-kernels-day0 when its `amd_kernels` package imports, otherwise the Triton kernel |
| `max-num-seqs` | 2 × CONC (128) |

## 1. Start the ATOM server

Start a fresh server for each run.

```bash
export MODEL_PATH="${MODEL_PATH:-/data/models/m3-pro}"
export PORT="${PORT:-8000}"
export CONC="${CONC:-64}"

export AITER_LOG_LEVEL=WARNING
unset AITER_CUSTOM_AR_DISABLE_LAMPORT          # ar_lamport_rows needs the Lamport path
export ATOM_M3PRO_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "split", "prep": "aiter", "index_split": true, "score_tq": 8, "ar_lamport_rows": 16, "k_only": true, "k_plain_fp8": true, "prefill_attn": "day0"}'
export ATOM_WARMUP_TOKENS=auto
export ATOM_ONLINE_QUANT_STREAMING=1
export ATOM_CACHE_AWARE_ADMISSION=1

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
  --spec-decode-acceptance-length 4.0 \
  2>&1 | tee "server-m31pro-agentx-c${CONC}.log"
```

Check the startup log:

- `Memory budget: ... available_for_kv=67.43GB ... num_kvcache_blocks=156002`. A pool near 35 GB
  means pad_align 128 is not in; near 11 GB, streaming quant is not on.
- One `Creating LMCacheEngine` line per rank with `'max_local_cpu_size': 184.0`.

### Host memory for LMCache

184 GiB a rank pins about 1.5 TB of host memory. On a node where the NUMA node behind GPUs 0-3
is short of free memory (page cache from the checkpoint counts), ranks 0-3 can stall in LMCache's
CPU backend setup and the server never becomes ready. Drop the checkpoint from the page cache
before starting, or lower `LMCACHE_MAX_LOCAL_CPU_SIZE`.

### Synthetic acceptance

`--spec-decode-acceptance-length 4.0` is the InferenceX synthetic acceptance override: the
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

## InferenceX rules this configuration keeps

- Online quantization touches only the routed experts (`*.experts`, NVFP4 to MXFP4). Attention,
  dense, shared experts and the DSpark draft run as shipped.
- No integer all-reduce: leave `AITER_QUICK_REDUCE_QUANTIZATION` unset.
- The KV cache is FP8, Pro's native format.
- Cache-aware admission and LMCache offload are engine features; an official run needs them in
  the pinned image, not as local patches.

## Not covered yet

- Other concurrencies: only CONC 64 has been measured with these settings.
- A 3600 s run; the table above is 600 s.
- LMCache correctness on this configuration (answers served from the CPU tier compared with a
  fresh prefill); the GSM8K above ran without offload.
- DCP=8, which shards the target's K and index K over the eight GPUs (about 4.9× the cached
  tokens with the DSpark draft unsharded), is being brought up and is not part of this recipe.
