# MiniMax-M3.1-Pro Usage Guide (8xMI355 GPUs)

MiniMax-M3.1-Pro (`MiniMaxM3SparseForConditionalGeneration` with a `MiniMaxM3TinyForCausalLM`
text model; 68 layers, 384 experts, MXFP8 attention/dense + NVFP4 routed experts) runs through
the native ATOM OpenAI-compatible server. The attention has two implementations: a plain-torch
reference (`atom/model_ops/minimax_m3/pro_attn.py`; about 0.14 s per decoded token at TP8, eager
only) and a kernel path (`atom/model_ops/minimax_m3/pro_kernels.py`: Triton prep, block scores and
block selection, aiter's paged attention, an FP8 KV cache, decode CUDA graphs; about 11-12 ms per
decoded token at TP8). The kernel path is selected through
`ATOM_M3PRO_VARIANTS` (see Kernels below). Images go through the checkpoint's vision tower, prefix
caching is on, and the checkpoint's embedded DSpark draft runs speculative decoding (sections below).

What differs from MiniMax-M3.1-Flash (everything else, i.e. MoE, dense MLP, norms and the
loader, is the Flash path with Pro's sizes):

| | M3.1-Flash | M3.1-Pro |
|---|---|---|
| Q heads / KV heads | 64 / 4 | 128 / 1 (MQA) |
| head_dim (q,k / v) | 128 / 128 | 256 / 192 |
| K/V projection | separate | one `k_proj`; V is the K cache without its 64 rope dims |
| RoPE | theta 10M, partial 0.5 | theta 500K + YaRN x48 (from 24576), partial 0.25 |
| Sparse selection | 16 blocks x 128 tokens | 128 blocks x 16 tokens, 1 init + 9 local forced |
| Indexer | 4 heads x 128 | 16 heads x 128, input from 2 layers back (lookahead) |

## Preparing Environment

```bash
docker pull rocm/atom-dev:latest
```

On `m3.1` the image's aiter has to be from 2026-11-19 or later: `m3.1`'s MiniMax-M3 mono decode
imports aiter's NVFP4 index-score helpers (`minimax_m3_index_score.LUT_ENTRIES` and others) when the
model module loads, so with an older aiter no MiniMax-M3 model starts, Pro included. The
`rocm/atom-dev:latest` built on 2026-11-19 (`sha256:23ee7cf4c16f`, aiter `956fd46d8`) has all of them.
Mono decode itself never runs for Pro: it is built for M3's TP4 shapes, and Pro's TP8 / block-16
deployment is refused at startup (`MiniMax-M3 mono decode off: ...` in the log).

The numbers in this guide were measured on ATOM `main` `9b2e67b7f` with these Pro commits on top,
before they were rebased onto `m3.1`, in the `rocm/atom-dev:latest` of 2026-11-15 (aiter `95d211284`).
The branch carries `feat(minimax-m3.1): load the MiniMax-M3.1-Flash release checkpoint (NVFP4 routed
experts)`, which `m3.1` does not have yet: Pro's routed experts are NVFP4 (`nvfp4-pack-quantized`) in
the same format, and `--online_quant_config` converts them to MXFP4 at load time (the dense/attention
MXFP8 weights are used as shipped).

## Launching Server (TP8)

The fast configuration (kernel path, decode graphs, FP8 KV and index caches):

```bash
export AITER_CUSTOM_AR_DISABLE_LAMPORT=1
export ATOM_M3PRO_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "split"}'

python -m atom.entrypoints.openai_server \
  --model /data/models/yuhua/m3-pro \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --server-port 8300 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 \
  --block-size 16 \
  --kv_cache_dtype fp8 \
  --index-cache-dtype fp8 \
  --level 0 \
  --online_quant_config '{"layer_quant_config": {"*.experts": "mxfp4"}}'
```

Add `--method dspark --num-speculative-tokens 7` for speculative decoding with the checkpoint's
draft. For the torch reference, leave `ATOM_M3PRO_VARIANTS` unset, add `--enforce-eager` and use
`--kv_cache_dtype bf16 --index-cache-dtype bf16`; prefix caching was checked on the kernel path only.

* With `ATOM_ONLINE_QUANT_STREAMING=1` and Pro's MoE `pad_align` 128 (the default now) the weights
  take about 175 GiB per GPU and 0.95 works; see [`Agentic-MiniMax-M3.1-Pro.md`](Agentic-MiniMax-M3.1-Pro.md).
  The note below is for the old padded, non-streaming load.
* The weights take about 210 GB per GPU, so keep `--gpu-memory-utilization` at 0.85: at 0.88 the
  KV pool takes every free byte and the first NCCL barrier fails to allocate.
* `--block-size 16` and `--level 0` are required. Decode graphs need the kernel path: the torch
  path reads the sequence lengths back to the host, raises if a capture reaches it, and runs with
  `--enforce-eager`.
* `--kv_cache_dtype fp8` stores K and V scaled per token (half the KV memory); the index cache
  follows the KV cache's dtype unless `--index-cache-dtype` says otherwise. bf16 and fp8 are both
  validated (Kernels below).

## Quick check

Pro is a plain-completion model whose chat template is just the BOS token `]~!b[` in front of
the text. `/v1/chat/completions` adds it; on `/v1/completions` it is up to the prompt (the tokenizer
does not add it). On an earlier ATOM base leaving it out cost GSM8K accuracy; on the current base it
makes no measurable difference, and the GSM8K run below sends none.

```bash
curl -s http://127.0.0.1:8300/v1/completions -H 'Content-Type: application/json' -d '{
  "model": "/data/models/yuhua/m3-pro",
  "prompt": "]~!b[The capital of France is",
  "max_tokens": 24, "temperature": 0}'
# " Paris. The city is located in the north of the country, on the banks of the Seine River. Paris is the"
```

## Accuracy (GSM8K, 64-shot)

With 64 shots a prompt is about 10k tokens (9.0k-11.9k, median 10.4k), so the indexer picks 128 of
about 650 blocks; up to 2048 tokens it selects every block and the score cannot see it. The server
is the kernel path with an FP8 KV cache, a bf16 index cache and prefix caching off:

```bash
export AITER_CUSTOM_AR_DISABLE_LAMPORT=1
export ATOM_M3PRO_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "fused"}'

python -m atom.entrypoints.openai_server \
  --model /data/models/yuhua/m3-pro \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --server-port 8300 \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 \
  --block-size 16 \
  --kv_cache_dtype fp8 \
  --index-cache-dtype bf16 \
  --level 0 \
  --no-enable_prefix_caching \
  --online_quant_config '{"layer_quant_config": {"*.experts": "mxfp4"}}'
```

lm-eval's stock `gsm8k` task (raw text, no BOS), greedy, all 1319 questions:

```bash
lm_eval --model local-completions \
  --model_args "model=/data/models/yuhua/m3-pro,tokenizer=/data/models/yuhua/m3-pro,base_url=http://127.0.0.1:8300/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
  --tasks gsm8k --num_fewshot 64
```

| Indexer reading | flexible-extract | strict-match |
|---|---|---|
| pool4 / max (default) | 0.9409 +- 0.0065 | 0.9393 +- 0.0066 |
| lse / lse (`{"block_score": "lse", "head_reduce": "lse"}`) | 0.9477 +- 0.0061 | 0.9469 +- 0.0062 |

The two readings are about one standard error apart. Greedy decoding is not bit-reproducible on
this stack (MoE reductions and the batch composition change between runs), so two runs of one
server differ by about as much.

## Accuracy (AIME 2026)

AIME 2026 I and II (30 problems), 0-shot with thinking, 16 samples per problem, on the kernel path
with an FP8 KV cache, a bf16 index cache and a 131k context (prefix caching and DSpark off):

```bash
export AITER_CUSTOM_AR_DISABLE_LAMPORT=1
export ATOM_M3PRO_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "fused"}'

python -m atom.entrypoints.openai_server \
  --model /data/models/yuhua/m3-pro \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --server-port 8300 \
  --max-model-len 131072 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 16384 \
  --gpu-memory-utilization 0.85 \
  --block-size 16 \
  --kv_cache_dtype fp8 \
  --index-cache-dtype bf16 \
  --level 0 \
  --no-enable_prefix_caching \
  --online_quant_config '{"layer_quant_config": {"*.experts": "mxfp4"}}'
```

The checkpoint's own `chat_template.jinja` is the BOS followed by the user text (no roles, no
thinking switch), so the client renders each prompt with MiniMax-M3's chat template instead
(`chat_template.jinja` of `amd/MiniMax-M3-MXFP4`, `thinking_mode="enabled"`, generation prompt
added): the system turn with the thinking instructions, the default developer turn, the user turn
and an assistant turn opened with `<mm:think>`. The user turn asks for the final answer in
`\boxed{}` (the number or expression only) and ends with `Problem: {problem}\nAnswer:`.

| | |
|---|---|
| Endpoint | `/v1/completions`, streamed; the client stops each reply at the first `]~b]`, `[e~[` or `]~!b[` |
| Sampling | temperature 1.0, top-p 0.95, up to 98,304 new tokens, 16 samples per problem, 16 concurrent |
| Scoring | the content of the last `\boxed{}` in the reply, compared as a string with the reference (no numeric normalization); avg@16 over the 480 samples |

| Indexer reading | avg@16 | pass@16 | maj@16 | Replies at the 98k cap | Median reply (tokens) |
|---|---:|---:|---:|---:|---:|
| pool4 / max (default) | **0.9563** (459/480) | 1.000 | 0.967 | 2 | 4930 |
| lse / lse (`{"block_score": "lse", "head_reduce": "lse"}`) | **0.9625** (462/480) | 1.000 | 0.967 | 1 | 4932 |

The two readings are within sampling noise. Problem 15 is the only one missed more than twice
(1 of 16 right in both runs); every other miss is one or two samples, on different problems in the
two runs. Scoring only the text after `</mm:think>` gives the same averages.

The prompt format matters more than either reading: the lse / lse server scored 0.8229 when the
prompt was the bare M3.1 chat layout (user and assistant turns only, no system or developer turn)
with a 32k cap; 82 of the 480 replies hit the cap and only 177 closed their thinking.

## Images

`/v1/chat/completions` takes images as `image_url` parts (a base64 data URL or an http(s) URL). Pro
is a base model, so the text after the image is a prompt to continue:

```bash
IMG=$(base64 -w0 photo.jpg)
curl -s http://127.0.0.1:8300/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "/data/models/yuhua/m3-pro", "max_tokens": 48, "temperature": 0,
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$IMG"'"}},
    {"type": "text", "text": "The image above is a photograph of"}]}]}'
```

The tower is Flash's (`atom/models/minimax_m3/vl.py`) at Pro's sizes: 32 layers 1280 wide, patch 14
with 3D rope, a 2x2 patch merge and a projector to 7168, replicated on every rank, with its unquantized
weights from the checkpoint's last shard. A 448x448 image is 256 tokens. Two things differ from Flash:
Pro's packed-module mapping keeps `q_proj` / `k_proj` / `v_proj` (its text attention uses those
names), so the tower keeps them unfused; and Pro's `preprocessor_config.json` gives `size` as a
2016x2016 box, which transformers' image processor rejects, so ATOM reads it as the pixel range from
56x56 to `max_pixels` (4064256). ATOM refuses images under sequence parallelism or the TR_HYBRID graph
mode (the `--level 3` default), as for Flash; the launch above has neither.

Checked: on a CPU, the tower, projector and merge match transformers' `MiniMaxM3VLVisionModel` and
`MiniMaxM3VLMultiModalProjector` on the real weights to a relative error of 3e-6. On the server
(greedy, with and without DSpark and prefix caching), the continuations name Grace Hopper from her
portrait, the Summer Palace's Tower of Buddhist Incense and a dahlia, also with the temple photo at 3x
(3184 prompt tokens); read "ATOM 4729" off a rendered image; put a red circle left of a blue square;
describe the two images of one request in order; and call a white image blank. Text requests give the
same responses as on the text-only model (one request at a time, 33-35 of 40 identical against 34-36
between two runs of either). Video input is untested.

## Prefix caching

On (the server default). The kernel path keys everything on positions and the block table, and the
index keys live in the same blocks as K and V, so a prefill that starts after cached blocks runs like a
later chunk of a chunked prefill. Checked on the kernel path with fp8 caches, with and without DSpark:

* Prompts of ~12k tokens listing random 8-hex keys and values, with the answer in one line of the
  cached part, so the selection has to find that line's block among ~750: 34 of 34 right over cold
  requests, full hits (12064 of 12067 tokens cached), half hits (the second half prefilled against the
  cached first half) and 24 more questions over a cached list.
* A cached and an uncached run of the same prompt give the same answer tokens with the same logprobs
  (to three decimals). They part ways after the answer, at tokens whose top two are within 0.5 of each
  other: the last block goes through other GEMM shapes, and decoding is not bit-reproducible anyway.
* GSM8K with one fixed few-shot prefix for every question (an lm-eval task that includes `gsm8k` with
  `fewshot_config: {sampler: first_n}`; 89% of the prompt tokens cached, 16 concurrent, DSpark on)
  scores within run-to-run noise of the same prompts without prefix caching (40 questions turn
  wrong, 51 turn right). Stock GSM8K, whose few-shot examples differ per question, hits almost nothing.

The checks above ran on the separate K and V caches. For the K-only cache the Agentic recipe
(`Agentic-MiniMax-M3.1-Pro.md`) serves, with day0 prefill, the compressed index cache, the indexer
split over the ranks and the FlyDSL / aiter decode, three checks cover hits:

* `tests/test_minimax_m3_pro_prefix_hit.py`: a request admitted through the block manager's prefix walk
  (a full hit, a half hit) prefills only its uncached rows. It must select the blocks a cold prefill
  selects, write the same cache bytes for its new tokens, leave the shared blocks untouched and give the
  same attention output, and so must a chunked cold prefill and a rejected, then accepted, k = 7 verify
  after a hit. Without a GPU the small variants run in Triton's interpreter; the served variants need a
  GPU, `amd_kernels` on `PYTHONPATH` and an aiter whose `pa_decode_gluon` takes `value_rows`.
  `tests/test_scheduler_spec_prefix_publish.py` checks that DSpark decode publishes a block only once
  all of its KV is final.
* `scripts/m31pro/pfx_probe.py probe` against a running server (started as the Agentic recipe does,
  without LMCache and without `--spec-decode-acceptance-length`): key / value questions over 32k and 64k
  lists, each request served from the cache (`hit`, `half`, and `turn2` on blocks that decode published)
  against the same token ids prefilled cold, at concurrency 1 and 8. It passes when every answer is
  right and hits part from their cold twins only at near-ties, at most 2 more often than two cold runs
  part from each other.

  ```bash
  python3 scripts/m31pro/pfx_probe.py probe --port 8000 --model "${MODEL_PATH}" --out probe.jsonl --lengths 32000
  ```
* GSM8K behind one shared 64-shot prefix (`scripts/m31pro/lm_eval_tasks/gsm8k_pfx.yaml`), on one server
  with prefix caching and one with `--no-enable_prefix_caching`, then the paired comparison (exact
  McNemar on the questions that flip, the cached share and the acceptance length from `/metrics`):

  ```bash
  python3 scripts/m31pro/pfx_probe.py snapshot --port 8000 --out before.json
  lm_eval --model local-completions \
    --model_args "model=${MODEL_PATH},tokenizer=${MODEL_PATH},base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
    --include_path scripts/m31pro/lm_eval_tasks --tasks gsm8k_pfx --num_fewshot 64 --log_samples --output_path on
  python3 scripts/m31pro/pfx_probe.py snapshot --port 8000 --out after.json
  # the same against the server without prefix caching (--output_path off), then
  python3 scripts/m31pro/pfx_probe.py pair --on on/*/samples_gsm8k_pfx_*.jsonl \
    --off off/*/samples_gsm8k_pfx_*.jsonl --on-metrics before.json after.json
  ```

## CPU KV offload

Prefix blocks evicted from HBM can be kept in host DRAM and loaded back instead of recomputed, through
ATOM's in-process LMCache connector. Pro resolves to its `m3` PAGE layout
(`atom/kv_transfer/offload/hybrid/m3/`), which saves every byte of a scheduler block: per layer the K
rows, their fp8 scale plane and the index keys (and the V cache, unless the K-only cache, `"k_only": true`
in `ATOM_M3PRO_VARIANTS`, leaves none),
then the DSpark draft's own K / V pages. No Pro-specific code is involved; the layout comes from the
attention builder's transfer regions.

Add to the server's environment:

```bash
export PYTHONHASHSEED=0
export LMCACHE_LOCAL_CPU=True
export LMCACHE_MAX_LOCAL_CPU_SIZE=160          # GiB per TP rank
export LMCACHE_CHUNK_SIZE=256
export LMCACHE_LOOKUP_SERVER_WORKER_IDS=0,1,2,3,4,5,6,7
export ATOM_PREFIX_CACHE_POLICY=slru
export ATOM_PREFIX_CACHE_PROTECTED_RATIO=0.5
export LMCACHE_CACHE_POLICY=ATOM_SLRU
```

and to its arguments `--kv-transfer-config '{"kv_connector":"lmcache_offload","kv_role":"offload"}'`.
Prefix caching must stay on. `PYTHONHASHSEED=0` makes every rank key a prompt alike; without it the
ranks never agree on a hit. The lookup ids name all eight ranks, so a hit counts only as far as every
rank still holds it.

Host memory, in GiB (`LMCACHE_MAX_LOCAL_CPU_SIZE` is GiB): each rank stores its own copy of the cache
(Pro's one KV head is the same on every rank), so the pool is `LMCACHE_MAX_LOCAL_CPU_SIZE` times 8:
1,280 GiB (1.37 TB) at 160 GiB. At 160 GiB a rank holds about 5.9M tokens (463,744 bytes per 16-token
block, K-only cache, fp8 index cache, DSpark draft). Measured once on the server above, at startup with
the pool allocated and before it filled (process RSS and `free -g`): each TP process held about 260 GiB,
the pool and about 100 GiB of its own, and the node showed 2,171 of 2,751 GiB used, 583 GiB available.
InferenceX allows CPU DRAM only, up to 3 TB per MI355X server.

Measured on the 15-minute agentic proxy (`scripts/m31pro/proxy_bench.sh 32`: an aiperf replay of 64
`semianalysisai/cc-traces-weka-062126` sessions capped at 128k,
DSpark k=7 at a forced acceptance length of 4.0) with
`ATOM_M3PRO_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "fused", "k_only": true}'`, both arms on
one tree; the HBM pool is 18,600 blocks (about 298k tokens):

| c32 | tok/s/GPU | p90 E2E-normalized interactivity | failed | requests | prompt tokens served from cache |
|---|---:|---:|---:|---:|---:|
| offload off | 1037.5 | 0.60 | 0/157 | 157 | 0.01 |
| offload on | 3681.0 | 2.25 | 0/496 | 496 | 0.95 (0.12 HBM, 0.83 loaded from CPU) |

Real request rows per decode forward (`atom:decode_batch_size`), share of forwards:

| batch | 1 | 2 | 3-4 | 5-8 | 9+ | mean | forwards |
|---|---:|---:|---:|---:|---:|---:|---:|
| offload off | 0 | 0 | 41.3% | 58.6% | 0 | 4.96 | 4,369 |
| offload on | 0.4% | 1.3% | 46.7% | 51.6% | 0 | 4.56 | 21,426 |

Offload does not raise the running batch: a running request still needs its whole context in HBM, and
about five contexts of 60k tokens fill the pool. It removes the recompute of evicted prefixes, so the
same batch runs five times as many decode forwards in the window.

Accuracy with offload on (the server above at max-model-len 16384, the GSM8K section's lm-eval
command): 0.9363 +- 0.0067 flexible-extract over all 1319 questions, against 0.9416 +- 0.0065 with
offload off. Stock GSM8K shares no prefix, so nothing is loaded back there; restored KV was checked by
running the first 450 questions twice on one server, the second pass with 4.60M of its 4.66M prompt
tokens loaded from CPU: 0.9667 cold, 0.9689 restored, 257 of 450 responses identical.

## Speculative decoding (DSpark)

`--method dspark --num-speculative-tokens 7` drafts with the checkpoint's own 5-layer DSpark model
(`language_model.model.dspark.*`, block size 7, fed by target layers 4/19/35/51/67). It changes no
answer beyond the run-to-run noise: on GSM8K a run with the draft agrees with one without it on as
many questions as two runs without it (41% identical responses, 68 questions flipping either way,
against 40% and 76). It
accepts 43% of the drafted tokens on GSM8K (4.0 tokens per target forward), the same with prefix
caching on, where the draft's context for cached blocks comes from the cache too: copying lines out of
an 8k-token list, it accepts 40% cold and 42-43% with the list cached.

### Decode graphs at large batch

A decode step replays a captured graph when its batch fits the capture ladder
(`--cudagraph-capture-sizes`, default `[1..8, 16, 32, 48, 64, 128, 256, 512]`),
cut at startup to what the scheduler can form: `min(max-num-seqs,
max-num-batched-tokens // (k + 1))`. A DSpark verify graph holds `k + 1` rows a
sequence, and the draft's block pass is captured at the same batch sizes. At the
agentic proxy's c32 point (`--max-num-seqs 64`, 16384 tokens, k=7) that is
`[1..8, 16, 32, 48, 64]`, so every decode batch replays.

`atom:forward_steps_total` on `/metrics` counts forwards by `model` (target,
draft), `phase` and `cudagraph`; eager decode steps are
`atom:forward_steps_total{model="target",phase="decode",cudagraph="false"}`.

## What is verified, and what is not

Settled by running them (`ATOM_M3PRO_VARIANTS`, a JSON string or file re-read between requests):

* YaRN magnitude: the config's `attention_factor` (1.0, i.e. none) as transformers computes it,
  not aiter's `0.1 ln(factor) + 1` default (GSM8K on a 200-question hard subset: 150 vs 117 correct).
  aiter's YaRN frequencies match transformers' to 6e-8.
* V is the *normed* K without its rope dims: the raw `k_proj` output instead generates only newlines.
* Softmax scale `head_dim ** -0.5`: GSM8K is flat from 0.85x to 1.6x of it, so nothing argues for another.

Short prompts cannot test the indexer: up to 2048 tokens (128 blocks of 16) every block is selected,
so selection is the identity (hence the 64 shots of the GSM8K run). A 14k-token key-value retrieval
(12 lookups, chunked prefill, 4 concurrent sequences) is 12/12 with the indexer and with dense
attention, and 0/12 with a selector that
ignores the scores; every alternative reading of the unspecified parts (head reduction
sum/max/mean/lse, `softmax_pool` 4-centroid block scores with `index_compress_query`, lookahead
source `ffn_in`/`resid`, no lookahead, no indexer rope, 0 init / 1 local blocks) also scores 12/12,
so that test cannot tell them apart.

### The indexer, measured against the dense attention

A trained indexer is a predictor of the dense attention, so each reading of it is scored by how well
it predicts the dense attention instead (`atom/model_ops/minimax_m3/pro_diag.py`, a developer tool):

```bash
ATOM_M3PRO_VARIANTS='{"diag": true, "diag_tag": "code"}'   # dense attention + indexer readings
# one prefill chunk of >= 4104 tokens (--max-num-batched-tokens >= its length), 1 token out;
# one JSON line per layer in $ATOM_M3PRO_DIAG_DIR/<diag_tag>.jsonl
```

For queries at positions >= 4096 it records, per reading (index q/k source x rope x relu x block score x
head reduction): the share of the oracle's headroom (top-128 by the true block mass, over the forced
init/local blocks) that the reading's top-k captures, also at 16/32/64 blocks, where readings that
all look alike at 128 separate; the Spearman correlation of its block scores with the block mass; and,
with relu off, the KL divergence of `softmax(score)` from the block mass. All of it was run on the
real weights, on a CPU, over 14.5k-token documents (the whole 68-layer model runs on a CPU in four
minutes for a short prompt, `' Paris'`), layers 0-11:

* rope on the index q/k is right (rank correlation 0.99 against 0.33 at layers 0-1), relu is off.
* The index scores are calibrated attention logits: `softmax(iq . ik / sqrt(128))` over the *tokens*
  predicts the dense block mass with a KL of 0.010 nats at layers 0-1, with its minimum at scale 1
  (best on a factor-2 grid from 1/16 to 4). The block score that follows from it is the log-sum-exp of
  the 16 token scores (`block_score: "lse"`); a hard max over them is 10x worse in KL (0.10) and the
  mean 40x worse, and no softmax-pooled 4-centroid reading (any grouping, max/lse/mean, pooling
  temperature 0.09-8) gets below 0.021.
* The head reduction barely matters: log-sum-exp over the heads and the mean of the per-head block
  distributions rank alike (rank correlation 0.950 and 0.949 over layers 2-11); a plain sum is 0.003 lower.
* The lookahead source matters, a little: the FFN norm's output of layer L-2 (what the first bring-up
  used) is clearly worse than the alternatives (rank correlation 0.90 against 0.95 over layers 2-11, and
  worse at every layer from 3 on). Any state of layers L-2 / L-1 put through *layer L's own input
  norm*, or the layer's own input, captures the same share of the dense attention to within 0.001. At
  layer 2 alone the layer's own input is clearly better than layer 0's state, and nothing explains
  layer 2's attention well (rank correlation <= 0.85 for every source). At layers 3-9 the best of 61
  candidates (residual of layer L-2 / L-1, before or after its attention, through no norm or one of four
  layers' norms; query and key from the same state or one of them from the layer's own input) calibrates
  1.9-7.3x better (KL 0.02-0.04) than the own input (0.06-0.20), but the winner differs from layer to
  layer (a key from an earlier state through a post-attention norm is common), so the exact recipe of
  `sparse_indexer_lookahead_source: pre_ffn_norm` stays unproven.

Share of the oracle's headroom captured by the top-128 / top-32 selection (layers 2-11; four 9.5k-14.5k
token documents, ATOM source, ATOM docs, a Chinese/English GitHub issue and GSM8K few-shot prompts,
1920 queries):

| reading | top-128 | top-32 |
|---|---|---|
| first bring-up: token max, summed over heads, FFN-norm output of L-2 | 0.951 | 0.900 |
| token max, summed over heads, no lookahead | 0.960 | 0.906 |
| token log-sum-exp, lse over heads, residual of L-2 through layer L's input norm | **0.978** | **0.953** |
| token log-sum-exp, lse over heads, no lookahead | 0.979 | 0.958 |
| token log-sum-exp, lse over heads, FFN-norm output of L-2 | 0.956 | 0.921 |

The ordering is the same on each of the four documents: log-sum-exp over the tokens beats the hard max
by 1.7-1.9 points at 128 blocks and 4-5 points at 32 (0.6 / 2 with the FFN-norm source), and the
FFN-norm source is the worst with either block score. The lookahead against the layer's own input is a
toss-up (0.978 / 0.953 against 0.979 / 0.958); it is the default because the config asks for a
lookahead, not because it is measurably better here.

The default block score and head reduction are not the best predictor above but those of a PyTorch
reference implementation of the model: each block's 4 groups of 4 index keys softmax-pooled with
`index_compress_query` at temperature 1/sqrt(128), the block scored by its best centroid, the heads
by their max (`block_score: "pool4"`, `head_reduce: "max"`, `pool_scale: 0.0884`). The
dense-attention calibration is a proxy; a reference implementation is closer evidence of what the
indexer was trained with. The rest of the reference agrees with the readings settled above
(YaRN magnitude, V from the normed K, rope on the index q/k, the lookahead source, the forced
blocks). The measured readings stay available through the variants (`{"block_score": "lse",
"head_reduce": "lse"}`, or the first bring-up's `{"block_score": "max", "head_reduce": "sum",
"lookahead_src": "ffn_in"}`). Both benchmarks are within noise between the two: GSM8K (64-shot, all
1319) 0.941 for the reference's reading and 0.948 for lse / lse, AIME 2026 0.9563 and 0.9625.

## Kernels

`ATOM_M3PRO_VARIANTS='{"kernels": true}'` runs prep (norms + rope + cache writes), the block scores
and the sparse attention as Triton kernels (a handful of launches per layer, no host sync past the
sequence lengths) instead of ~100 torch ops. `attn_splits` (programs per token over the selected
blocks, 0 = by batch size) and `attn_nblk` (1/2/4/8 blocks per step) tune the launch.

`{"kernels": true, "attn": "aiter"}` keeps prep and the scores and runs the attention on aiter's
paged-attention decode (`pa_decode_gluon` through ATOM's `run_pa_decode`, persistent dispatch, as
ATOM's M3 sparse path does): every token becomes a sequence of its own whose page table lists its
selected blocks, full blocks first and its own block last, so that the context length stops at the
token. aiter reads K and V from separate caches, so prep also writes V (K's 192 un-roped dims) into
the first 192 rows of the V cache the builder allocates anyway (256 wide, so no extra memory), and
the kernel runs with Q/K 256 and V 192 (aiter supports unequal Q/K and V head sizes with one KV head).

With `--kv_cache_dtype fp8`, prep quantizes K and V per token (one fp32 scale each, in ATOM's
`k_scale` / `v_scale` planes). The Triton attention dequantizes K (V is its tail, on K's scale); the
aiter attention passes the scales to the kernel and computes in fp8 (`attn_fp8_compute: false`
computes in bf16). The torch path writes and reads the same layout. The index cache follows the KV
cache's dtype unless `--index-cache-dtype` says otherwise; an fp8 index cache holds the (normed)
index keys cast to fp8 and clamped to its range, unscaled, as M3's does. It halves what the scores
read, which at long contexts is more than the attention reads (all of a sequence's index keys, against
the K / V of the 128 selected blocks).

`"topk": "aiter"` selects the blocks with aiter's per-row top-k (`top_k_per_row_decode`, every
token a row whose length is its causal block count, short rows padded with -1) instead of
`torch.topk` over the block table's width. `"topk": "fused"` (with `"attn": "aiter"` and a
power-of-two top-k, which Pro's 128 is) selects and builds the aiter page tables in one Triton
kernel: `tl.topk` over the scores' order-preserving integer bits packed with the block index (ties
go to the lower block), then the same rows rule as the separate rows kernel.

`"topk": "split"` builds the same tables bit for bit, and its cost follows the context. A decode
graph holds the block table's full width (max-model-len / 16 blocks: 8192 at 131072), and
`"fused"` sorts all of it for every token, about 29 us per layer whatever the context. `"split"`
works in two steps:
- Spans of 1024-block chunks reduce each row to its 128 best keys in separate programs, and the
  last span program to finish merges them. Spans past a row's own block exit at once.
- The scores fill only each token's causal blocks: those selections read no further.

It is the recommended setting. Measured on 8x MI355X, max-model-len 131072:
- Per decode step, scores plus selection take 0.68 instead of 2.35 ms at bs 1 (1k context), and
  1.15 instead of 4.08 ms at bs 64.
- TPOT drops 11–14% at 1k/1k (c 1–64) and 9–12% at 8k/1k.
- GSM8K is unchanged: the same tables.
- At a 1M context (65536 blocks) the sort alone is 0.7–0.8 ms per layer for `"fused"` and
  3–66 us for `"split"`.

The block scores run as a tiled kernel (4 or 8 blocks x 16 tokens per program, one `tl.dot`); the
softmax-pool weights depend on the keys alone, so the pooled scores are the token dots weighted per
group of 4, in the same tile. The module hands the four
projections to the kernels as they are (no packed copy), the lookahead input is normed by the
reading layer's own (aiter) input-norm kernel when it is put, and projections that share an input
share its FP8 quantization.

On decode-only steps the kernel path reads nothing back from the host: it runs every row of the
step (padding rows have slot -1 and write nothing), scores the block table's full width and always
selects, which is what a CUDA graph needs. So with the kernels on, the server can drop
`--enforce-eager` (`--level 0` stays) and capture decode graphs; the torch path cannot, and says
so if a capture reaches it.

The kernels cover the default reading and its `block_score` max / lse and `head_reduce` sum / lse
alternatives; the other switches (`index_rope: false`, `v_src: "pre"`, gptj rope, the diagnostic) send
the layer back to torch. `tests/test_minimax_m3_pro_kernels.py` checks them against the torch
reference on a CPU (Triton interpreter, fp32 and bf16, aiter through torch stand-ins) and on one
MI355X (fp8 caches, aiter's own kernels, a decode graph replayed on another batch).

`"prep": "aiter"` (with `"kernels": true`; default `"triton"`) runs prep as aiter's
`fused_qk_norm_rope_cache_mqa` (one HIP launch, a 32-lane group per head row) instead of the Triton
prep, falling back to Triton where the installed aiter lacks the op. Same math and rounding points:
K, V, index K and the fp8 codes and scales match Triton's; q / index q differ on ~1e-5 of the
values by one bf16 step (Triton's own fma contraction; both within that of the torch reference).
It takes any number of local heads, which removes the Triton prep's 32-heads-per-rank limit on the
kernel path (TP 2 / 1: 64 / 128 local heads); the rest of TP < 8 is still unverified. GPU time per
call in a CUDA graph (MI355X, Triton -> aiter, us):

| local heads, KV / index | 1 token | 128 tokens | 1024 tokens | 8192 tokens |
|---|---|---|---|---|
| 16, bf16 | 6.2 -> 3.9 | 6.8 -> 4.9 | 15.5 -> 10.7 | 79.7 -> 48.4 |
| 16, fp8 | 6.9 -> 4.9 | 7.4 -> 6.0 | 15.7 -> 13.2 | 85.2 -> 51.1 |
| 128, bf16 | 10.2 -> 3.9 | 11.9 -> 7.3 | 48.1 -> 27.5 | 439 -> 214 |
| 128, fp8 | 10.7 -> 4.9 | 12.2 -> 8.3 | 49.6 -> 28.5 | 440 -> 209 |

End to end (TP8, the fused-selection configuration, fp8 KV / bf16 index cache, graphs, 1k in /
512 out) it is ~1%: TPOT at concurrency 1 / 8 / 32 12.73 -> 12.56, 14.82 -> 14.68, 20.66 -> 20.64
ms, a bs1 decode step 14.29 -> 14.13 ms; GSM8K (20-shot, all 1319) 0.9356 against 0.9303 / 0.9363
for two baseline runs. `tests/test_minimax_m3_pro_prep_aiter.py` checks it against the Triton prep
and the torch reference.

End to end on 8x MI355X (TP8, `--max-num-batched-tokens 16384`, 256 tokens out, greedy; TPOT in
steady decode, ms per token):

| attention | KV / index | 14k retrieval | TPOT 1k bs1 | 1k bs16 | 8k bs1 | 8k bs4 | TTFT 8k bs1 |
|---|---|---|---|---|---|---|---|
| torch reference, eager | bf16 | 12/12 | ~140 | | | | |
| kernels, Triton attention, eager | bf16 | | ~95 | | | | |
| kernels, Triton attention, aiter top-k, graphs | bf16 | 12/12 | 14.92 | 18.14 | 15.22 | 16.23 | 759 ms |
| kernels, aiter attention, aiter top-k, graphs | bf16 | 12/12 | 14.80 | 18.11 | 14.92 | 16.06 | 789 ms |
| kernels, aiter attention, fused selection, graphs | bf16 | 12/12 | 11.85 | 14.75 | 11.76 | 12.62 | 574 ms |
| same | fp8 | 12/12 | 11.33 | 14.08 | 11.26 | 12.13 | 504 ms |
| same, tuned launch shapes | fp8 | 12/12 | 11.10 | 13.86 | 11.02 | 11.88 | 502 ms |

The GSM8K and AIME 2026 runs of the Accuracy sections use the fused-selection configuration with an
fp8 KV cache and a bf16 index cache. Of the 2.9 ms between the aiter top-k
and the fused-selection rows at 1k bs1, 0.6 ms is the fused selection and the tiled scores, and 2.4 ms
is outside the attention kernels (the lookahead norm, the shared quantization, the removed copies).

GPU time per layer of the decode kernels, replayed from a CUDA graph (`bench_attn.py --graph` and
`--tune`, batch 1-16, context 1k-16k): prep 6.3 us; block scores 2.9-12.5 us (the per-block kernel:
7.3-20.0); the fused selection 5.7-6.2 us (aiter's top-k and the rows kernel: 8.4-13.1); aiter
attention 9.5-11.5 us in bf16 and 7.8-8.1 us in fp8 (Triton attention: 17.6-18.7 and 20.6-21.1).
The tuned launch shapes are 8 warps for the selection, 4 blocks per score tile up to 16 tokens (8
above) and aiter's partition count capped at one program per CU (4 partitions instead of 8 at 64
tokens: 15 us against 21 at 1k context in bf16). On a 4096-token prefill chunk at 16k context the
scores take 1.1 ms (per-block kernel: 3.2 ms) and the aiter attention 0.73 ms in bf16, 0.26 ms in
fp8 (Triton: 0.82 / 1.04).

Where a decode step's GPU time goes (torch profiler, bf16, bs1, 1k context, 13.5 ms per step under
the profiler): Pro's attention kernels 22% (aiter attention 10%, prep, scores and selection 12%),
the MXFP8 / router GEMMs 24%, the MoE with its routing and elementwise ops 26%, the all-reduces 11%,
the activation quantization 11%, norms and the shared expert's activation 4%.

## Fused MXFP8 activation quantization

`fuse_mxfp8_quant` (a `ProVariants` field, on by default; `{"fuse_mxfp8_quant": false}` in
`ATOM_M3PRO_VARIANTS` turns it off, and `diag` does too) has the ops that produce the inputs of the
native MXFP8 (group-32) linears write them as MXFP8 themselves -- E4M3 values and E8M0 scales, the
bytes `quantize_fp8` would give -- instead of a separate `quantize_fp8` launch before the GEMM:

- the all-reduce + input norm writes the q / k / indexer projections' input (layer 0: the norm);
- the all-reduce + post-attention norm writes the dense MLP's or the shared expert's gate_up input,
  and still writes the bf16 rows the router and the routed experts read;
- the lookahead norm in `ProLookaheadState.put` writes the reading layer's indexer input;
- the SwiGLU writes down_proj's input (one Triton kernel).

o_proj's input, written by the paged attention, is still quantized on its own: `quantize_fp8` per
layer goes from 5 launches to 1.

The norms need aiter's MXFP8 norm outputs ([zhuyuhua-v/aiter-M3#2](https://github.com/zhuyuhua-v/aiter-M3/pull/2): the custom all-reduce
MXFP8 epilogue, `add_rmsnorm_mxfp8` for the RCCL route and an E8M0 mode of
`fused_qk_rmsnorm_group_quant`). On an aiter without them the norms keep the unfused path (a log
line at startup says which aiter features were found) and the SwiGLU is still fused. Where the
all-reduce takes a route the MXFP8 epilogue does not cover (the reduce-scatter 2-stage kernel above
80 rows, gluon, Lamport, quick-reduce), that norm runs unfused too. Every fused output is checked
byte for byte against the unfused pair (`tests/test_m3pro_mxfp8_fusion.py`, TP1 and TP8, eager and
in CUDA graphs).

End to end with the server command above (TP8, fp8 KV cache), off -> on:

| | c=1 | c=8 | c=32 |
|---|---|---|---|
| TPOT, ms | 12.73 -> 12.29 | 14.82 -> 14.35 | 20.66 -> 20.09 |

Decode step under the torch profiler: bs1 14.29 -> 13.24 ms, bs32 20.03 -> 18.92 ms. GSM8K (20-shot,
all 1319): 0.9303 and 0.9363 off (two runs), 0.9318 on. Greedy outputs differ between the two no
more than two runs of the same setting do.

Not done: TP other than 8. Untested: video input, prefix caching on the torch reference path.
