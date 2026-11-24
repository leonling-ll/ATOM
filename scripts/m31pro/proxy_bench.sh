#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# MiniMax-M3.1-Pro agentic proxy benchmark: one aiperf replay of a fixed-seed
# session subset of semianalysisai/cc-traces-weka-062126 against one TP8 server.
#
# Usage: bash scripts/m31pro/proxy_bench.sh <CONC> [OUT_DIR]
#
# Run it inside the ATOM image (aiperf from agentx-harness on PATH) with this
# tree importable, under the node's end-to-end lock. It
#   1. writes the session subset (trace_subset.py) to OUT_DIR/traces,
#   2. starts the server with the Pro recipe's fast configuration (RECIPE=pro)
#      or the Agentic recipe's launch (RECIPE=agentic), plus DSpark at
#      SPEC_TOKENS draft tokens and a forced acceptance length: Pro's golden AL
#      for that draft length unless ACCEPT_LENGTH is set,
#   3. prewarms the server (PREWARM, below) so first-use compiles do not land
#      in the measured window,
#   4. replays the subset at concurrency CONC for DURATION seconds,
#   5. stops the server and prints, from aiperf's per-request records, the
#      90th-percentile E2E-normalized interactivity, tok/s/GPU, the failed
#      request rate and how many replies differ in length from the trace's
#      output length (also written to OUT_DIR/summary.json). The scenario sends
#      ignore_eos, so that count should be 0 on any weights.
#
# Environment (defaults in brackets):
#   MODEL            checkpoint [/data/models/yuhua/m3-pro]
#   TRACES           traces.jsonl of the corpus [fetched through the HF cache]
#   SEED             subset and aiperf seed [42]
#   MAX_CONTEXT      session cap, input + output tokens [131072]
#   NUM_SESSIONS     sessions kept [64]
#   DURATION         replay seconds; aiperf's scenario floor is 900 [900]
#   SPEC_TOKENS      DSpark draft tokens per step; 0 serves without the draft [7]
#   THINKING         thinking mode whose golden AL curve applies [thinking_on]
#   ACCEPT_LENGTH    forced DSpark acceptance length [the golden AL of
#                    SPEC_TOKENS and THINKING; refused where there is none]
#   RECIPE           pro: recipes/MiniMax-M3-Pro.md's fast configuration;
#                    agentic: the launch in recipes/Agentic-MiniMax-M3.1-Pro.md's
#                    "1. Start the ATOM server", every env and flag of it
#                    (tests/test_m31pro_proxy_bench.py diffs the two) [pro]
#   CHAT_TEMPLATE    agentic: --chat-template file; unset, a checkpoint
#                    chat_template.jinja without the thinking_mode switch (the
#                    1 KB stub) is refused []
#   MAX_NUM_SEQS     [16 below CONC 8, else 2 x CONC]
#   MAX_MODEL_LEN    [MAX_CONTEXT + 1024: room for the chat template]
#   PORT             server port [8300]
#   SERVER_LOG       the server's log, copied to OUT_DIR [/app/logs_claude/atom_server.log]
#   GPU_MEM_UTIL     --gpu-memory-utilization [pro 0.94, agentic 0.95]
#   SERVER_EXTRA     more server flags, split on spaces, e.g. --load_dummy=xavier []
#   ATOM_CUSTOM_AR_POOL_BYTES  custom all-reduce input pool [134217728]
#   GPTOSS_SWIGLU_MXFP4_BF16_BOUND  agentic: bf16 MoE activation bound [1040]
#   ATOM_M3PRO_VARIANTS  Pro attention variants: pro replaces its variants with
#                    it; agentic merges it onto the recipe's, so a caller adds
#                    keys such as {"index": "fused", "ar_fused_rows": 128}
#                    [RECIPE's]
#   PREWARM          1: after the server is ready and before the client, send
#                    prompts of PREWARM_PROMPT_TOKENS tokens (one at a time,
#                    max_tokens 1: the prefill chunk sizes) and then, per
#                    PREWARM_BATCHES entry, that many concurrent short prompts
#                    that decode 64 tokens (the decode graph sizes), and wait
#                    for every reply; a failed reply stops the run. Prompt
#                    lengths count words, so they are approximate. [1 for
#                    RECIPE=agentic, else 0]
#   PREWARM_PROMPT_TOKENS  [256,512,1024,2048,4096,8192,16384,24576]
#   PREWARM_BATCHES  [every capture size of the agentic ladder up to
#                    MAX_NUM_SEQS]
#   PROXY_DRY_RUN    1: print the acceptance length, the server environment and
#                    flags, and exit before anything runs
#   With RECIPE=pro the server inherits the rest of the environment, so CPU
#   offload's LMCACHE_* settings go there and its --kv-transfer-config in
#   SERVER_EXTRA. Quick all-reduce is off in every run.

set -euo pipefail

CONC="${1:?usage: proxy_bench.sh <CONC> [OUT_DIR]}"
OUT_DIR="${2:-$HOME/m31pro-bench/c${CONC}_$(date +%m%d_%H%M)}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

MODEL="${MODEL:-/data/models/yuhua/m3-pro}"
SEED="${SEED:-42}"
MAX_CONTEXT="${MAX_CONTEXT:-131072}"
NUM_SESSIONS="${NUM_SESSIONS:-64}"
DURATION="${DURATION:-900}"
SPEC_TOKENS="${SPEC_TOKENS:-7}"
THINKING="${THINKING:-thinking_on}"
RECIPE="${RECIPE:-pro}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_CONTEXT + 1024))}"
PORT="${PORT:-8300}"
TP=8
PREWARM="${PREWARM:-$([ "$RECIPE" = agentic ] && echo 1 || echo 0)}"
PREWARM_PROMPT_TOKENS="${PREWARM_PROMPT_TOKENS:-256,512,1024,2048,4096,8192,16384,24576}"
if [ -z "${MAX_NUM_SEQS:-}" ]; then
    MAX_NUM_SEQS=$((2 * CONC))
    [ "$MAX_NUM_SEQS" -ge 16 ] || MAX_NUM_SEQS=16
fi

AGENTIC_CAPTURE_SIZES=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,48,64,128,256,512
if [ -z "${PREWARM_BATCHES:-}" ]; then
    PREWARM_BATCHES=
    for n in ${AGENTIC_CAPTURE_SIZES//,/ }; do
        [ "$n" -le "$MAX_NUM_SEQS" ] && PREWARM_BATCHES="${PREWARM_BATCHES:+$PREWARM_BATCHES,}$n"
    done
fi

# golden_al K THINKING: MiniMax-M3.1-Pro's golden acceptance length for the
# bundled DSpark draft at K draft tokens, from InferenceX's golden AL file for
# minimax-m3.1-pro (Inferact/moo, moo-tracking, infx/golden_al_distribution).
# Only the thinking_on curve exists; anything else is refused.
golden_al() {
    if [ "$2" != thinking_on ]; then
        echo "proxy_bench: no golden AL curve for $2; set ACCEPT_LENGTH" >&2
        return 1
    fi
    case "$1" in
    1) echo 1.75 ;;
    2) echo 2.23 ;;
    3) echo 2.53 ;;
    4) echo 2.71 ;;
    5) echo 2.89 ;;
    6) echo 2.94 ;;
    7) echo 2.94 ;;
    *)
        echo "proxy_bench: no golden AL for $1 draft tokens; set ACCEPT_LENGTH" >&2
        return 1
        ;;
    esac
}

read -r -a server_extra <<<"${SERVER_EXTRA:-}"
spec_args=()
if [ "$SPEC_TOKENS" -gt 0 ]; then
    ACCEPT_LENGTH="${ACCEPT_LENGTH:-$(golden_al "$SPEC_TOKENS" "$THINKING")}" || exit 1
    spec_args=(--method dspark --num-speculative-tokens "$SPEC_TOKENS"
        --spec-decode-acceptance-length "$ACCEPT_LENGTH")
fi
SERVER_LOG="${SERVER_LOG:-/app/logs_claude/atom_server.log}"

# --- server environment and flags ---
export AITER_LOG_LEVEL=WARNING
export AITER_QUICK_REDUCE_QUANTIZATION=NONE
export KV_CACHE_DTYPE=fp8
recipe_args=()
case "$RECIPE" in
pro)
    export AITER_CUSTOM_AR_DISABLE_LAMPORT=1
    export ATOM_CUSTOM_AR_POOL_BYTES="${ATOM_CUSTOM_AR_POOL_BYTES:-134217728}"
    default_variants='{"kernels": true, "attn": "aiter", "topk": "fused"}'
    export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"
    ;;
agentic)
    # ar_lamport_rows needs the Lamport path.
    unset AITER_CUSTOM_AR_DISABLE_LAMPORT
    recipe_variants='{"kernels": true, "attn": "aiter", "topk": "split", "prep": "aiter", "index_split": true, "score_tq": 8, "ar_lamport_rows": 16, "k_only": true, "k_plain_fp8": true, "prefill_attn": "day0", "index_split_decode": 64, "moe_route_glue": true, "index_compress": true}'
    default_variants=$(python3 -c 'import json, sys
print(json.dumps({**json.loads(sys.argv[1]), **json.loads(sys.argv[2] or "{}")}))'         "$recipe_variants" "${ATOM_M3PRO_VARIANTS:-}")
    unset ATOM_M3PRO_VARIANTS
    export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.95}"
    export AITER_REUSE_IDENTICAL_COMM_GROUPS=1
    export ATOM_CUSTOM_AR_POOL_BYTES="${ATOM_CUSTOM_AR_POOL_BYTES:-134217728}"
    export GPTOSS_SWIGLU_MXFP4_BF16_BOUND="${GPTOSS_SWIGLU_MXFP4_BF16_BOUND:-1040}"
    export ATOM_WARMUP_TOKENS=auto
    export ATOM_ONLINE_QUANT_STREAMING=1
    export ATOM_CACHE_AWARE_ADMISSION=1
    export ATOM_KV_OFFLOAD=lmcache
    export LMCACHE_LOCAL_CPU=True
    export LMCACHE_MAX_LOCAL_CPU_SIZE=184
    export LMCACHE_CHUNK_SIZE=256
    export PYTHONHASHSEED=0
    # Set but empty records no bucket: every prefill runs eagerly.
    export ATOM_TOKEN_GRAPH_SIZES="${ATOM_TOKEN_GRAPH_SIZES-256,512,1024,2048}"
    # Graphs between 8, 16 and 32 sequences: a step replays the smallest graph
    # that holds its sequences, and padded rows cost index score time.
    recipe_args=(--cudagraph-capture-sizes
        "[$AGENTIC_CAPTURE_SIZES]"
        --text-only --timeout-keep-alive 600)
    # The 1 KB stub some snapshots ship renders only the user messages.
    if [ -n "${CHAT_TEMPLATE:-}" ]; then
        recipe_args+=(--chat-template "$CHAT_TEMPLATE")
    elif [ -f "$MODEL/chat_template.jinja" ] &&
        ! grep -q thinking_mode "$MODEL/chat_template.jinja"; then
        echo "proxy_bench: $MODEL/chat_template.jinja is not the full M3.1-Pro"             "template; set CHAT_TEMPLATE to it" >&2
        exit 1
    fi
    ;;
*)
    echo "proxy_bench: RECIPE must be pro or agentic, not $RECIPE" >&2
    exit 1
    ;;
esac
export ATOM_M3PRO_VARIANTS="${ATOM_M3PRO_VARIANTS:-$default_variants}"
export MAX_NUM_SEQS
server_args=(--trust-remote-code
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens 16384
    --block-size 16
    --index-cache-dtype fp8
    --level 0
    --online_quant_config '{"layer_quant_config": {"*.experts": "mxfp4"}}'
    "${recipe_args[@]}" "${spec_args[@]}" "${server_extra[@]}")

if [ "${PROXY_DRY_RUN:-0}" = 1 ]; then
    echo "acceptance_length=${ACCEPT_LENGTH:-none}"
    echo "prewarm=$PREWARM"
    if [ "$PREWARM" = 1 ]; then
        echo "prewarm_prompt_tokens=$PREWARM_PROMPT_TOKENS"
        echo "prewarm_batches=$PREWARM_BATCHES"
    fi
    env | grep -E '^(ATOM_|AITER_|GPTOSS_|LMCACHE_|PYTHONHASHSEED=|GPU_MEM_UTIL=|KV_CACHE_DTYPE=|MAX_NUM_SEQS=)' | sort
    printf 'server_args:'
    printf ' %q' "${server_args[@]}"
    echo
    exit 0
fi

mkdir -p "$OUT_DIR" "$(dirname "$SERVER_LOG")"
echo "proxy_bench: CONC=$CONC RECIPE=$RECIPE AL=${ACCEPT_LENGTH:-none} out=$OUT_DIR model=$MODEL"
python3 -c 'import atom, sys; print("atom from", atom.__file__)'

# --- 1. session subset ---
subset_args=(--out "$OUT_DIR/traces" --seed "$SEED" --max-context "$MAX_CONTEXT"
    --num-sessions "$NUM_SESSIONS")
[ -z "${TRACES:-}" ] || subset_args+=(--traces "$TRACES")
python3 "$HERE/trace_subset.py" "${subset_args[@]}"
NUM_TRACES=$(find "$OUT_DIR/traces" -maxdepth 1 -name '*.json' | wc -l)

# --- 2. server: RECIPE's launch, DSpark (SPEC_TOKENS), forced AL ---
# A compile cache left by other code fails silently after a code change.
rm -rf "$HOME"/.cache/atom/*

stop_server() {
    bash "$REPO/scripts/stop_atom_server.sh" || true
    cp -f "$SERVER_LOG" "$OUT_DIR/server.log" 2>/dev/null || true
}
trap stop_server EXIT

bash "$REPO/scripts/start_atom_server.sh" "$MODEL" "$TP" "$PORT" "${server_args[@]}"
# start_atom_server.sh stops waiting after two minutes; loading Pro takes longer.
bash "$REPO/scripts/wait_server_ready.sh" "$PORT" 60 30 "$SERVER_LOG"

# --- 3. prewarm ---
if [ "$PREWARM" = 1 ]; then
    echo "proxy_bench: prewarm: prompts $PREWARM_PROMPT_TOKENS, batches $PREWARM_BATCHES"
    python3 - "http://localhost:${PORT}/v1/chat/completions" "$MODEL"         "$PREWARM_PROMPT_TOKENS" "$PREWARM_BATCHES" <<'PY'
import json
import random
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

url, model, prompt_tokens, batches = sys.argv[1:]
rng = random.Random(0)
words = [f"w{i}" for i in range(5000)]


def send(words_in_prompt, max_tokens):
    # Fresh words per prompt, so no prefix-cache hit shortens the prefill.
    text = " ".join(rng.choices(words, k=words_in_prompt))
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "ignore_eos": True,
    }).encode()
    request = urllib.request.Request(url, body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as reply:
        return json.load(reply)["usage"]["completion_tokens"]


for n in map(int, prompt_tokens.split(",")):
    send(n, 1)
for n in map(int, batches.split(",")):
    with ThreadPoolExecutor(n) as pool:
        done = list(pool.map(lambda _: send(64, 64), range(n)))
    assert all(done), f"batch {n}: empty reply"
print("proxy_bench: prewarm done")
PY
fi

# --- 4. replay ---
export AIPERF_HTTP_TCP_USER_TIMEOUT=900000
export AIPERF_DATASET_CONFIGURATION_TIMEOUT=1800
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=1800
export AIPERF_TIMING_CANCEL_DRAIN_TIMEOUT=300
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES=0
export AIPERF_UI_REALTIME_METRICS_ENABLED=true
export AIPERF_FAILED_REQUEST_THRESHOLD=0.10
export AIPERF_LIVE_FAILED_REQUEST_THRESHOLD=0.10
export AIPERF_WARMUP_REQUESTS_PER_LANE=10
export AIPERF_BENCHMARK_GRACE_PERIOD=30
export AIPERF_SERVER_METRICS_URLS="http://localhost:${PORT}/metrics"
export AIPERF_REQUIRED_SERVER_METRIC_PREFIX="atom:"

# The scenario cannot verify a local subset's corpus identity and refuses it
# without --unsafe-override, which stamps the run submission_valid=false: a
# proxy number, never a submission. aiperf's own --max-context-length filter is
# the same as trace_subset.py's, so it keeps every session; passing it makes
# aiperf log that it did.
aiperf profile --scenario inferencex-agentx-mvp --unsafe-override \
    --url "http://localhost:${PORT}" --endpoint /v1/chat/completions \
    --endpoint-type chat --streaming \
    --model "$MODEL" --tokenizer "$MODEL" \
    --tokenizer-trust-remote-code \
    --apply-chat-template \
    --concurrency "$CONC" --benchmark-duration "$DURATION" --stats-interval 30 \
    --random-seed "$SEED" --failed-request-threshold 0.10 \
    --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
    --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds 300 \
    --agentic-warmup-grace-period 1800 \
    --use-server-token-count --no-gpu-telemetry \
    --num-dataset-entries "$NUM_TRACES" --slice-duration 1.0 \
    --server-metrics "http://localhost:${PORT}/metrics" \
    --custom-dataset-type weka_trace --input-file "$OUT_DIR/traces" \
    --max-context-length "$MAX_CONTEXT" \
    --output-artifact-dir "$OUT_DIR/aiperf" 2>&1 | tee "$OUT_DIR/aiperf.log"

stop_server
trap - EXIT

# --- 5. summary ---
python3 - "$REPO" "$OUT_DIR" "$CONC" "$TP" <<'PY'
import json
import sys
from pathlib import Path

repo, out_dir, conc, gpus = Path(sys.argv[1]), Path(sys.argv[2]), *map(int, sys.argv[3:])
sys.path.insert(0, str(repo / ".github/scripts/atomesh"))
from interactivity import (  # noqa: E402
    PROFILING_PHASE,
    agentic_interactivity,
    agentic_throughput,
    iter_records,
)

records = out_dir / "aiperf/profile_export.jsonl"
if not records.is_file():
    sys.exit(f"proxy_bench: {records} was not produced")

total = failed = off_trace = 0
for record, _raw in iter_records(records):
    if record is None:
        continue
    metadata = record.get("metadata") or {}
    if metadata.get("benchmark_phase") not in (None, PROFILING_PHASE):
        continue
    total += 1
    failed += bool(record.get("error"))
    # aiperf's gap between the requested max_tokens and the tokens returned.
    mismatch = (record.get("metrics") or {}).get("osl_mismatch_diff_pct") or {}
    off_trace += bool(mismatch.get("value"))

interactivity = agentic_interactivity(records)
throughput = agentic_throughput(records)
summary = {
    "concurrency": conc,
    "p90_e2e_normalized_interactivity": interactivity["value"],
    "p90_interactivity": interactivity["itl_value"],
    "tok_s_per_gpu": throughput["total_tput_tps"] / gpus,
    "output_tok_s_per_gpu": throughput["output_tput_tps"] / gpus,
    "failed_request_rate": failed / total if total else None,
    "requests": total,
    "failed_requests": failed,
    "osl_mismatch_requests": off_trace,
    "throughput_window_s": throughput["duration_s"],
}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(f"c{conc}: p90 E2E-normalized interactivity "
      f"{summary['p90_e2e_normalized_interactivity']:.2f} tok/s/user, "
      f"{summary['tok_s_per_gpu']:.1f} tok/s/GPU, "
      f"failed {failed}/{total} ({100 * failed / max(total, 1):.2f}%), "
      f"output length off the trace in {off_trace}")
PY
