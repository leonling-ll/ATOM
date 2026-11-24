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
#   2. starts the server with the Pro recipe's fast configuration plus DSpark
#      at 7 draft tokens and a forced acceptance length,
#   3. replays the subset at concurrency CONC for DURATION seconds,
#   4. stops the server and prints, from aiperf's per-request records, the
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
#   ACCEPT_LENGTH    forced DSpark acceptance length [4.0]
#   MAX_NUM_SEQS     [16 below CONC 8, else 2 x CONC]
#   MAX_MODEL_LEN    [MAX_CONTEXT + 1024: room for the chat template]
#   PORT             server port [8300]
#   GPU_MEM_UTIL     --gpu-memory-utilization [0.94]
#   SERVER_EXTRA     more server flags, split on spaces, e.g. --load_dummy=xavier []
#   ATOM_CUSTOM_AR_POOL_BYTES  custom all-reduce input pool [134217728]
#   ATOM_M3PRO_VARIANTS  Pro attention variants
#                    [{"kernels": true, "attn": "aiter", "topk": "fused"}]
#   The server inherits the rest of the environment, so CPU offload's LMCACHE_*
#   settings go there and its --kv-transfer-config in SERVER_EXTRA.

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
ACCEPT_LENGTH="${ACCEPT_LENGTH:-4.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_CONTEXT + 1024))}"
PORT="${PORT:-8300}"
TP=8
if [ -z "${MAX_NUM_SEQS:-}" ]; then
    MAX_NUM_SEQS=$((2 * CONC))
    [ "$MAX_NUM_SEQS" -ge 16 ] || MAX_NUM_SEQS=16
fi
read -r -a server_extra <<<"${SERVER_EXTRA:-}"
SERVER_LOG=/app/logs_claude/atom_server.log

mkdir -p "$OUT_DIR" "$(dirname "$SERVER_LOG")"
echo "proxy_bench: CONC=$CONC out=$OUT_DIR model=$MODEL"
python3 -c 'import atom, sys; print("atom from", atom.__file__)'

# --- 1. session subset ---
subset_args=(--out "$OUT_DIR/traces" --seed "$SEED" --max-context "$MAX_CONTEXT"
    --num-sessions "$NUM_SESSIONS")
[ -z "${TRACES:-}" ] || subset_args+=(--traces "$TRACES")
python3 "$HERE/trace_subset.py" "${subset_args[@]}"
NUM_TRACES=$(find "$OUT_DIR/traces" -maxdepth 1 -name '*.json' | wc -l)

# --- 2. server: the recipe's fast configuration, DSpark k=7, forced AL ---
export AITER_LOG_LEVEL=WARNING
export AITER_CUSTOM_AR_DISABLE_LAMPORT=1
export ATOM_CUSTOM_AR_POOL_BYTES="${ATOM_CUSTOM_AR_POOL_BYTES:-134217728}"
default_variants='{"kernels": true, "attn": "aiter", "topk": "fused"}'
export ATOM_M3PRO_VARIANTS="${ATOM_M3PRO_VARIANTS:-$default_variants}"
export KV_CACHE_DTYPE=fp8
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.94}"
export MAX_NUM_SEQS

stop_server() {
    bash "$REPO/scripts/stop_atom_server.sh" || true
    cp -f "$SERVER_LOG" "$OUT_DIR/server.log" 2>/dev/null || true
}
trap stop_server EXIT

bash "$REPO/scripts/start_atom_server.sh" "$MODEL" "$TP" "$PORT" \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 16384 \
    --block-size 16 \
    --index-cache-dtype fp8 \
    --level 0 \
    --online_quant_config '{"layer_quant_config": {"*.experts": "mxfp4"}}' \
    --method dspark \
    --num-speculative-tokens 7 \
    --spec-decode-acceptance-length "$ACCEPT_LENGTH"     "${server_extra[@]}"
# start_atom_server.sh stops waiting after two minutes; loading Pro takes longer.
bash "$REPO/scripts/wait_server_ready.sh" "$PORT" 60 30 "$SERVER_LOG"

# --- 3. replay ---
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

# --- 4. summary ---
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
