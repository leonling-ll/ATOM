#!/bin/bash
# MiniMax-M3.1-Pro accuracy gate: GSM8K 64-shot, all 1319 questions; passes when the
# flexible-extract score plus its standard error reaches M31PRO_THRESHOLD.
#
# Launches the server exactly as the GSM8K section of recipes/MiniMax-M3-Pro.md does
# (TP8 kernel path, fp8 KV cache, bf16 index cache, prefix caching off, no DSpark), runs
# that section's lm-eval command, stops the server and judges the flexible-extract score,
# the column the recipe's baseline is quoted in.
#
# Run it from the root of the tree under test, inside the ATOM container, on a whole
# 8xMI355X node, under the node lock:
#   flock /tmp/m31pro-gpu.lock docker exec <container> bash scripts/m31pro/accuracy_gate.sh
#
# Environment:
#   M31PRO_MODEL      checkpoint (default /data/models/yuhua/m3-pro, the recipe's path)
#   M31PRO_PORT       server port (default 8300)
#   M31PRO_OUT        output directory (default $HOME/m31pro-logs/<UTC stamp>)
#   M31PRO_THRESHOLD  bar for flexible-extract plus its stderr (default 0.94)
#   M31PRO_LIMIT      run only the first N questions; the verdict then gates nothing
#   M31PRO_READY_S    seconds to wait for the server (default 3600)
#   M31PRO_GPU_WAIT_S seconds to wait for memory a previous run is freeing (default 300)
#
# `accuracy_gate.sh --check DIR` only judges the lm-eval results already under DIR.
# nightly.sh sources this file for its server and GSM8K helpers.

M31PRO_MODEL="${M31PRO_MODEL:-/data/models/yuhua/m3-pro}"
M31PRO_PORT="${M31PRO_PORT:-8300}"
M31PRO_OUT="${M31PRO_OUT:-$HOME/m31pro-logs/$(date -u +%Y%m%d-%H%M%S)}"
M31PRO_THRESHOLD="${M31PRO_THRESHOLD:-0.94}"
M31PRO_LIMIT="${M31PRO_LIMIT:-}"
M31PRO_READY_S="${M31PRO_READY_S:-3600}"
SERVER_PID=""

# Refuse to start on a node whose GPUs hold memory: that is someone else's run. A run
# that just released the node lock can still be freeing its memory, so wait for that
# first, up to M31PRO_GPU_WAIT_S seconds (default 300).
gpus_free() {
    local used waited=0
    while :; do
        used=$(rocm-smi --showmemuse 2>/dev/null | awk '/VRAM%/ && $NF + 0 > 0 {n++} END {print n + 0}')
        [ "$used" -gt 0 ] && [ "$waited" -lt "${M31PRO_GPU_WAIT_S:-300}" ] || break
        sleep 10
        waited=$((waited + 10))
    done
    if [ "$used" -gt 0 ]; then
        echo "ERROR: $used GPU(s) already hold memory; not starting." >&2
        rocm-smi --showpidgpus 2>&1 | grep "is using" >&2 || true
        return 1
    fi
}

# start_server TAG [EXTRA_ARGS...]: the recipe's GSM8K launch, log in $M31PRO_OUT/TAG.
start_server() {
    local tag=$1
    shift
    gpus_free || return 1
    mkdir -p "$M31PRO_OUT/$tag"
    # A compile cache left by other code fails silently after a code change.
    rm -rf "$HOME"/.cache/atom/*
    echo "Starting server ($tag), log $M31PRO_OUT/$tag/server.log"
    # Quick all-reduce stays off whatever the caller's environment says.
    AITER_LOG_LEVEL="${AITER_LOG_LEVEL:-WARNING}" \
        AITER_QUICK_REDUCE_QUANTIZATION=NONE \
        AITER_CUSTOM_AR_DISABLE_LAMPORT=1 \
        ATOM_M3PRO_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "fused"}' \
        setsid python -m atom.entrypoints.openai_server \
        --model "$M31PRO_MODEL" \
        --trust-remote-code \
        --tensor-parallel-size 8 \
        --server-port "$M31PRO_PORT" \
        --max-model-len 16384 \
        --max-num-seqs 16 \
        --max-num-batched-tokens 16384 \
        --gpu-memory-utilization 0.85 \
        --block-size 16 \
        --kv_cache_dtype fp8 \
        --index-cache-dtype bf16 \
        --level 0 \
        --no-enable_prefix_caching \
        --online_quant_config '{"layer_quant_config": {"*.experts": "mxfp4"}}' \
        "$@" >"$M31PRO_OUT/$tag/server.log" 2>&1 &
    SERVER_PID=$!
    local waited=0
    until curl -sf "http://127.0.0.1:$M31PRO_PORT/v1/models" >/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server exited during startup; see $M31PRO_OUT/$tag/server.log" >&2
            SERVER_PID=""
            return 1
        fi
        if [ "$waited" -ge "$M31PRO_READY_S" ]; then
            echo "ERROR: server not ready after ${M31PRO_READY_S}s" >&2
            return 1
        fi
        sleep 10
        waited=$((waited + 10))
    done
    echo "Server ready after ${waited}s"
}

# Stop the server's whole process group (setsid made the server its leader).
stop_server() {
    [ -n "$SERVER_PID" ] || return 0
    kill -TERM -- "-$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 60); do
        kill -0 "$SERVER_PID" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null
    SERVER_PID=""
}

# run_gsm8k TAG: the recipe's lm-eval command; results land in $M31PRO_OUT/TAG/gsm8k.
run_gsm8k() {
    local tag=$1
    lm_eval --model local-completions \
        --model_args "model=$M31PRO_MODEL,tokenizer=$M31PRO_MODEL,base_url=http://127.0.0.1:$M31PRO_PORT/v1/completions,num_concurrent=16,max_retries=3,tokenized_requests=False,trust_remote_code=True" \
        --tasks gsm8k --num_fewshot 64 \
        ${M31PRO_LIMIT:+--limit "$M31PRO_LIMIT"} \
        --output_path "$M31PRO_OUT/$tag/gsm8k" 2>&1 | tee "$M31PRO_OUT/$tag/lm_eval.log"
    return "${PIPESTATUS[0]}"
}

# check_gsm8k DIR: judge the newest lm-eval results file under DIR against the bar.
check_gsm8k() {
    python - "$1" "$M31PRO_THRESHOLD" <<'PY'
import json, pathlib, sys

files = sorted(pathlib.Path(sys.argv[1]).rglob("results_*.json"), key=lambda p: p.stat().st_mtime)
if not files:
    sys.exit(f"ERROR: no lm-eval results under {sys.argv[1]}")
doc = json.loads(files[-1].read_text())
r = doc["results"]["gsm8k"]
n = doc.get("n-samples", {}).get("gsm8k", {}).get("effective", "?")
flex, strict = r["exact_match,flexible-extract"], r["exact_match,strict-match"]
print(f"GSM8K 64-shot, {n} questions: flexible-extract {flex:.4f} +- "
      f"{r['exact_match_stderr,flexible-extract']:.4f}, strict-match {strict:.4f} +- "
      f"{r['exact_match_stderr,strict-match']:.4f}")
# A run passes when the bar is within one standard error of its score.
stderr, bar = r["exact_match_stderr,flexible-extract"], float(sys.argv[2])
verdict = "PASS" if flex + stderr >= bar else "FAIL"
print(f"{verdict}: flexible-extract {flex:.4f} + stderr {stderr:.4f} = {flex + stderr:.4f} vs bar {bar}")
sys.exit(verdict == "FAIL")
PY
}

main() {
    if [ "${1:-}" = "--check" ]; then
        check_gsm8k "$2"
        return
    fi
    trap stop_server EXIT
    start_server gate || return 1
    run_gsm8k gate || {
        echo "ERROR: lm_eval failed" >&2
        return 1
    }
    stop_server
    [ -z "$M31PRO_LIMIT" ] || echo "NOTE: M31PRO_LIMIT=$M31PRO_LIMIT; this run gates nothing."
    check_gsm8k "$M31PRO_OUT/gate/gsm8k"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    set -uo pipefail
    main "$@"
fi
