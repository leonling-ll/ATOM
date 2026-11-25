#!/bin/bash
# MiniMax-M3.1-Pro at TP8 + DCP8 with its DSpark draft (k=5) and the LMCache CPU tier.
# Run it like accuracy_gate.sh (tree root, ATOM container, whole 8xMI355X node, node lock):
#   flock /tmp/m31pro-gpu.lock docker exec <container> bash scripts/m31pro/dcp8_check.sh STEP
#
# STEP:
#   serve    the CONC 64 launch (recipes/Agentic-MiniMax-M3.1-Pro.md with a 1M context, 128
#            sequences, DSpark k=5 at the forced AL 2.89) at DCP8, left running for the
#            600 s CONC 64 client; stop it with kill -TERM -- -<pid it prints>
#   smoke    the same launch at real acceptance: the startup log's DCP, KV pool and offload
#            lines, then greedy key-value retrievals at about 8k, 32k and 96k tokens
#   restore  a 24k-token retrieval twice, the second pass on a fresh server (lmcache_mp: the
#            MP server keeps the tier) or with prefix caching off (lmcache): pass 2 loads its
#            prefix from the CPU tier (atom:lmcache_loaded_tokens) and answers the same text
#   ab       smoke's retrievals one at a time under DCP_VARIANTS, then AB_VARIANTS (default:
#            DCP_VARIANTS with dcp_topk_merge off): the same text
#   gsm8k    accuracy_gate.sh's GSM8K 64-shot with DSpark k=5 at real acceptance and these
#            variants, at DCP8 and then at DCP1
#
# Environment, besides accuracy_gate.sh's M31PRO_*:
#   DCP           --decode-context-parallel-size (default 8)
#   OFFLOAD       lmcache_mp (default; the LMCache MP server must already run, as for the
#                 m1_c64_mp2240_106 baseline, with that run's ATOM_KV_OFFLOAD_EXTRA_CONFIG,
#                 lmcache.mp.tp_rank_collapse unset or false: true is refused under DCP),
#                 lmcache (in-process, LMCACHE_MAX_LOCAL_CPU_SIZE GiB a rank, default 280)
#                 or none
#   DCP_VARIANTS  ATOM_M3PRO_VARIANTS (default C64_VARIANTS below: the CONC 64 recipe's, plus
#                 score_tq_causal, index_day0 (no effect under DCP) and dcp_topk_merge)
#   AB_VARIANTS   ab's second arm's ATOM_M3PRO_VARIANTS
#   MAX_MODEL_LEN --max-model-len (default 1048576)
#   DRY_RUN=1     print the server's environment and command line instead of starting it

source "$(dirname "${BASH_SOURCE[0]}")/accuracy_gate.sh"

DCP="${DCP:-8}"
OFFLOAD="${OFFLOAD:-lmcache_mp}"
C64_VARIANTS='{"kernels": true, "attn": "aiter", "topk": "split", "prep": "aiter", "index_split": true, "score_tq": 8, "ar_lamport_rows": 16, "k_only": true, "k_plain_fp8": true, "prefill_attn": "day0", "index_split_decode": 64, "moe_route_glue": true, "index_compress": true, "score_tq_causal": true, "index_day0": true, "dcp_topk_merge": true}'
DCP_VARIANTS="${DCP_VARIANTS:-$C64_VARIANTS}"
CAPTURE_SIZES="[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,18,20,22,24,26,28,30,32,48,64,128,256,512]"

wait_ready() {
    local waited=0
    until curl -sf "http://127.0.0.1:$M31PRO_PORT/v1/models" >/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server exited during startup; see $M31PRO_OUT/$1/server.log" >&2
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

# start_dcp TAG [EXTRA_ARGS...]: the CONC 64 launch at --decode-context-parallel-size $DCP
# with $OFFLOAD and $DCP_VARIANTS; log in $M31PRO_OUT/TAG/server.log.
start_dcp() {
    local tag=$1
    shift
    local env=(
        AITER_LOG_LEVEL=WARNING
        ATOM_M3PRO_VARIANTS="$DCP_VARIANTS"
        ATOM_WARMUP_TOKENS=auto
        ATOM_ONLINE_QUANT_STREAMING=1
        ATOM_CACHE_AWARE_ADMISSION=1
        AITER_REUSE_IDENTICAL_COMM_GROUPS=1
        ATOM_CUSTOM_AR_POOL_BYTES=134217728
        GPTOSS_SWIGLU_MXFP4_BF16_BOUND=1040
        ATOM_TOKEN_GRAPH_SIZES=256,512,1024,2048
        PYTHONHASHSEED=0
    )
    case "$OFFLOAD" in
    lmcache_mp) env+=(ATOM_KV_OFFLOAD=lmcache_mp LMCACHE_CHUNK_SIZE=256) ;;
    lmcache)
        env+=(ATOM_KV_OFFLOAD=lmcache LMCACHE_LOCAL_CPU=True
            LMCACHE_MAX_LOCAL_CPU_SIZE="${LMCACHE_MAX_LOCAL_CPU_SIZE:-280}"
            LMCACHE_CHUNK_SIZE=256)
        ;;
    none) ;;
    *)
        echo "ERROR: OFFLOAD must be lmcache_mp, lmcache or none, got '$OFFLOAD'" >&2
        return 1
        ;;
    esac
    local cmd=(
        python -m atom.entrypoints.openai_server
        --model "$M31PRO_MODEL"
        --trust-remote-code
        --tensor-parallel-size 8
        --decode-context-parallel-size "$DCP"
        --server-port "$M31PRO_PORT"
        --kv_cache_dtype fp8
        --index-cache-dtype fp8
        --block-size 16
        --level 0
        --max-model-len "${MAX_MODEL_LEN:-1048576}"
        --max-num-seqs 128
        --max-num-batched-tokens 16384
        --cudagraph-capture-sizes "$CAPTURE_SIZES"
        --gpu-memory-utilization 0.95
        --online_quant_config '{"layer_quant_config": {"*.experts": "mxfp4"}}'
        --method dspark
        --num-speculative-tokens 5
        --text-only
        --timeout-keep-alive 600
        "$@"
    )
    if [ -n "${DRY_RUN:-}" ]; then
        echo "unset AITER_CUSTOM_AR_DISABLE_LAMPORT"
        printf 'export %q\n' "${env[@]}"
        printf '%q ' "${cmd[@]}"
        echo
        return 0
    fi
    gpus_free || return 1
    mkdir -p "$M31PRO_OUT/$tag"
    # A compile cache left by other code fails silently after a code change.
    rm -rf "$HOME"/.cache/atom/*
    echo "Starting server ($tag, DCP $DCP, offload $OFFLOAD), log $M31PRO_OUT/$tag/server.log"
    echo "  ATOM_M3PRO_VARIANTS=$DCP_VARIANTS"
    # ar_lamport_rows needs the Lamport all-reduce path
    env -u AITER_CUSTOM_AR_DISABLE_LAMPORT "${env[@]}" setsid "${cmd[@]}" \
        >"$M31PRO_OUT/$tag/server.log" 2>&1 &
    SERVER_PID=$!
    wait_ready "$tag"
}

# startup_report TAG: the lines a DCP8 + DSpark + offload start has to show; fails on a
# traceback or without the DCP interleave.
startup_report() {
    local log="$M31PRO_OUT/$1/server.log"
    grep -E "interleave_size|num_kvcache_blocks|M3-Pro attention|DSpark|dspark|LMCache|lmcache|capture" \
        "$log" | grep -v "^\[aiter\]" | sort | uniq -c | sort -rn | head -40
    if grep -q "Traceback" "$log"; then
        echo "ERROR: traceback in $log" >&2
        return 1
    fi
    if [ "$DCP" -gt 1 ] && ! grep -q "interleave_size -> 16" "$log"; then
        echo "ERROR: no 'interleave_size -> 16' line: DCP was not admitted" >&2
        return 1
    fi
}

# retrieval LINES LOOKUPS CONCURRENCY OUT: nightly.sh's key-value retrieval over LINES lines
# (about 7 tokens each), greedy; the answers (24 tokens each) go to OUT as JSON.
retrieval() {
    python - "$M31PRO_PORT" "$M31PRO_MODEL" "$@" <<'PY'
import concurrent.futures as cf, json, random, sys, urllib.request

port, model = sys.argv[1], sys.argv[2]
lines, n, conc, out = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
names = "red blue green yellow purple orange black white pink brown gray cyan magenta violet indigo teal"
rng = random.Random(0)
colors = [rng.choice(names.split()) for _ in range(lines)]
body = "".join(f"Item {i} is colored {c}.\n" for i, c in enumerate(colors))

def ask(item):
    prompt = (f"]~!b[Here is a list of facts.\n{body}\n"
              f"Question: What color is item {item}?\nAnswer: Item {item} is colored")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", headers={"Content-Type": "application/json"},
        data=json.dumps({"model": model, "prompt": prompt, "max_tokens": 24, "temperature": 0}).encode())
    with urllib.request.urlopen(req, timeout=3600) as r:
        doc = json.loads(r.read())
    text = doc["choices"][0]["text"]
    words = text.split()
    return item, text, (words[0].strip(".").lower() if words else ""), doc["usage"]["prompt_tokens"]

items = [min(int(lines * (0.03 + 0.9 * k / max(n - 1, 1))), lines - 25) for k in range(n)]
with cf.ThreadPoolExecutor(conc) as ex:
    rows = list(ex.map(ask, items))
right = sum(got == colors[item] for item, _, got, _ in rows)
print(f"retrieval {lines} lines ({rows[0][3]} tokens): {right}/{n}")
json.dump([[item, text] for item, text, _, _ in rows], open(out, "w"))
sys.exit(right < n - n // 4)
PY
}

# counter NAME: the sum of a Prometheus counter over its label sets.
counter() {
    curl -sf "http://127.0.0.1:$M31PRO_PORT/metrics" | python -c '
import re, sys
name = sys.argv[1]
total = 0.0
for line in sys.stdin:
    m = re.match(r"(\S+?)(\{.*\})? (\S+)$", line.strip())
    if m and m.group(1) in (name, name + "_total"):
        total += float(m.group(3))
print(int(total))' "$1"
}

LENGTHS="1150 4600 13700" # retrieval lines: about 8k, 32k and 96k tokens

step_smoke() {
    local out="$M31PRO_OUT/smoke" rc=0 lines
    start_dcp smoke || return 1
    [ -n "${DRY_RUN:-}" ] && return 0
    startup_report smoke || rc=1
    for lines in $LENGTHS; do
        retrieval "$lines" 8 4 "$out/retrieval_$lines.json" || rc=1
    done
    stop_server
    return $rc
}

# wait_saved TOKENS: until the server reports that many tokens saved (at most 120 s).
wait_saved() {
    local saved=0
    for _ in $(seq 1 24); do
        saved=$(counter atom:lmcache_saved_tokens)
        [ "${saved:-0}" -ge "$1" ] && break
        sleep 5
    done
    echo "saved ${saved:-0} tokens"
    [ "${saved:-0}" -ge "$1" ]
}

step_restore() {
    local out="$M31PRO_OUT/restore" rc=0 loaded0 loaded1 failures
    [ -n "${DRY_RUN:-}" ] || mkdir -p "$out"
    if [ "$OFFLOAD" = lmcache_mp ]; then
        # The MP server outlives this one: pass 2 runs on a fresh server, HBM empty.
        start_dcp restore_pass1 || return 1
        [ -n "${DRY_RUN:-}" ] && return 0
        startup_report restore_pass1 || rc=1
        retrieval 3400 1 1 "$out/pass1.json" || rc=1
        wait_saved 16384 || rc=1
        stop_server
        start_dcp restore_pass2 || return 1
    elif [ "$OFFLOAD" = lmcache ]; then
        # The in-process tier dies with its server: HBM prefix caching off instead.
        start_dcp restore --no-enable_prefix_caching || return 1
        [ -n "${DRY_RUN:-}" ] && return 0
        startup_report restore || rc=1
        retrieval 3400 1 1 "$out/pass1.json" || rc=1
        wait_saved 16384 || rc=1
    else
        echo "ERROR: restore needs OFFLOAD=lmcache_mp or lmcache" >&2
        return 1
    fi
    loaded0=$(counter atom:lmcache_loaded_tokens)
    retrieval 3400 1 1 "$out/pass2.json" || rc=1
    loaded1=$(counter atom:lmcache_loaded_tokens)
    failures=$(counter atom:lmcache_load_failures)
    stop_server
    local loaded=$((${loaded1:-0} - ${loaded0:-0}))
    echo "restore: pass 2 loaded $loaded tokens from the CPU tier, load failures ${failures:-?}"
    if [ "$loaded" -lt 16384 ] || [ "${failures:-1}" -ne 0 ]; then
        echo "ERROR: pass 2 did not load its prefix from the CPU tier" >&2
        rc=1
    fi
    if ! cmp -s "$out/pass1.json" "$out/pass2.json"; then
        echo "ERROR: pass 2 answered differently:" >&2
        cat "$out/pass1.json" "$out/pass2.json" >&2
        rc=1
    fi
    return $rc
}

step_ab() {
    local out="$M31PRO_OUT/ab" rc=0 lines arm
    local a="$DCP_VARIANTS"
    local b="${AB_VARIANTS:-${a/\"dcp_topk_merge\": true/\"dcp_topk_merge\": false}}"
    if [ "$a" = "$b" ]; then
        echo "ERROR: ab's arms are the same variants; set AB_VARIANTS" >&2
        return 1
    fi
    [ -n "${DRY_RUN:-}" ] || mkdir -p "$out"
    for arm in a b; do
        if [ "$arm" = a ]; then
            DCP_VARIANTS="$a"
        else
            DCP_VARIANTS="$b"
        fi
        start_dcp "ab_$arm" || return 1
        [ -n "${DRY_RUN:-}" ] && continue
        for lines in $LENGTHS; do
            retrieval "$lines" 4 1 "$out/${arm}_$lines.json" || rc=1
        done
        stop_server
    done
    [ -n "${DRY_RUN:-}" ] && return 0
    for lines in $LENGTHS; do
        if ! cmp -s "$out/a_$lines.json" "$out/b_$lines.json"; then
            echo "ERROR: the $lines-line answers differ between $a and $b" >&2
            rc=1
        fi
    done
    return $rc
}

step_gsm8k() {
    local rc=0 dcp
    GATE_VARIANTS="$DCP_VARIANTS"
    GATE_DISABLE_LAMPORT=0
    for dcp in 8 1; do
        if [ -n "${DRY_RUN:-}" ]; then
            echo "GATE_VARIANTS='$GATE_VARIANTS' GATE_DISABLE_LAMPORT=0 accuracy_gate.sh" \
                "--decode-context-parallel-size $dcp --method dspark --num-speculative-tokens 5"
            continue
        fi
        start_server "gsm8k_dcp$dcp" --decode-context-parallel-size "$dcp" \
            --method dspark --num-speculative-tokens 5 || {
            rc=1
            continue
        }
        run_gsm8k "gsm8k_dcp$dcp" || rc=1
        stop_server
        check_gsm8k "$M31PRO_OUT/gsm8k_dcp$dcp/gsm8k" || rc=1
    done
    return $rc
}

step_serve() {
    start_dcp c64 --spec-decode-acceptance-length 2.89 "$@" || return 1
    [ -n "${DRY_RUN:-}" ] && return 0
    startup_report c64 || return 1
    echo "Serving (pid $SERVER_PID); stop with: kill -TERM -- -$SERVER_PID"
    trap - EXIT
    wait "$SERVER_PID"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    set -uo pipefail
    step=${1:-}
    shift || true
    case "$step" in
    serve | smoke | restore | ab | gsm8k) ;;
    *)
        echo "usage: $0 serve|smoke|restore|ab|gsm8k" >&2
        exit 2
        ;;
    esac
    trap stop_server EXIT
    "step_$step" "$@"
    rc=$?
    [ "$rc" -eq 0 ] && echo "DCP8 CHECK $step: PASS" || echo "DCP8 CHECK $step: FAIL"
    exit "$rc"
fi
