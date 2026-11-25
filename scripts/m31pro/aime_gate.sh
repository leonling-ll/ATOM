#!/bin/bash
# MiniMax-M3.1-Pro AIME gate: AIME 2025 (30 problems), 0-shot with thinking, through the
# chat endpoint; passes when avg@k plus its standard error reaches AIME_THRESHOLD.
#
# Launches accuracy_gate.sh's server (so GATE_VARIANTS, GATE_DISABLE_LAMPORT and the
# M31PRO_* variables apply) with the AIME settings of recipes/MiniMax-M3-Pro.md: 131k
# context, 32 sequences, and MiniMax-M3's chat template with thinking_mode enabled, which
# that recipe serves Pro with for chat. aime_client.py then sends the recipe's AIME prompt
# with its sampling (temperature 1.0, top-p 0.95, up to 98,304 new tokens) and scores the
# last \boxed{} of each reply's content.
#
# Pass bar (provisional): 0.90. There is no AIME 2025 score for Pro yet; the closest
# reference is MiniMax-M3.1-Flash on ATOM, AIME 2025 mean@16 0.906 with this protocol
# (recipes/MiniMax-M3.md, Accuracy Test). Pro scores above Flash on AIME 2026 (0.956 against
# 0.927), so Pro under 0.90 on AIME 2025 is a regression. Replace the bar with Pro's own
# score once a full run has measured it.
#
# Run it like accuracy_gate.sh: tree root, ATOM container, whole node, node lock:
#   flock /tmp/m31pro-gpu.lock docker exec <container> bash scripts/m31pro/aime_gate.sh
#
# Environment, besides accuracy_gate.sh's:
#   AIME_CHAT_TEMPLATE  chat template file (default MiniMax-M3-MXFP4's on /shared_nfs)
#   AIME_DATASET        HF dataset with a test split, or a .jsonl of problem/answer/id
#                       rows (default math-ai/aime25)
#   AIME_SAMPLES        samples a problem (default 8)
#   AIME_CONCURRENCY    requests in flight (default 32, the server's --max-num-seqs)
#   AIME_MAX_TOKENS     new tokens a reply (default 98304)
#   AIME_LIMIT          only the first N problems; the verdict then gates nothing
#   AIME_THRESHOLD      bar for avg@k plus its stderr (default 0.90)
#
# Arguments other than --check are added to the server's launch line.
# `aime_gate.sh --check DIR` only judges the run already under DIR.

source "$(dirname "${BASH_SOURCE[0]}")/accuracy_gate.sh"
AIME_CLIENT="$(dirname "${BASH_SOURCE[0]}")/aime_client.py"
AIME_CHAT_TEMPLATE="${AIME_CHAT_TEMPLATE:-/shared_nfs/huggingface_models/amd/MiniMax-M3-MXFP4/chat_template.jinja}"
AIME_THRESHOLD="${AIME_THRESHOLD:-0.90}"
AIME_CONCURRENCY="${AIME_CONCURRENCY:-32}"

# run_aime TAG: the client against the running server; results in $M31PRO_OUT/TAG/aime.
run_aime() {
    python "$AIME_CLIENT" run "$M31PRO_OUT/$1/aime" --port "$M31PRO_PORT" --model "$M31PRO_MODEL" \
        --dataset "${AIME_DATASET:-math-ai/aime25}" --samples "${AIME_SAMPLES:-8}" \
        --concurrency "$AIME_CONCURRENCY" --max-tokens "${AIME_MAX_TOKENS:-98304}" \
        ${AIME_LIMIT:+--limit "$AIME_LIMIT"} 2>&1 | tee "$M31PRO_OUT/$1/aime_client.log"
    return "${PIPESTATUS[0]}"
}

check_aime() {
    python "$AIME_CLIENT" check "$1" --threshold "$AIME_THRESHOLD"
}

aime_main() {
    if [ "${1:-}" = "--check" ]; then
        check_aime "$2"
        return
    fi
    trap stop_server EXIT
    start_server aime --max-model-len 131072 --max-num-seqs "$AIME_CONCURRENCY" \
        --chat-template "$AIME_CHAT_TEMPLATE" \
        --default-chat-template-kwargs '{"thinking_mode": "enabled"}' "$@" || return 1
    run_aime aime || echo "ERROR: some AIME requests failed" >&2
    stop_server
    [ -z "${AIME_LIMIT:-}" ] || echo "NOTE: AIME_LIMIT=$AIME_LIMIT; this run gates nothing."
    check_aime "$M31PRO_OUT/aime/aime"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    set -uo pipefail
    aime_main "$@"
fi
