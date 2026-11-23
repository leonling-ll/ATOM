#!/bin/bash
# MiniMax-M3.1-Pro nightly checks, on two servers launched as accuracy_gate.sh launches its
# own (the GSM8K section of recipes/MiniMax-M3-Pro.md):
#
#   DSpark k=7 server:  GSM8K 64-shot, score + stderr >= M31PRO_THRESHOLD; 14k-token key-value retrieval, 12/12;
#                       the InferenceX acceptance-length (AL) gate replay, report only.
#   no-DSpark server:   GSM8K 64-shot, score + stderr >= M31PRO_THRESHOLD; a greedy completion smoke.
#
# Exits 1 when any gated check fails; the AL replay only prints warnings.
# Run it like accuracy_gate.sh (tree root, ATOM container, whole node, node lock); it
# takes the same M31PRO_* variables, plus, for the AL replay:
#   INFERENCEX_DIR       an InferenceX checkout (the replay is skipped without it)
#   AL_GATE_CURVE        golden curve name in infx/golden_al_distribution (skipped without it)
#   AL_GATE_DATASET      SPEED-Bench prompts, relative to InferenceX's inferencex-e2e/
#   AL_GATE_THINKING     the curve's collection mode (default thinking_on)
#   M31PRO_AL_BASELINE   this deployment's own earlier AL; warns 5% under it
# The replay also warns 10% under the curve's golden AL.

source "$(dirname "${BASH_SOURCE[0]}")/accuracy_gate.sh"
set -uo pipefail
FAILED=()

# 2000 lines "Item i is colored c." (about 14k tokens), 12 lookups spread over the list,
# 4 at a time: the answer sits in one 16-token block the indexer has to select.
retrieval() {
    python - "$M31PRO_PORT" "$M31PRO_MODEL" <<'PY'
import concurrent.futures as cf, json, random, sys, urllib.request

port, model, lines, n = sys.argv[1], sys.argv[2], 2000, 12
names = "red blue green yellow purple orange black white pink brown gray cyan magenta violet indigo teal"
rng = random.Random(0)
colors = [rng.choice(names.split()) for _ in range(lines)]
body = "".join(f"Item {i} is colored {c}.\n" for i, c in enumerate(colors))

def ask(item):
    prompt = (f"]~!b[Here is a list of facts.\n{body}\n"
              f"Question: What color is item {item}?\nAnswer: Item {item} is colored")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions", headers={"Content-Type": "application/json"},
        data=json.dumps({"model": model, "prompt": prompt, "max_tokens": 3, "temperature": 0}).encode())
    with urllib.request.urlopen(req, timeout=3600) as r:
        words = json.loads(r.read())["choices"][0]["text"].split()
    return item, words[0].strip(".").lower() if words else ""

items = [min(int(lines * (0.03 + 0.9 * k / (n - 1))), lines - 25) for k in range(n)]
with cf.ThreadPoolExecutor(4) as ex:
    right = sum(got == colors[item] for item, got in ex.map(ask, items))
print(f"retrieval 14k: {right}/{n}")
sys.exit(right != n)
PY
}

smoke() {
    local text
    text=$(curl -sf "http://127.0.0.1:$M31PRO_PORT/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\": \"$M31PRO_MODEL\", \"prompt\": \"]~!b[The capital of France is\", \"max_tokens\": 24, \"temperature\": 0}" |
        python -c 'import json, sys; print(json.load(sys.stdin)["choices"][0]["text"])')
    echo "smoke: $text"
    [[ "$text" == *Paris* ]]
}

# InferenceX's own al-gate command against this server, then this deployment's warnings.
al_gate() {
    if [ -z "${INFERENCEX_DIR:-}" ] || [ -z "${AL_GATE_CURVE:-}" ] || [ -z "${AL_GATE_DATASET:-}" ]; then
        echo "AL gate: skipped (set INFERENCEX_DIR, AL_GATE_CURVE and AL_GATE_DATASET)"
        return 0
    fi
    local out="$M31PRO_OUT/dspark/al_gate"
    (cd "$INFERENCEX_DIR/inferencex-e2e" && AL_GATE=true AL_GATE_DRAFT_TOKENS=7 \
        AL_GATE_OUTPUT_DIR="$out" THINKING_MODE="${AL_GATE_THINKING:-thinking_on}" \
        RESULT_FILENAME=m31pro MODEL="$M31PRO_MODEL" FRAMEWORK=atom PORT="$M31PRO_PORT" \
        AIPERF_SERVER_METRICS_URLS="http://127.0.0.1:$M31PRO_PORT/metrics" \
        timeout "${AL_GATE_TIMEOUT_S:-1800}" python3 -m infx.bench al-gate) >"$out.log" 2>&1
    python - "$out/al_gate_m31pro.json" "${M31PRO_AL_BASELINE:-}" <<'PY'
import json, sys

try:
    doc = json.load(open(sys.argv[1]))
except OSError as e:
    print(f"AL gate: no result ({e}); see the log beside it")
    sys.exit(0)
al, golden = doc["acceptance_length"], doc["golden_acceptance_length"]
print(f"AL gate: status {doc['status']}, AL {al}, golden {golden}, error {doc['error']}")
if al is not None and golden and al < 0.90 * golden:
    print(f"WARNING: AL {al} is more than 10% under golden {golden}")
if al is not None and sys.argv[2] and al < 0.95 * float(sys.argv[2]):
    print(f"WARNING: AL {al} is more than 5% under this deployment's baseline {sys.argv[2]}")
PY
}

check() {
    local name=$1
    shift
    "$@" || FAILED+=("$name")
}

trap stop_server EXIT
if start_server dspark --method dspark --num-speculative-tokens 7; then
    check "gsm8k dspark" run_gsm8k dspark
    check "retrieval" retrieval
    al_gate
    stop_server
    check "gsm8k dspark score" check_gsm8k "$M31PRO_OUT/dspark/gsm8k"
else
    FAILED+=("dspark server")
fi
if start_server plain; then
    check "gsm8k plain" run_gsm8k plain
    check "smoke" smoke
    stop_server
    check "gsm8k plain score" check_gsm8k "$M31PRO_OUT/plain/gsm8k"
else
    FAILED+=("plain server")
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
    printf 'NIGHTLY FAIL: %s\n' "${FAILED[*]}"
    exit 1
fi
echo "NIGHTLY PASS"
