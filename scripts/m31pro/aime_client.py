# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AIME through an OpenAI-compatible chat endpoint, for scripts/m31pro/aime_gate.sh.

  aime_client.py run DIR --port P --model M [--samples K] [--concurrency C] ...
      sends every problem K times to /v1/chat/completions (streamed), appending one
      record a reply to DIR/replies.jsonl, and writes DIR/run.json (settings, wall time);
  aime_client.py check DIR --threshold BAR
      prints avg@K, pass@K, maj@K, output tokens and wall time, and exits 1 unless
      every reply is in and avg@K plus its standard error reaches BAR.

The prompt and the scoring are those of the AIME numbers in recipes/MiniMax-M3-Pro.md:
the user turn asks for the answer in \\boxed{} and ends "Problem: ...\\nAnswer:"; the
answer is the content of the last \\boxed{} of the reply's text after the thinking,
compared with the reference as an integer. A reply that did not finish with "stop" (the
token cap, a cut stream) scores 0.
"""

import argparse
import collections
import concurrent.futures as cf
import http.client
import json
import math
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

PROMPT = (
    "Please solve the math problem and provide your final answer in the following format:\n\n"
    "1. Wrap your final answer with \\boxed{}\n"
    "2. The content inside \\boxed{} should be concise - only the number or mathematical expression, "
    "no extra text\n\n"
    "Examples of correct format:\n- \\boxed{42}\n\n"
    "Examples of incorrect format:\n"
    "- The answer is 42. (missing \\boxed{})\n"
    "- \\boxed{The answer is 42} (extra text inside)\n"
    "- 42 (missing \\boxed{})\n\n"
    "Problem: {question}\nAnswer:"
)
THINK_END = "</mm:think>"


def last_boxed(text):
    """Content of the last \\boxed{...} or \\fbox{...} (braces matched), or of the last
    "\\boxed x" up to the next '$'; None when there is none."""
    start = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if start < 0:
        return None
    rest = text[start:].removeprefix("\\boxed").removeprefix("\\fbox")
    if rest.startswith(" "):
        return rest[1:].split("$")[0]
    depth = 0
    for i, c in enumerate(rest):
        depth += {"{": 1, "}": -1}.get(c, 0)
        if depth == 0:
            return rest[1:i] if i else None
    return None


def extract_answer(text):
    """The last boxed answer; else the text between the first and last '$'; else all.
    Stripped of spaces, '$' and a trailing '.'; an integer in canonical form, since
    AIME answers are integers (070 and 70 are one answer)."""
    answer = last_boxed(text)
    if answer is None:
        dollars = [i for i, c in enumerate(text) if c == "$"]
        answer = text[dollars[0] + 1 : dollars[-1]] if len(dollars) > 1 else text
    answer = answer.strip().strip("$").strip().rstrip(".").strip()
    return str(int(answer)) if re.fullmatch(r"[+-]?\d+", answer) else answer


def load_problems(dataset, limit):
    if dataset.endswith(".jsonl"):
        text = Path(dataset).read_text(encoding="utf-8")
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        from datasets import load_dataset

        rows = list(load_dataset(dataset, split="test"))
    rows = [
        {"id": str(r.get("id", i)), "problem": r["problem"], "answer": str(r["answer"])}
        for i, r in enumerate(rows)
    ]
    return rows[:limit] if limit else rows


def ask(a, row, sample):
    body = {
        "model": a.model,
        "messages": [
            {"role": "user", "content": PROMPT.replace("{question}", row["problem"])}
        ],
        "max_tokens": a.max_tokens,
        "temperature": a.temperature,
        "top_p": a.top_p,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{a.port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0, content, reasoning, finish, usage = time.time(), [], [], None, None
    with urllib.request.urlopen(req, timeout=a.timeout_s) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                finish = choice.get("finish_reason") or finish
    text, thought = "".join(content), "".join(reasoning)
    # Only a finished reply scores, and only its text after the thinking: with a
    # reasoning split that is the content; without one, what follows </mm:think>
    # (a reply that never closes its thinking has no answer).
    if THINK_END in text:
        scored = text.rsplit(THINK_END, 1)[1]
    else:
        scored = text if thought else ""
    pred = extract_answer(scored) if finish == "stop" and scored else ""
    return {
        "id": row["id"],
        "sample": sample,
        "target": extract_answer(row["answer"]),
        "pred": pred,
        "match": int(bool(pred) and pred == extract_answer(row["answer"])),
        "finish": finish,
        "thinking_closed": bool(thought) or THINK_END in text,
        "completion_tokens": usage["completion_tokens"] if usage else None,
        "secs": round(time.time() - t0, 1),
        "content": text,
    }


def run(a):
    out = Path(a.dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_problems(a.dataset, a.limit)
    jobs = [(row, k) for k in range(a.samples) for row in rows]
    t0, failed = time.time(), 0
    with (
        open(out / "replies.jsonl", "w") as f,
        cf.ThreadPoolExecutor(a.concurrency) as ex,
    ):
        futures = [ex.submit(ask, a, row, k) for row, k in jobs]
        for done, fut in enumerate(cf.as_completed(futures), 1):
            try:
                f.write(json.dumps(fut.result()) + "\n")
                f.flush()
            except (
                OSError,
                ValueError,
                KeyError,
                http.client.HTTPException,
            ) as e:  # check counts the lost reply
                failed += 1
                print(f"request failed: {e!r}", file=sys.stderr, flush=True)
            if done % max(1, len(jobs) // 20) == 0:
                print(
                    f"{done}/{len(jobs)} replies, {time.time() - t0:.0f} s", flush=True
                )
    settings = {k: v for k, v in vars(a).items() if k not in ("func", "dir")}
    doc = {
        **settings,
        "problems": len(rows),
        "expected": len(jobs),
        "failed": failed,
        "wall_s": round(time.time() - t0, 1),
    }
    (out / "run.json").write_text(json.dumps(doc, indent=1))
    return 0 if not failed else 1


def check(a):
    out = Path(a.dir)
    try:
        meta = json.loads((out / "run.json").read_text())
        lines = (out / "replies.jsonl").read_text().splitlines()
        recs = [json.loads(line) for line in lines if line.strip()]
    except OSError as e:
        print(f"ERROR: no AIME run under {out}: {e}", file=sys.stderr)
        return 1
    if not recs:
        print(f"ERROR: no replies under {out}", file=sys.stderr)
        return 1
    by_id = collections.defaultdict(list)
    for r in recs:
        by_id[r["id"]].append(r)
    k = meta["samples"]
    acc = sum(r["match"] for r in recs) / len(recs)
    # Sampling error of avg@k on this problem set: each problem's share right varies
    # from run to run; the problems themselves are fixed.
    var = sum(
        statistics.variance([r["match"] for r in rs]) / len(rs)
        for rs in by_id.values()
        if len(rs) > 1
    )
    stderr = math.sqrt(var) / len(by_id)
    pass_k = sum(any(r["match"] for r in rs) for rs in by_id.values()) / len(by_id)
    maj_k = sum(
        collections.Counter(r["pred"] for r in rs).most_common(1)[0][0]
        == rs[0]["target"]
        for rs in by_id.values()
    ) / len(by_id)
    toks = [r["completion_tokens"] for r in recs if r["completion_tokens"] is not None]
    capped = sum(r["finish"] == "length" for r in recs)
    unfinished = sum(r["finish"] is None for r in recs)
    print(
        f"AIME ({meta['dataset']}), {meta['problems']} problems x {k} samples, "
        f"{len(recs)}/{meta['expected']} replies: avg@{k} {acc:.4f} +- {stderr:.4f}, "
        f"pass@{k} {pass_k:.4f}, maj@{k} {maj_k:.4f}"
    )
    if toks:
        print(
            f"Output tokens: mean {statistics.mean(toks):.0f}, median "
            f"{statistics.median(toks):.0f}, max {max(toks)}; {capped} at the "
            f"{meta['max_tokens']} cap, {unfinished} without a finish; thinking closed "
            f"{sum(r['thinking_closed'] for r in recs)}/{len(recs)}"
        )
    print(
        f"Wall time: {meta['wall_s']:.0f} s at concurrency {meta['concurrency']}; "
        f"finish {dict(collections.Counter(r['finish'] for r in recs))}"
    )
    if len(recs) < meta["expected"]:
        print(f"FAIL: {meta['expected'] - len(recs)} replies missing")
        return 1
    bar = a.threshold
    verdict = "PASS" if acc + stderr >= bar else "FAIL"
    print(
        f"{verdict}: avg@{k} {acc:.4f} + stderr {stderr:.4f} = {acc + stderr:.4f} vs bar {bar}"
    )
    return int(verdict == "FAIL")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(required=True)
    r = sub.add_parser("run")
    r.add_argument("dir")
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--dataset", default="math-ai/aime25")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--samples", type=int, default=8)
    r.add_argument("--concurrency", type=int, default=32)
    r.add_argument("--max-tokens", type=int, default=98304)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--top-p", type=float, default=0.95)
    r.add_argument("--timeout-s", type=float, default=14400)
    r.set_defaults(func=run)
    c = sub.add_parser("check")
    c.add_argument("dir")
    c.add_argument("--threshold", type=float, required=True)
    c.set_defaults(func=check)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
