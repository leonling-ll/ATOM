#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Prefix-cache hits against cold prefills on a running ATOM server.

``probe`` asks key -> value questions over long lists of random keys, sending every
prompt as token ids (greedy, top-2 logprobs, ``/v1/chat/completions`` with
``prompt_token_ids``: the completions route returns neither logprobs nor cached tokens),
and compares each request served from the prefix cache with the same ids prefilled cold.
Every prompt opens with its own session line, so nothing but its own earlier copies can
hit. For each length and concurrency, with each depth on its own list:

  cold    a list and a question on it, first sent
  hit     the same ids again
  warm    a second list, first sent (fills the cache for ``half``)
  half    that list's first half, a new second half, a question on either half
  turn1   a third list, first sent
  turn2   turn1's ids, its answer's ids, then a suffix of new keys and a question:
          the answer's blocks were published by decode (DSpark verify)

then, once the pool has been turned over (``--flush`` times its size, of random ids:
ATOM has no reset endpoint), the twins: ``cold2`` (cold again: the noise floor),
``half_cold`` and ``turn2_cold``.

Pass: every answer right; a hit and its twin produce the same tokens up to their first
divergence, which counts only where the twin's top-1 minus top-2 logprob is under 0.5,
with each chosen token's logprob within 0.05 before it; per phase, at most the cold2
divergences plus 2. A run counts only where the cache did what the phase needs: a hit
has the shared length cached (to a block, less one), turn2 more than turn1's prompt, a
twin nothing. Per phase it reports DSpark's acceptance length from ``/metrics``.

``snapshot`` saves the server's ``atom:`` counters; ``pair`` compares two lm-eval
``--log_samples`` runs of the same prompts (prefix caching on, off): the flips and the
exact two-sided McNemar p, with the cached share and acceptance length from snapshots.

  pfx_probe.py probe --port 8000 --model /data/models/m3-pro --out probe.jsonl
  pfx_probe.py snapshot --port 8000 --out before.json
  pfx_probe.py pair --on on/samples.jsonl --off off/samples.jsonl --on-metrics b.json a.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BOS = "]~!b["
HEAD = "Below is a list of keys and their values. Each key appears once.\n"
QUESTION = "\nQuestion: What is the value of key {k}?\nAnswer: The value of key {k} is"
BLOCK = 16
NEAR_TIE = 0.5  # a divergence counts only where the twin's top-1 - top-2 is below this
LP_TOL = 0.05  # chosen-token logprob difference allowed before the first divergence
SLACK = 2  # divergences a phase may have beyond the cold2 noise floor
PAIRS = {"hit": "cold", "half": "half_cold", "turn2": "turn2_cold"}


class HFCodec:
    """Token ids of text, and the id of a token's bytes (as the server's logprobs
    report it), from the model's byte-level tokenizer."""

    def __init__(self, path):
        from transformers import AutoTokenizer
        from transformers.convert_slow_tokenizer import bytes_to_unicode

        self.tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        inv = {c: b for b, c in bytes_to_unicode().items()}
        self.ids = {}
        for token, i in self.tok.get_vocab().items():
            try:
                raw = bytes(inv[c] for c in token)
            except KeyError:
                raw = token.encode()
            self.ids[raw] = None if raw in self.ids else i

    def encode(self, texts):
        return self.tok(texts, add_special_tokens=False).input_ids

    def token_id(self, raw):
        return self.ids.get(raw)


class Client:
    def __init__(self, port, model, timeout=3600):
        self.base, self.model, self.timeout = f"http://127.0.0.1:{port}", model, timeout

    def ask(self, ids, max_tokens, top=2):
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": "-"}],
            "prompt_token_ids": ids,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if top:
            body.update(logprobs=True, top_logprobs=top)
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            res = json.loads(r.read())
        ch = res["choices"][0]
        lp, msg = ch.get("logprobs") or {}, ch["message"]
        entries = (lp.get("reasoning_content") or []) + (lp.get("content") or [])
        toks = [
            (bytes(e["bytes"]), e["logprob"], sorted(
                (t["logprob"] for t in e.get("top_logprobs") or []), reverse=True))
            for e in entries
        ]  # fmt: skip
        details = res["usage"].get("prompt_tokens_details") or {}
        return {
            "text": (msg.get("reasoning_content") or "") + (msg.get("content") or ""),
            "toks": toks,
            "cached": details.get("cached_tokens", 0),
            "prompt_tokens": res["usage"]["prompt_tokens"],
        }

    def metrics(self):
        with urllib.request.urlopen(self.base + "/metrics", timeout=30) as r:
            text = r.read().decode()
        out = {}
        for line in text.splitlines():
            if line.startswith("atom:"):
                name, _, val = line.rpartition(" ")
                base = name.split("{", 1)[0].removesuffix("_total")
                try:
                    out[base] = out.get(base, 0.0) + float(val)
                except ValueError:
                    pass
        return out


def deltas(m0, m1):
    """Cached share and DSpark acceptance length between two snapshots."""
    d = {k: m1.get(k, 0.0) - m0.get(k, 0.0) for k in set(m0) | set(m1)}
    full, steps = d.get("atom:prefix_cache_full_tokens", 0), d.get(
        "atom:mtp_decode_steps", 0
    )
    return {
        "cached_tokens": d.get("atom:prefix_cache_cached_tokens", 0),
        "prompt_tokens": full,
        "cached_share": (
            d.get("atom:prefix_cache_cached_tokens", 0) / full if full else None
        ),
        "al": 1 + d.get("atom:mtp_accepted_tokens", 0) / steps if steps else None,
    }


def compare(twin, toks):
    """``toks`` against its cold ``twin`` (lists of (bytes, logprob, top logprobs)):
    where they first part and whether that counts (a near-tie of the twin's), and the
    largest chosen-token logprob difference before it."""
    at, dlp, tie = None, 0.0, None
    for i, (a, b) in enumerate(zip(twin, toks)):
        if a[0] != b[0]:
            at, top = i, a[2]
            tie = len(top) > 1 and top[0] - top[1] < NEAR_TIE
            break
        dlp = max(dlp, abs(a[1] - b[1]))
    if at is None and len(twin) != len(toks):
        # one stopped on a token the logprobs leave out (a special token): its
        # margin is unknown, so it counts as a divergence, not as a failure
        at = min(len(twin), len(toks))
    ok = tie is not False and dlp <= LP_TOL
    return {"diverged": at is not None, "at": at, "near_tie": tie, "dlp": dlp, "ok": ok}


def lines(rng, n):
    keys = set()
    while len(keys) < n:
        keys.add(f"{rng.getrandbits(32):08x}")
    keys = sorted(keys)
    rng.shuffle(keys)
    return [(k, f"{rng.getrandbits(32):08x}") for k in keys]


class Prompts:
    """Prompts built from separately encoded pieces, so a shared prefix is shared ids."""

    def __init__(self, codec, seed):
        self.codec, self.seed = codec, seed
        self.per_line = None

    def n_lines(self, target):
        if self.per_line is None:
            sample = lines(random.Random(0), 200)
            ids = self.codec.encode([f"{k}: {v}\n" for k, v in sample])
            self.per_line = sum(map(len, ids)) / len(ids)
        return max(4, int(target / self.per_line))

    def body(self, tag, n):
        rng = random.Random(f"{self.seed}-{tag}")
        kv = lines(rng, n)
        head = f"{BOS}Session {rng.getrandbits(64):016x}.\n{HEAD}"
        enc = self.codec.encode([head] + [f"{k}: {v}\n" for k, v in kv])
        return kv, enc[0], enc[1:]

    def question(self, key):
        return self.codec.encode([QUESTION.format(k=key)])[0]


def flat(rows):
    return [t for r in rows for t in r]


def depth_at(d, n_depths, n):
    return int((0.02 + 0.96 * d / max(1, n_depths - 1)) * (n - 1))


def build(pr, length, conc, n_depths, suffix):
    """Per depth: the requests of every phase, as (ids, key, want, extra)."""
    n = pr.n_lines(length)
    out = {p: [] for p in ("cold", "warm", "half", "turn1")}
    out["suffix"] = []
    for d in range(n_depths):
        tag = f"{length}-{conc}-{d}"
        kv, head, rows = pr.body(f"a-{tag}", n)
        k, v = kv[depth_at(d, n_depths, n)]
        out["cold"].append((head + flat(rows) + pr.question(k), k, v, {}))
        kv, head, rows = pr.body(f"b-{tag}", n)
        k, v = kv[depth_at(d, n_depths, n)]
        out["warm"].append((head + flat(rows) + pr.question(k), k, v, {}))
        # the first half of the warm list, then new keys; ask either half in turn
        h = n // 2
        new, _, new_rows = pr.body(f"c-{tag}", n - h)
        part = new if d % 2 else kv[:h]
        k, v = part[depth_at(d, n_depths, len(part))]
        shared = head + flat(rows[:h])
        ids = shared + flat(new_rows) + pr.question(k)
        out["half"].append((ids, k, v, {"shared": len(shared)}))
        kv, head, rows = pr.body(f"d-{tag}", n)
        k, v = kv[depth_at(d, n_depths, n)]
        out["turn1"].append((head + flat(rows) + pr.question(k), k, v, {}))
        extra, _, extra_rows = pr.body(f"e-{tag}", pr.n_lines(suffix))
        part = extra if d % 2 else kv
        k, v = part[depth_at(d, n_depths, len(part))]
        tail = pr.codec.encode(["\n"])[0] + flat(extra_rows) + pr.question(k)
        out["suffix"].append((tail, k, v))
    return out


def answer_ok(text, want):
    m = re.search(r"[0-9a-f]{8}", text)
    return bool(m) and m.group(0) == want


class Probe:
    def __init__(self, client, codec, args, log):
        self.c, self.codec, self.a, self.log = client, codec, args, log
        self.res = {}  # (length, conc, phase) -> list of request records
        self.phase_stats = {}

    def phase(self, key, items):
        """Send ``items`` (ids, key, want, extra) at the key's concurrency."""
        length, conc, name = key
        m0 = self.c.metrics()

        def one(item):
            ids, k, want, extra = item
            r = self.c.ask(ids, self.a.max_tokens)
            rec = {"length": length, "conc": conc, "phase": name, "key": k}
            rec.update(want=want, ok=answer_ok(r["text"], want), cached=r["cached"])
            rec.update(prompt_tokens=len(ids), text=r["text"], **extra)
            self.log.write(json.dumps(rec) + "\n")
            return {**rec, "toks": r["toks"], "ids": ids}

        with ThreadPoolExecutor(max_workers=conc) as ex:
            recs = list(ex.map(one, items))
        self.log.flush()
        self.res[key] = recs
        self.phase_stats[key] = deltas(m0, self.c.metrics())
        ok = sum(r["ok"] for r in recs)
        print(f"{length} c{conc} {name:10s} acc {ok}/{len(recs)} "
              f"AL {self.phase_stats[key]['al']}", flush=True)  # fmt: skip
        return recs

    def out_ids(self, toks):
        ids = [self.codec.token_id(t[0]) for t in toks]
        return None if None in ids else ids

    def pre(self, length, conc, plan):
        k = lambda p: (length, conc, p)
        self.phase(k("cold"), plan["cold"])
        self.phase(k("hit"), plan["cold"])
        self.phase(k("warm"), plan["warm"])
        self.phase(k("half"), plan["half"])
        items = []
        for r, (tail, key, want) in zip(self.phase(k("turn1"), plan["turn1"]),
                                        plan["suffix"]):  # fmt: skip
            # turn 1's answer as the ids the server sampled (from its logprobs)
            out = self.out_ids(r["toks"]) or []
            extra = {"turn1": len(r["ids"]), "answer_ids": len(out)}
            items.append((r["ids"] + out + tail, key, want, extra))
        plan["turn2"] = items
        self.phase(k("turn2"), items)

    def post(self, length, conc, plan):
        k = lambda p: (length, conc, p)
        self.phase(k("cold2"), plan["cold"])
        self.phase(k("half_cold"), plan["half"])
        self.phase(k("turn2_cold"), plan["turn2"])

    def flush(self):
        m = self.c.metrics()
        pool = int(m.get("atom:kv_cache_blocks", 0)) * BLOCK
        chunk = max(1024, min(self.a.max_model_len - 2048, 120000))
        n = max(1, math.ceil(self.a.flush * pool / chunk))
        rng = random.Random(f"flush-{self.a.seed}")
        t0 = time.time()
        for _ in range(n):
            self.c.ask([rng.randrange(1000, 100000) for _ in range(chunk)], 1, top=0)
        print(f"flush: {n} x {chunk} tokens over a pool of {pool}, "
              f"{time.time() - t0:.0f}s", flush=True)  # fmt: skip

    def judge(self):
        """Per (length, conc, phase): accuracy, divergences against the twin, and
        whether the cache did what the phase needs."""
        rows, bad, invalid = [], [], []
        twins = {"cold2": "cold", **PAIRS}
        for (length, conc, name), recs in self.res.items():
            row = {"length": length, "conc": conc, "phase": name,
                   "acc": f"{sum(r['ok'] for r in recs)}/{len(recs)}",
                   "cached": sum(r["cached"] for r in recs),
                   "prompt": sum(r["prompt_tokens"] for r in recs),
                   **self.phase_stats[(length, conc, name)]}  # fmt: skip
            if not all(r["ok"] for r in recs):
                bad.append(f"{length}/c{conc}/{name}: wrong answers")
            invalid += [
                f"{length}/c{conc}/{name}/{r['key']}: cached {r['cached']}"
                for r in recs
                if not cache_ok(name, r)
            ]
            twin = self.res.get((length, conc, twins.get(name)))
            if twin is not None:
                cmp = [compare(t["toks"], r["toks"]) for t, r in zip(twin, recs)]
                row["divergences"] = sum(c["diverged"] for c in cmp)
                row["off_tie"] = sum(not c["ok"] for c in cmp)
                row["max_dlp"] = round(max(c["dlp"] for c in cmp), 4)
                if name in PAIRS and row["off_tie"]:
                    bad.append(f"{length}/c{conc}/{name}: {row['off_tie']} off-tie")
            rows.append(row)
        floor = {(r["length"], r["conc"]): r.get("divergences")
                 for r in rows if r["phase"] == "cold2"}  # fmt: skip
        for r in rows:
            n = floor.get((r["length"], r["conc"]))
            if r["phase"] in PAIRS and n is not None and "divergences" in r:
                r["noise_floor"] = n
                if r["divergences"] > n + SLACK:
                    bad.append(f"{r['length']}/c{r['conc']}/{r['phase']}: "
                               f"{r['divergences']} > {n} + {SLACK}")  # fmt: skip
        return rows, bad, invalid


def cache_ok(phase, r):
    """Whether the cache did what ``phase`` needs (else the request does not count)."""
    n = r["prompt_tokens"]
    if phase == "hit":
        return r["cached"] >= n // BLOCK * BLOCK - BLOCK
    if phase == "half":
        return r["cached"] >= r["shared"] // BLOCK * BLOCK - BLOCK
    if phase == "turn2":
        return r["cached"] > r["turn1"] // BLOCK * BLOCK
    if phase in ("cold", "warm", "turn1", "cold2", "half_cold", "turn2_cold"):
        return r["cached"] < BLOCK
    return True


def run_probe(a, client, codec):
    pr = Prompts(codec, a.seed)
    lengths = [int(x) for x in a.lengths.split(",")]
    concs = [int(x) for x in a.concurrency.split(",")]
    with open(a.out, "a") as log:
        p = Probe(client, codec, a, log)
        plans = {}
        for length in lengths:
            for conc in concs:
                plans[length, conc] = build(pr, length, conc, a.depths, a.suffix)
                p.pre(length, conc, plans[length, conc])
        p.flush()
        for (length, conc), plan in plans.items():
            p.post(length, conc, plan)
    rows, bad, invalid = p.judge()
    for r in rows:
        print("ROW " + json.dumps(r), flush=True)
    verdict = "INVALID" if invalid else "FAIL" if bad else "PASS"
    print(f"RESULT {verdict} " + json.dumps({"fail": bad, "invalid": invalid[:20]}))
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}[verdict]


def read_samples(path, filt):
    """doc_id -> right (bool) from an lm-eval ``--log_samples`` file."""
    out = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("filter", filt) != filt:
                continue
            out[rec["doc_id"]] = float(rec["exact_match"]) > 0.5
    return out


def mcnemar(b, c):
    """Exact two-sided McNemar p for ``b`` and ``c`` discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def run_pair(a):
    on, off = read_samples(a.on, a.filter), read_samples(a.off, a.filter)
    docs = sorted(set(on) & set(off))
    lost = sum(off[d] and not on[d] for d in docs)  # right without the cache only
    won = sum(on[d] and not off[d] for d in docs)
    out = {
        "paired": len(docs),
        "score_on": sum(on[d] for d in docs) / max(1, len(docs)),
        "score_off": sum(off[d] for d in docs) / max(1, len(docs)),
        "right_to_wrong": lost,
        "wrong_to_right": won,
        "p": mcnemar(lost, won),
    }
    for tag, snaps in (("on", a.on_metrics), ("off", a.off_metrics)):
        if snaps:
            out[tag] = deltas(*(load_json(s) for s in snaps))
    share = (out.get("on") or {}).get("cached_share")
    fails = []
    if len(docs) != len(on) or len(docs) != len(off):
        fails.append("unpaired documents")
    if out["score_on"] < a.min_score:
        fails.append(f"score {out['score_on']:.4f} < {a.min_score}")
    if share is None or share < a.min_share:
        fails.append(f"cached share {share} < {a.min_share}")
    if out["p"] < a.alpha:
        fails.append(f"McNemar p {out['p']:.4f} < {a.alpha}")
    print(json.dumps(out))
    print("RESULT " + ("PASS" if not fails else "FAIL " + "; ".join(fails)))
    return 1 if fails else 0


def main(argv=None, codec=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True, help="per-request JSON lines (appended)")
    p.add_argument("--lengths", default="32000,64000")
    p.add_argument("--depths", type=int, default=10)
    p.add_argument("--concurrency", default="1,8")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--suffix", type=int, default=2200, help="turn2's new tokens")
    p.add_argument("--flush", type=float, default=1.5, help="pool turnovers")
    p.add_argument("--max-model-len", type=int, default=132096)
    p.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("snapshot")
    s.add_argument("--port", type=int, required=True)
    s.add_argument("--out", required=True)
    q = sub.add_parser("pair")
    q.add_argument("--on", required=True, help="samples, prefix caching on")
    q.add_argument("--off", required=True, help="samples, prefix caching off")
    q.add_argument("--filter", default="flexible-extract")
    q.add_argument("--on-metrics", nargs=2, metavar=("BEFORE", "AFTER"))
    q.add_argument("--off-metrics", nargs=2, metavar=("BEFORE", "AFTER"))
    q.add_argument("--min-score", type=float, default=0.94)
    q.add_argument("--min-share", type=float, default=0.85)
    q.add_argument("--alpha", type=float, default=0.05)
    a = ap.parse_args(argv)
    if a.cmd == "pair":
        return run_pair(a)
    if a.cmd == "snapshot":
        with open(a.out, "w") as f:
            json.dump(Client(a.port, "").metrics(), f)
        return 0
    return run_probe(a, Client(a.port, a.model), codec or HFCodec(a.model))


if __name__ == "__main__":
    sys.exit(main())
