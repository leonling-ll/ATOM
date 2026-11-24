# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the MiniMax-M3.1-Pro prefix-cache probe (scripts/m31pro/pfx_probe.py),
against a stand-in server: a prefix cache over token ids with LRU eviction, greedy
answers read off the prompt, logprobs, and the /metrics counters the probe reads."""

import importlib.util
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/m31pro/pfx_probe.py"
SPEC = importlib.util.spec_from_file_location("m31pro_pfx_probe", SCRIPT)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

BS = 16


class Codec:
    """One token a character."""

    def encode(self, texts):
        return [[ord(c) + 3 for c in t] for t in texts]

    def token_id(self, raw):
        return raw[0] + 3 if len(raw) == 1 else None


class Server:
    """What the probe sees of ATOM: prompts as ids, a block-granular prefix cache that
    keeps the last block back and indexes the answer's blocks too, LRU over ``cap``."""

    def __init__(self, cap=8000, cache=True, flip_on_hit=False):
        self.cap, self.cache, self.flip_on_hit = cap, cache, flip_on_hit
        self.store, self.lock = [], threading.Lock()
        self.m = {"cached": 0, "full": 0, "acc": 0, "steps": 0}

    def cached(self, ids):
        best = 0
        for s in self.store:
            n = 0
            for a, b in zip(ids, s):
                if a != b:
                    break
                n += 1
            best = max(best, n)
        return min(best // BS, (len(ids) - 1) // BS) * BS if self.cache else 0

    def complete(self, ids, n):
        text = "".join(chr(i - 3) for i in ids)
        keys = re.findall(r"value of key ([0-9a-f]{8})", text)
        val = (
            re.search(rf"^{keys[-1]}: ([0-9a-f]{{8}})$", text, re.MULTILINE)
            if keys
            else None
        )
        out = (f" {val.group(1) if val else '?'}." + "\nQuestion: next" * 4)[:n]
        with self.lock:
            hit = self.cached(ids)
            if hit and self.flip_on_hit and n > 12:
                out = out[:12] + "#" + out[13:]  # a confident token turns
            self.store.append(ids + [ord(c) + 3 for c in out])
            while sum(map(len, self.store)) > self.cap:
                self.store.pop(0)
            self.m["cached"] += hit
            self.m["full"] += len(ids)
            self.m["acc"] += 2 * len(out)
            self.m["steps"] += len(out)
        return out, hit

    def metrics(self):
        m = self.m
        return (
            "# HELP atom:mtp_accepted_tokens x\n"
            f"atom:kv_cache_blocks_total {self.cap // BS}\n"
            f"atom:prefix_cache_cached_tokens {m['cached']}\n"
            f"atom:prefix_cache_full_tokens {m['full']}\n"
            f"atom:mtp_accepted_tokens_total {m['acc']}\n"
            f'atom:mtp_decode_steps_total{{accepted_tokens="2"}} {m["steps"]}\n'
        )


@pytest.fixture
def server():
    state = Server()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json"):
            b = body.encode()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send(state.metrics(), "text/plain")

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            ids, n = req["prompt_token_ids"], req["max_tokens"]
            out, hit = state.complete(ids, n)
            ent = [
                {
                    "token": c,
                    "bytes": [ord(c)],
                    "logprob": -0.01,
                    "top_logprobs": [{"logprob": -0.01}, {"logprob": -5.0}],
                }
                for c in out
            ]
            res = {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": out},
                        "logprobs": {"content": ent} if req.get("logprobs") else None,
                    }
                ],
                "usage": {
                    "prompt_tokens": len(ids),
                    "completion_tokens": len(out),
                    "prompt_tokens_details": {"cached_tokens": hit},
                },
            }
            self._send(json.dumps(res))

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state.port = httpd.server_address[1]
    yield state
    httpd.shutdown()


def _probe(server, tmp_path, capsys):
    out = tmp_path / "probe.jsonl"
    rc = probe.main(
        ["probe", "--port", str(server.port), "--model", "m", "--out", str(out),
         "--lengths", "400", "--depths", "3", "--concurrency", "1,2",
         "--max-tokens", "40", "--suffix", "60", "--max-model-len", "4096"],
        codec=Codec(),
    )  # fmt: skip
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    return rc, recs, capsys.readouterr().out


def test_probe_passes_on_a_faithful_cache(server, tmp_path, capsys):
    rc, recs, out = _probe(server, tmp_path, capsys)
    assert rc == 0, out
    by = {}
    for r in recs:
        by.setdefault(r["phase"], []).append(r)
    assert len(by) == 9 and all(len(v) == 6 for v in by.values())
    assert all(r["ok"] for r in recs)
    assert all(r["cached"] >= r["prompt_tokens"] - 2 * BS for r in by["hit"])
    # turn 2 hit the blocks turn 1's answer filled, past turn 1's prompt
    assert all(r["cached"] > r["turn1"] // BS * BS for r in by["turn2"])
    for twin in ("cold", "cold2", "half_cold", "turn2_cold"):
        assert all(r["cached"] == 0 for r in by[twin]), twin
    assert "RESULT PASS" in out


def test_a_confident_token_turning_on_a_hit_fails(server, tmp_path, capsys):
    server.flip_on_hit = True
    rc, _, out = _probe(server, tmp_path, capsys)
    assert rc == 1
    assert "off-tie" in out.splitlines()[-1]


def test_a_run_without_hits_does_not_count(server, tmp_path, capsys):
    server.cache = False
    rc, _, out = _probe(server, tmp_path, capsys)
    assert rc == 2 and "RESULT INVALID" in out


def _toks(text, margin=5.0, lp=-0.01):
    return [(c.encode(), lp, [lp, lp - margin]) for c in text]


def test_divergence_rule():
    twin = _toks("abcdef")
    assert probe.compare(twin, _toks("abcdef")) == {
        "diverged": False, "at": None, "near_tie": None, "dlp": 0.0, "ok": True,
    }  # fmt: skip
    # parting where the twin was sure: a failure
    got = probe.compare(twin, _toks("abcxef"))
    assert got["diverged"] and got["at"] == 3 and got["near_tie"] is False
    assert not got["ok"]
    # parting at a near-tie of the twin's: a divergence, not a failure
    tie = twin[:3] + _toks("d", margin=0.3) + twin[4:]
    got = probe.compare(tie, _toks("abcxef"))
    assert got["diverged"] and got["near_tie"] and got["ok"]
    # same tokens, a logprob off by more than the tolerance before the parting
    assert not probe.compare(twin, _toks("ab") + _toks("cdef", lp=-0.2))["ok"]
    # one stopped early on a token the logprobs leave out
    got = probe.compare(twin, _toks("abcd"))
    assert got["diverged"] and got["at"] == 4 and got["ok"]


def _samples(path, rights):
    with open(path, "w") as f:
        for i, r in enumerate(rights):
            for filt, em in (("strict-match", 0.0), ("flexible-extract", float(r))):
                f.write(json.dumps({"doc_id": i, "filter": filt, "exact_match": em}))
                f.write("\n")
    return str(path)


def _snap(path, cached, full):
    path.write_text(
        json.dumps(
            {
                "atom:prefix_cache_cached_tokens": cached,
                "atom:prefix_cache_full_tokens": full,
                "atom:mtp_accepted_tokens": 2 * full / 100,
                "atom:mtp_decode_steps": full / 100,
            }
        )
    )
    return str(path)


def test_mcnemar_exact():
    assert probe.mcnemar(0, 0) == 1.0
    assert probe.mcnemar(1, 9) == pytest.approx(2 * 11 / 1024)
    assert probe.mcnemar(5, 5) == 1.0


@pytest.mark.parametrize("flips,share,rc", [(0, 0.9, 0), (9, 0.9, 1), (0, 0.5, 1)])
def test_pair(tmp_path, capsys, flips, share, rc):
    off = [1] * 95 + [0] * 5
    on = list(off)
    for i in range(flips):  # right without the cache, wrong with it
        on[i] = 0
    on[-1] = 1  # and one the other way
    got = probe.main(
        ["pair", "--on", _samples(tmp_path / "on.jsonl", on),
         "--off", _samples(tmp_path / "off.jsonl", off),
         "--on-metrics", _snap(tmp_path / "b.json", 0, 0),
         _snap(tmp_path / "a.json", share * 1000, 1000), "--min-score", "0.9"]
    )  # fmt: skip
    res = json.loads(capsys.readouterr().out.splitlines()[0])
    assert got == rc
    assert res["right_to_wrong"] == flips and res["wrong_to_right"] == 1
    assert res["on"]["cached_share"] == pytest.approx(share)
    assert res["on"]["al"] == pytest.approx(3.0)
