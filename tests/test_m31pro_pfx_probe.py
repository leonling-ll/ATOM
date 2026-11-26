# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for the MiniMax-M3.1-Pro prefix-cache probe (scripts/m31pro/pfx_probe.py),
against a stand-in server: a prefix cache over token ids with LRU eviction, greedy
answers read off the prompt, logprobs, and the /metrics counters the probe reads."""

import importlib.util
import json
import re
import threading
import time
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


SPECIAL = (
    1  # a token id the logprobs leave out (no character: the codec never makes it)
)


class Server:
    """What the probe sees of ATOM: prompts as ids, a block-granular prefix cache that
    keeps the last block back and indexes the answer's blocks too, LRU over ``cap``, and
    /metrics counters from a snapshot refreshed every ``refresh`` seconds. Knobs:
    ``flip`` (index: the character there turns, confidently, on a hit), ``repeat_flip``
    (it turns on every repeat of a prompt, cached or not: run-to-run noise),
    ``special`` (odd-sum prompts get a special token after the answer)."""

    def __init__(self, cap=8000, refresh=0.05):
        self.cap, self.cache, self.refresh = cap, True, refresh
        self.flip, self.repeat_flip, self.special = None, False, False
        self.store, self.seen, self.lock = [], set(), threading.Lock()
        self.m = {"cached": 0, "full": 0, "acc": 0, "steps": 0}
        self.snap, self.stamp = dict(self.m), time.time()
        self.stop = threading.Event()
        threading.Thread(target=self._refresher, daemon=True).start()

    def _refresher(self):
        while not self.stop.wait(self.refresh):
            with self.lock:
                self.snap, self.stamp = dict(self.m), time.time()

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
            again = tuple(ids) in self.seen
            self.seen.add(tuple(ids))
            at = 12 if self.repeat_flip and again else self.flip if hit else None
            if at is not None and n > at:
                out = out[:at] + "#" + out[at + 1 :]  # a confident token turns
            out_ids = [ord(c) + 3 for c in out]
            if self.special and sum(ids) % 2 and n > 12:
                out_ids = out_ids[:10] + [SPECIAL] + out_ids[11:]
                out = out[:10] + out[11:]
            self.store.append(ids + out_ids)
            while sum(map(len, self.store)) > self.cap:
                self.store.pop(0)
            self.m["cached"] += hit
            self.m["full"] += len(ids)
            self.m["acc"] += 2 * len(out_ids)
            self.m["steps"] += len(out_ids)
        return out, hit, len(out_ids)

    def metrics(self):
        with self.lock:
            m, stamp = dict(self.snap), self.stamp
        return (
            "# HELP atom:mtp_accepted_tokens x\n"
            f"atom:metrics_last_refresh_timestamp_seconds {stamp}\n"
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
            out, hit, n_out = state.complete(ids, n)
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
                    "completion_tokens": n_out,
                    "prompt_tokens_details": {"cached_tokens": hit},
                },
            }
            self._send(json.dumps(res))

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    state.port = httpd.server_address[1]
    yield state
    state.stop.set()
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
    text = capsys.readouterr().out
    rows = [json.loads(x[4:]) for x in text.splitlines() if x.startswith("ROW ")]
    return rc, recs, rows, text


def test_probe_passes_on_a_faithful_cache(server, tmp_path, capsys):
    rc, recs, rows, out = _probe(server, tmp_path, capsys)
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
    # every token logged: bytes, chosen logprob, top logprobs
    assert all(len(r["toks"]) == 40 and len(r["toks"][0]) == 3 for r in recs)
    # the phase's counters are read after the server's refresh, not before it
    for r in rows:
        assert r["prompt_tokens"] == r["req_prompt"], r
        assert r["cached_tokens"] == r["req_cached"], r
    assert "RESULT PASS" in out


def test_a_confident_token_turning_on_a_hit_fails(server, tmp_path, capsys):
    server.flip = 12  # after the answer
    rc, _, _, out = _probe(server, tmp_path, capsys)
    assert rc == 1
    assert "off_tie 3 > 0 + 2" in out.splitlines()[-1]


def test_an_answer_turning_on_a_hit_fails(server, tmp_path, capsys):
    server.flip = 3  # inside the 8-hex answer
    rc, _, _, out = _probe(server, tmp_path, capsys)
    assert rc == 1
    assert "answers differ" in out.splitlines()[-1]


def test_hits_at_the_noise_floor_pass(server, tmp_path, capsys):
    # every repeat of a prompt turns a confident token, cold or cached: cold2 parts
    # from cold as much as the hits do
    server.repeat_flip = True
    rc, _, rows, out = _probe(server, tmp_path, capsys)
    assert rc == 0, out
    hit = [r for r in rows if r["phase"] == "hit"]
    assert all(r["off_tie"] == r["floor"]["off_tie"] == 3 for r in hit)


def test_turn1_answers_with_a_left_out_token_are_not_judged(server, tmp_path, capsys):
    server.special = True
    rc, recs, _, out = _probe(server, tmp_path, capsys)
    assert rc == 0, out
    turn2 = [r for r in recs if r["phase"] == "turn2"]
    gaps = [r for r in turn2 if r["answer_gap"]]
    assert 0 < len(gaps) < len(turn2)
    # the cache matched up to the left-out token's block only
    assert all(r["cached"] <= (r["turn1"] + 10) // BS * BS for r in gaps)
    assert all(
        r["cached"] > r["turn1"] // BS * BS for r in turn2 if not r["answer_gap"]
    )


def test_a_run_without_hits_does_not_count(server, tmp_path, capsys):
    server.cache = False
    rc, _, _, out = _probe(server, tmp_path, capsys)
    assert rc == 2 and "RESULT INVALID" in out


def _toks(text, margin=5.0, lp=-0.01):
    return [(c.encode(), lp, [lp, lp - margin]) for c in text]


def test_divergence_rule():
    twin = _toks("abcdef")
    assert probe.compare(twin, _toks("abcdef")) == {
        "diverged": False, "at": None, "near_tie": None, "off_tie": False,
        "dlp": 0.0, "strict": True,
    }  # fmt: skip
    # parting where the twin was sure: an off-tie divergence
    got = probe.compare(twin, _toks("abcxef"))
    assert got["diverged"] and got["at"] == 3 and got["off_tie"] and not got["strict"]
    # parting at a near-tie of the twin's: a divergence, not an off-tie one
    tie = twin[:3] + _toks("d", margin=0.3) + twin[4:]
    got = probe.compare(tie, _toks("abcxef"))
    assert got["diverged"] and got["near_tie"] and not got["off_tie"] and got["strict"]
    # same tokens, logprobs apart before the parting: reported against the 0.05
    got = probe.compare(twin, _toks("ab") + _toks("cdef", lp=-0.2))
    assert not got["diverged"] and got["dlp"] == pytest.approx(0.19)
    assert not got["strict"] and not got["off_tie"]
    # one stopped early on a token the logprobs leave out
    got = probe.compare(twin, _toks("abcd"))
    assert got["diverged"] and got["at"] == 4 and not got["off_tie"]


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


def _log(path, cold2_dlp, hit_dlp=0.0, cold2_flips=0, hit_flips=0):
    """A logged run of one cell (400 tokens, c1, 3 depths): cold, hit and cold2, with
    ``*_dlp`` added to one chosen-token logprob before any parting and the first
    ``*_flips`` requests turning a confident token after the answer. The hit lines
    are written in reverse, as a c8 phase's log is in completion order."""

    def rec(phase, d, dlp, flip, cached):
        text = f" {d:08x}." + "\nQuestion: next" * 4
        toks = [[c.encode().hex(), -0.01, [-0.01, -5.0]] for c in text[:40]]
        toks[6][1] -= dlp
        if flip:
            toks[12][0] = b"#".hex()
        return {"length": 400, "conc": 1, "phase": phase, "key": f"k{d}",
                "want": f"{d:08x}", "ok": True, "cached": cached,
                "prompt_tokens": 400, "text": text, "completion_tokens": 40,
                "toks": toks}  # fmt: skip

    recs = [rec("cold", d, 0.0, False, 0) for d in range(3)]
    recs += [rec("hit", d, hit_dlp, d < hit_flips, 384) for d in reversed(range(3))]
    recs += [rec("cold2", d, cold2_dlp, d < cold2_flips, 0) for d in range(3)]
    path.write_text("".join(json.dumps(r) + "\n" for r in recs))
    return str(path)


def _judge(capsys, *runs):
    argv = ["judge"]
    for r in runs:
        argv += ["--run", r]
    rc = probe.main(argv)
    out = capsys.readouterr().out
    rows = [json.loads(x[4:]) for x in out.splitlines() if x.startswith("ROW ")]
    return rc, out, rows


def test_dlp_floor_is_the_largest_cold2_over_runs(tmp_path, capsys):
    quiet = _log(tmp_path / "a.jsonl", cold2_dlp=0.1, hit_dlp=0.6)
    loud = _log(tmp_path / "b.jsonl", cold2_dlp=1.3, cold2_flips=3)
    # alone, the hit's gap is past this run's floor
    rc, out, _ = _judge(capsys, quiet)
    assert rc == 1 and "400/c1/hit: max_dlp 0.6 > 0.1 + 0.25" in out
    # with a second run of the same prompts, the floor is the larger cold2 gap
    rc, out, rows = _judge(capsys, quiet, loud)
    assert rc == 0, out
    hit = [r for r in rows if r["phase"] == "hit"]
    assert [r["floor"]["max_dlp"] for r in hit] == [1.3, 1.3]
    # twins pair by key: the reversed hit log parts nowhere
    assert [r["divergences"] for r in hit] == [0, 0]
    # divergences and off-tie ones stay judged against the same run's floor
    flips = _log(tmp_path / "c.jsonl", cold2_dlp=0.1, hit_flips=3)
    rc, out, _ = _judge(capsys, flips, loud)
    assert rc == 1
    assert "run0 FAIL" in out and "400/c1/hit: off_tie 3 > 0 + 2" in out
