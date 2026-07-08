"""Replay the trace against an OpenAI-compatible endpoint and score it like BTC.

Run this INSIDE the GPU pod (localhost) so network jitter doesn't pollute TTFT.
Requires: pip install aiohttp

Usage:
  # Key experiment: does the prefix cache give a warm hit? (cold vs warm TTFT)
  python replay_trace.py --url http://localhost:8000 --probe 0

  # Full run like BTC: pass 1 = primer (unscored), pass 2 = scored, waves every 5s
  python replay_trace.py --url http://localhost:8000 --passes 2

  # Quick smoke: first N requests only, no wave timing
  python replay_trace.py --url http://localhost:8000 --limit 5 --no-timing
"""
import argparse
import asyncio
import json
import os
import time

import aiohttp

TRACE = os.path.join(os.path.dirname(__file__), "..", "input", "trace-round1.jsonl")

# BTC scoring constants
F_TTFT, C_TTFT = 100.0, 1500.0   # ms
F_TPOT, C_TPOT = 20.0, 45.0      # ms
GAMMA, W = 2.0, 0.5


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def request_score(ttft_ms, tpot_ms):
    s_ttft = clamp((C_TTFT - ttft_ms) / (C_TTFT - F_TTFT)) ** GAMMA
    s_tpot = clamp((C_TPOT - tpot_ms) / (C_TPOT - F_TPOT)) ** GAMMA
    return W * s_ttft + (1 - W) * s_tpot


async def send_one(session, url, body):
    """POST with streaming; returns dict with ttft_ms, tpot_ms, ntok or error."""
    body = dict(body)
    body["stream"] = True
    body["stream_options"] = {"include_usage": True}
    start = time.perf_counter()
    ttft = None
    t_last = None
    chunks = 0
    completion_tokens = None
    try:
        async with session.post(
            f"{url}/v1/chat/completions", json=body,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            if resp.status != 200:
                return {"error": f"http {resp.status}: {(await resp.text())[:200]}"}
            async for raw in resp.content:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    completion_tokens = chunk["usage"].get("completion_tokens")
                if not chunk.get("choices"):
                    continue
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - start
                    t_last = now
                    chunks += 1
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if ttft is None or chunks == 0:
        return {"error": "0 tokens"}
    ntok = completion_tokens or chunks  # chunks undercount with speculative decoding
    decode_s = t_last - (start + ttft)  # time from first token to last token
    tpot_ms = (decode_s / (ntok - 1) * 1000) if ntok > 1 else 0.0
    return {"ttft_ms": ttft * 1000, "tpot_ms": tpot_ms, "ntok": ntok}


async def run_pass(rows, url, timed, concurrency_note=""):
    results = [None] * len(rows)
    t0 = time.perf_counter()

    async with aiohttp.ClientSession() as session:
        async def worker(i, row):
            if timed:
                delay = row["timestamp_ms"] / 1000 - (time.perf_counter() - t0)
                if delay > 0:
                    await asyncio.sleep(delay)
            results[i] = await send_one(session, url, row["body"])

        await asyncio.gather(*(worker(i, r) for i, r in enumerate(rows)))
    return results


def summarize(rows, results, label):
    scores, errs = [], 0
    for r in results:
        if "error" in r:
            scores.append(0.0)
            errs += 1
        else:
            scores.append(request_score(r["ttft_ms"], r["tpot_ms"]))
    ers = sum(scores) / len(scores)
    ok = [r for r in results if "error" not in r]
    print(f"\n=== {label}: ERS = {ers:.4f}  (score ~ {100*ers:.2f} before accuracy gate) ===")
    print(f"errors: {errs}/{len(results)}")
    if ok:
        for key in ("ttft_ms", "tpot_ms"):
            vals = sorted(r[key] for r in ok)
            p = lambda q: vals[min(len(vals) - 1, int(q * len(vals)))]
            print(f"{key}: p50={p(0.5):.0f} p90={p(0.9):.0f} max={vals[-1]:.0f}")
    # per-wave breakdown
    waves = {}
    for row, s in zip(rows, scores):
        waves.setdefault(row["timestamp_ms"] // 5000, []).append(s)
    print("per-wave mean score:", {w: round(sum(v) / len(v), 3) for w, v in sorted(waves.items())})
    return ers


async def probe(url, rows, idx):
    """Send the same request twice sequentially: 2nd TTFT << 1st => prefix cache works."""
    async with aiohttp.ClientSession() as session:
        for run in ("cold", "warm"):
            r = await send_one(session, url, rows[idx]["body"])
            print(f"probe req {idx} [{run}]: {r}")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--passes", type=int, default=1, help="1 = cold run; 2 = primer + scored")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-timing", action="store_true", help="fire all at once, ignore waves")
    ap.add_argument("--probe", type=int, default=None, help="send request N twice, print cold/warm TTFT")
    ap.add_argument("--trace", default=TRACE)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.trace, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]

    if args.probe is not None:
        await probe(args.url, rows, args.probe)
        return

    for p in range(args.passes):
        label = f"pass {p+1}/{args.passes}" + (" (scored)" if p == args.passes - 1 else " (primer)")
        print(f"\n--- {label}: {len(rows)} requests, timed={not args.no_timing} ---")
        results = await run_pass(rows, args.url, timed=not args.no_timing)
        summarize(rows, results, label)


if __name__ == "__main__":
    asyncio.run(main())
