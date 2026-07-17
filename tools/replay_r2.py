"""Replay the ROUND-2 trace (input_part2/trace_grading_public.jsonl) against an
OpenAI-compatible endpoint and score it with the round-2 constants.

The public trace has NO prompts (BTC keeps them), only sizes + arrival times,
so prompts are synthesized. Two regimes bracket reality:
  --mode shared  each conversation = one fixed ~3.9k-token context + a small
                 per-turn question suffix  -> prefix-cache BEST case
  --mode fresh   every turn is brand-new random ~4k tokens -> APC WORST case
Prompts are seeded per (mode, conv_id, turn_idx): identical across server
configs, so config A vs B comparisons are apples-to-apples.

Timing is closed-loop like BTC: conversation turn 0 fires at timestamp_ms;
turn k+1 fires think_ms after turn k's response completes. Warmup rows
(in_warmup=true) are sent but excluded from ERS.

Run INSIDE the GPU pod (localhost). Requires: pip install aiohttp
(transformers + the model tokenizer are used if available, else a chars/3
heuristic sizes the prompts).

Usage:
  python replay_r2.py --url http://localhost:8001 --mode shared
  python replay_r2.py --url http://localhost:8001 --mode fresh
  python replay_r2.py --url http://localhost:8001 --mode shared --limit-convs 5   # smoke
"""
import argparse
import asyncio
import json
import os
import random
import statistics
import time

import aiohttp

TRACE = os.path.join(os.path.dirname(__file__), "..", "input_part2", "trace_grading_public.jsonl")

# Round-2 scoring constants
F_TTFT, C_TTFT = 10.0, 400.0   # ms
F_TPOT, C_TPOT = 1.0, 10.0     # ms
GAMMA, W = 2.0, 0.5

WORDS = (
    "time year people way day man thing woman life child world school state family "
    "student group country problem hand part place case week company system program "
    "question work night point home water room mother area money story fact month lot "
    "right study book eye job word business issue side kind head house service friend "
    "father power hour game line end member law car city community name team minute "
    "idea body back parent face others level office door health person art war history "
    "party result change morning reason research girl guy moment air teacher force "
    "education foot boy age policy process music market sense nation plan college "
    "interest death experience effect use class control care field development role "
    "effort rate heart drug show leader light voice wife whole police mind price "
    "report decision son view relationship town road arm difference value building "
    "action model season society tax director position player record paper space "
    "ground form event official matter center couple site project activity star table "
    "need court oil situation cost industry figure street image phone data picture "
    "practice piece land product doctor wall patient worker news test movie north "
    "love support technology"
).split()


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def request_score(ttft_ms, tpot_ms):
    s_ttft = clamp((C_TTFT - ttft_ms) / (C_TTFT - F_TTFT)) ** GAMMA
    s_tpot = clamp((C_TPOT - tpot_ms) / (C_TPOT - F_TPOT)) ** GAMMA
    return W * s_ttft + (1 - W) * s_tpot


class PromptFactory:
    """Sizes synthetic prompts in real tokenizer tokens when possible."""

    def __init__(self, tokenizer_path):
        self.tok = None
        if tokenizer_path:
            try:
                from transformers import AutoTokenizer
                self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
                print(f"[prompts] sizing with tokenizer at {tokenizer_path}")
            except Exception as e:
                print(f"[prompts] tokenizer unavailable ({e}); falling back to chars/3")

    def _count(self, text):
        if self.tok is not None:
            return len(self.tok.encode(text, add_special_tokens=False))
        return max(1, len(text) // 3)

    def make_text(self, target_tokens, rng):
        # start from an estimate, then linearly correct twice — lands within ~1%
        words = [rng.choice(WORDS) for _ in range(int(target_tokens * 0.9))]
        for _ in range(3):
            text = " ".join(words)
            n = self._count(text)
            if abs(n - target_tokens) <= max(4, target_tokens // 200):
                break
            if n < target_tokens:
                words.extend(rng.choice(WORDS) for _ in range(int((target_tokens - n) * 0.9) + 1))
            else:
                del words[int(len(words) * target_tokens / n):]
        return " ".join(words)


def build_prompts(rows, mode, tokenizer_path):
    """Returns {(conv_id, turn_idx): prompt}. Seeded => identical across runs."""
    pf = PromptFactory(tokenizer_path)
    prompts = {}
    convs = {}
    for r in rows:
        convs.setdefault(r["conv_id"], []).append(r)
    for cid, rs in sorted(convs.items()):
        rs.sort(key=lambda r: r["turn_idx"])
        if mode == "shared":
            ctx_rng = random.Random(10_000 + cid)
            ctx_tokens = min(r["in_tokens_est"] for r in rs) - 100
            context = pf.make_text(ctx_tokens, ctx_rng)
            for r in rs:
                qrng = random.Random(20_000 + cid * 100 + r["turn_idx"])
                q = pf.make_text(80, qrng)
                prompts[(cid, r["turn_idx"])] = (
                    f"{context}\n\nQuestion {r['turn_idx']}: {q}\nAnswer in detail."
                )
        else:  # fresh
            for r in rs:
                rng = random.Random(30_000 + cid * 100 + r["turn_idx"])
                prompts[(cid, r["turn_idx"])] = pf.make_text(r["in_tokens_est"], rng)
    return prompts


async def send_one(session, url, model_name, prompt, max_tokens, ignore_eos):
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        body["ignore_eos"] = True
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
    decode_s = t_last - (start + ttft)
    tpot_ms = (decode_s / (ntok - 1) * 1000) if ntok > 1 else 0.0
    return {"ttft_ms": ttft * 1000, "tpot_ms": tpot_ms, "ntok": ntok}


async def run_conv(session, url, model_name, conv_rows, prompts, t0, scale, ignore_eos, results):
    conv_rows = sorted(conv_rows, key=lambda r: r["turn_idx"])
    arrival = conv_rows[0]["timestamp_ms"] / 1000.0 * scale
    delay = arrival - (time.perf_counter() - t0)
    if delay > 0:
        await asyncio.sleep(delay)
    for r in conv_rows:
        res = await send_one(
            session, url, model_name,
            prompts[(r["conv_id"], r["turn_idx"])],
            r["out_tokens_max"], ignore_eos,
        )
        results[(r["conv_id"], r["turn_idx"])] = res
        await asyncio.sleep(r["think_ms"] / 1000.0 * scale)


def pct(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q * len(vals)))]


def summarize(rows, results, label):
    scored = [r for r in rows if not r["in_warmup"]]
    warm = [r for r in rows if r["in_warmup"]]

    def stats(part, name):
        rs = [results[(r["conv_id"], r["turn_idx"])] for r in part]
        errs = sum(1 for x in rs if "error" in x)
        ok = [x for x in rs if "error" not in x]
        scores = [0.0 if "error" in x else request_score(x["ttft_ms"], x["tpot_ms"]) for x in rs]
        ers = sum(scores) / len(scores) if scores else 0.0
        print(f"\n=== {label} [{name}] ERS = {ers:.4f} (~{100*ers:.2f} points)  errors {errs}/{len(rs)} ===")
        if ok:
            t = [x["ttft_ms"] for x in ok]
            p = [x["tpot_ms"] for x in ok]
            print(f"  ttft_ms: p50={pct(t,0.5):.0f} p95={pct(t,0.95):.0f} mean={statistics.mean(t):.0f} max={max(t):.0f}")
            print(f"  tpot_ms: p50={pct(p,0.5):.2f} p95={pct(p,0.95):.2f} mean={statistics.mean(p):.2f}")
        return ers

    ers = stats(scored, "SCORED")
    # per-turn TTFT on scored: turn0 = cold conv start; turns 1-5 show APC effect
    print("  scored TTFT p50 by turn_idx:", end=" ")
    for t in sorted(set(r["turn_idx"] for r in scored)):
        vals = [results[(r["conv_id"], r["turn_idx"])].get("ttft_ms")
                for r in scored if r["turn_idx"] == t]
        vals = [v for v in vals if v is not None]
        print(f"t{t}={pct(vals,0.5):.0f}" if vals else f"t{t}=err", end=" ")
    print()
    if warm:
        stats(warm, "warmup (not scored)")
    return ers


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--mode", choices=("shared", "fresh"), default="shared")
    ap.add_argument("--model-name", default="LFM2.5-1.2B-Instruct")
    ap.add_argument("--tokenizer", default="/workspace/model",
                    help="path for token-accurate prompt sizing; '' disables")
    ap.add_argument("--trace", default=TRACE)
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="<1 compresses the schedule (raises concurrency!) — smoke only")
    ap.add_argument("--limit-convs", type=int, default=None)
    ap.add_argument("--no-ignore-eos", action="store_true",
                    help="let the model stop early (BTC-like); default forces 200 tokens")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.trace, encoding="utf-8")]
    if args.limit_convs is not None:
        keep = sorted(set(r["conv_id"] for r in rows))[: args.limit_convs]
        rows = [r for r in rows if r["conv_id"] in keep]

    print(f"[replay] {len(rows)} requests, mode={args.mode}, scale={args.time_scale}")
    prompts = build_prompts(rows, args.mode, args.tokenizer or None)

    convs = {}
    for r in rows:
        convs.setdefault(r["conv_id"], []).append(r)

    results = {}
    t0 = time.perf_counter()
    conn = aiohttp.TCPConnector(limit=256)
    async with aiohttp.ClientSession(connector=conn) as session:
        await asyncio.gather(*(
            run_conv(session, args.url, args.model_name, cr, prompts, t0,
                     args.time_scale, not args.no_ignore_eos, results)
            for cr in convs.values()
        ))
    print(f"[replay] wall time {time.perf_counter()-t0:.0f}s")
    summarize(rows, results, f"mode={args.mode}")


if __name__ == "__main__":
    asyncio.run(main())
