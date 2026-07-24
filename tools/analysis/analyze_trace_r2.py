# Analyze the round-2 public trace (arrival + token counts only; prompts withheld by BTC).
# Usage: python tools/analysis/analyze_trace_r2.py
import json
import statistics
from collections import defaultdict
from pathlib import Path

TRACE = (
    Path(__file__).resolve().parents[2]
    / "input_part2"
    / "trace_grading_public.jsonl"
)

rows = [json.loads(l) for l in TRACE.read_text().splitlines() if l.strip()]
print(f"total rows: {len(rows)}")

marked_warm = [r for r in rows if r["in_warmup"]]
marked_regular = [r for r in rows if not r["in_warmup"]]
print(
    f"stale in_warmup marker: {len(marked_warm)} rows "
    f"(convs: {sorted(set(r['conv_id'] for r in marked_warm))})"
)
print(
    f"rows without marker: {len(marked_regular)} "
    f"(convs: {len(set(r['conv_id'] for r in marked_regular))})"
)
print(f"CURRENT BTC scored rows: {len(rows)} (all requests; no warm-up)")

for name, part in (
    ("stale in_warmup marker", marked_warm),
    ("rows without marker", marked_regular),
):
    convs = defaultdict(list)
    for r in part:
        convs[r["conv_id"]].append(r)
    turns = [len(v) for v in convs.values()]
    print(f"\n[{name}] {len(convs)} conversations, turns/conv: min={min(turns)} max={max(turns)} "
          f"mean={statistics.mean(turns):.2f}  total={sum(turns)}")
    tc = sorted(turns)
    print(f"  turns distribution: {dict((t, tc.count(t)) for t in sorted(set(tc)))}")

    arrivals = sorted(r["timestamp_ms"] for r in part if r["turn_idx"] == 0)
    print(f"  conv arrivals (turn0 timestamp_ms): first={arrivals[0]} last={arrivals[-1]} "
          f"span={(arrivals[-1]-arrivals[0])/1000:.1f}s")
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    if gaps:
        print(f"  inter-arrival gaps ms: min={min(gaps)} p50={statistics.median(gaps):.0f} "
              f"mean={statistics.mean(gaps):.0f} max={max(gaps)}")

    ic = [r["in_chars"] for r in part]
    it = [r["in_tokens_est"] for r in part]
    ot = [r["out_tokens_max"] for r in part]
    tm = [r["think_ms"] for r in part]
    print(f"  in_chars: min={min(ic)} p50={statistics.median(ic):.0f} max={max(ic)}")
    print(f"  in_tokens_est: min={min(it)} p50={statistics.median(it):.0f} max={max(it)}  sum={sum(it)}")
    print(f"  out_tokens_max: values={sorted(set(ot))}")
    print(f"  think_ms: values={sorted(set(tm))}")

# Does input grow with turn index (prefix accumulation) or stay flat?
print("\nin_tokens_est by turn_idx (all 420 currently scored rows):")
byturn = defaultdict(list)
for r in rows:
    byturn[r["turn_idx"]].append(r["in_tokens_est"])
for t in sorted(byturn):
    v = byturn[t]
    print(f"  turn {t}: n={len(v)} min={min(v)} mean={statistics.mean(v):.0f} max={max(v)}")

# Concurrency simulation: conv arrives at timestamp_ms; each turn = request whose service
# time we model as S ms; next turn arrives think_ms after previous completes.
def sim_concurrency(part, service_ms):
    convs = defaultdict(list)
    for r in part:
        convs[r["conv_id"]].append(r)
    events = []  # (time, +1/-1)
    for cid, rs in convs.items():
        rs.sort(key=lambda r: r["turn_idx"])
        t = rs[0]["timestamp_ms"]
        for r in rs:
            events.append((t, +1))
            t += service_ms
            events.append((t, -1))
            t += r["think_ms"]
    events.sort()
    cur = peak = 0
    area = 0.0
    last = events[0][0]
    for tt, d in events:
        area += cur * (tt - last)
        last = tt
        cur += d
        peak = max(peak, cur)
    dur = events[-1][0] - events[0][0]
    return peak, area / dur if dur else 0.0, dur / 1000.0

print("\nconcurrency sim on CURRENT ALL-420 workload:")
for s in (500, 1000, 2000, 3000, 5000):
    peak, avg, dur = sim_concurrency(rows, s)
    print(f"  service {s:>4}ms: peak={peak:>3} avg={avg:5.1f} makespan={dur:7.1f}s")

# Same but everything (warmup runs first? check overlap of arrival windows)
wa = [r["timestamp_ms"] for r in marked_warm if r["turn_idx"] == 0]
sa = [r["timestamp_ms"] for r in marked_regular if r["turn_idx"] == 0]
print(
    f"\nstale-marker arrival window: {min(wa)}..{max(wa)} ms; "
    f"remaining arrival window: {min(sa)}..{max(sa)} ms"
)
