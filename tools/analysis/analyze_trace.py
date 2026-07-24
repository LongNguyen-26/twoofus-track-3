"""Tokenize the trace with the real Qwen3.5-2B tokenizer and print workload facts.

Needs: pip install transformers huggingface_hub  (downloads ~15MB of tokenizer files,
never the weights). Run from the repo root:
python tools/analysis/analyze_trace.py
"""
import json
import os

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

REPO = "Qwen/Qwen3.5-2B"
TRACE = os.path.join(
    os.path.dirname(__file__), "..", "..", "input", "trace-round1.jsonl"
)

tok = AutoTokenizer.from_pretrained(os.path.dirname(hf_hub_download(REPO, "tokenizer.json")))
chat_template = open(hf_hub_download(REPO, "chat_template.jinja"), encoding="utf-8").read()

rows = [json.loads(l) for l in open(TRACE, encoding="utf-8")]


def ids_of(i):
    return tok.apply_chat_template(
        rows[i]["body"]["messages"], chat_template=chat_template, add_generation_prompt=True
    )


counts = [len(ids_of(i)) for i in range(len(rows))]

waves = {}
for r, c in zip(rows, counts):
    waves.setdefault(r["timestamp_ms"] // 5000, []).append(c)

print("=== exact prompt tokens (with chat template) ===")
print("min", min(counts), "max", max(counts), "avg", sum(counts) // len(counts))
for w in sorted(waves):
    ws = waves[w]
    print(f"wave {w} (t={w*5}s): min={min(ws)} max={max(ws)} avg={sum(ws)//len(ws)}")

a, b = ids_of(0), ids_of(1)  # two different conversations -> share only the system prompt
sys_len = 0
while sys_len < min(len(a), len(b)) and a[sys_len] == b[sys_len]:
    sys_len += 1
print("shared token prefix across conversations (system prompt block):", sys_len)

print("turn deltas conv0:", [counts[i + 20] - counts[i] for i in range(0, 100, 20)])

naive_total = sum(counts)
final_wave = [counts[i] for i in range(100, 120)]
perfect = sum(final_wave) - 19 * sys_len  # system prompt counted once instead of 20x
print("naive total prefill tokens:", naive_total)
print("perfect-reuse unique tokens:", perfect, f"({naive_total/perfect:.1f}x reduction)")
