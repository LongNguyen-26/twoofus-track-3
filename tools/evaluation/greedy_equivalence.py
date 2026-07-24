"""Check that a speculative-decoding server preserves greedy model outputs.

The first run writes a deterministic reference captured from the clean FP8
baseline. Later runs compare the exact response text against that reference.
This is a small correctness gate; it complements (but does not replace) the
long-context needle test.

Examples:
  python tools/evaluation/greedy_equivalence.py --url http://localhost:8001 \
      --write-reference results/equivalence_reference.json
  python tools/evaluation/greedy_equivalence.py --url http://localhost:8001 \
      --reference results/equivalence_reference.json
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path


PROMPTS = [
    (
        "Continue the following deterministic pattern for twelve more items. "
        "Return only the comma-separated continuation.\n"
        "red, blue, green, red, blue, green, red, blue, green,"
    ),
    (
        "Complete this Python function. Return only the function body.\n"
        "def triangular(n):\n"
        "    \"\"\"Return 1 + 2 + ... + n for a non-negative integer.\"\"\"\n"
    ),
    (
        "A warehouse repeats the sequence A17, B04, C29 every three rows. "
        "Rows 1 through 18 follow that rule. State the values in rows 13, 14, "
        "15, 16, 17, and 18, in order. Return only the six values."
    ),
    (
        "Summarize the rule in one sentence: Every request must be served by "
        "vLLM, must use the mounted model, must not access the external "
        "network, and must preserve greedy decoding semantics."
    ),
    (
        "The checksum fragments are ka, li, ka, li, ka, li. Continue the "
        "alternating sequence until it contains sixteen total fragments. "
        "Return only the ten new fragments separated by spaces."
    ),
    (
        "Answer with a JSON object using keys result and explanation. "
        "Compute 37 * 19, then explain the multiplication in at most twelve "
        "words."
    ),
]


def request_one(url, model_name, prompt):
    body = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 128,
        "temperature": 0,
        "seed": 42,
    }
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        payload = json.load(resp)
    return payload["choices"][0]["message"]["content"]


def capture(url, model_name):
    outputs = []
    for index, prompt in enumerate(PROMPTS):
        try:
            text = request_one(url, model_name, prompt)
        except Exception as exc:
            print(f"equivalence {index}: ERROR {type(exc).__name__}: {exc}")
            raise
        outputs.append(text)
        print(f"equivalence {index}: captured {len(text)} chars")
    return {
        "model": model_name,
        "temperature": 0,
        "seed": 42,
        "max_tokens": 128,
        "prompts": PROMPTS,
        "outputs": outputs,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8001")
    ap.add_argument("--model-name", default="LFM2.5-1.2B-Instruct")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write-reference")
    mode.add_argument("--reference")
    args = ap.parse_args()

    captured = capture(args.url, args.model_name)
    if args.write_reference:
        path = Path(args.write_reference)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(captured, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"EQUIVALENCE {len(PROMPTS)}/{len(PROMPTS)} "
            "(reference written)"
        )
        return

    reference = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    if reference.get("prompts") != PROMPTS:
        print("EQUIVALENCE 0/0: reference prompt set does not match this script")
        raise SystemExit(2)

    matches = 0
    for index, (expected, actual) in enumerate(
        zip(reference["outputs"], captured["outputs"], strict=True)
    ):
        same = expected == actual
        matches += int(same)
        print(f"compare {index}: {'MATCH' if same else 'MISMATCH'}")
        if not same:
            print(f"  expected: {expected[:160]!r}")
            print(f"  actual:   {actual[:160]!r}")
    print(f"EQUIVALENCE {matches}/{len(PROMPTS)}")
    if matches != len(PROMPTS):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
