"""Needle-in-haystack long-context integrity check (~28k tokens).

Run from the repo root with:
python tools/evaluation/needle_test.py http://localhost:8001
"""
import json
import random
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001"
WORDS = ("time year people way day man thing woman life child world school state "
         "family student group country problem hand part place case week company "
         "system program question work night point home water room mother area "
         "money story fact month lot right study book eye job word business").split()
FACTS = [("alpha", "739214"), ("bravo", "apricot42"), ("charlie", "M4X7Q2"),
         ("delta", "seventeen"), ("echo", "turquoise9")]

rng = random.Random(7)
ok = 0
for name, val in FACTS:
    filler = " ".join(rng.choice(WORDS) for _ in range(21000))
    ctx = (f"IMPORTANT FACT: the secret code for project {name} is {val}.\n\n"
           + filler)
    q = (f"\n\nQuestion: What is the secret code for project {name} stated at "
         f"the very beginning? Answer with just the code.")
    body = {"model": "LFM2.5-1.2B-Instruct",
            "messages": [{"role": "user", "content": ctx + q}],
            "max_tokens": 30, "temperature": 0}
    req = urllib.request.Request(URL + "/v1/chat/completions",
                                 json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    try:
        resp = json.load(urllib.request.urlopen(req, timeout=300))
        ans = resp["choices"][0]["message"]["content"]
    except Exception as e:
        ans = f"ERROR {e}"
    hit = val.lower() in ans.lower()
    ok += hit
    print(f"needle {name}: {'HIT' if hit else 'MISS'} -> {ans[:80]!r}")
print(f"RETRIEVAL {ok}/{len(FACTS)}")
