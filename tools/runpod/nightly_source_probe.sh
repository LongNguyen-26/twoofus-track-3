#!/usr/bin/env bash
# Ask vLLM main whether the doors that closed on v0.25.1 have reopened.
#
# Every speculative-decoding route died on v0.25.1 with a BUG-shaped failure,
# not a physics limit:
#   * draft model / TLI  -> AssertionError "All drafting layers should belong to
#                           the same kv cache group" (hybrid TARGET unsupported)
#   * ngram_gpu          -> corrupted long-context output, failed the harness probe
#   * suffix             -> arctic-inference pinned torch==2.7.0, then 0/6 equivalence
# Those are exactly the things that get fixed release over release, and spec
# decode is the ONLY lever with a ceiling above ~72 (1.8 tokens/step => TPOT
# 2.26ms => ERS ~78). v0.25.1 is still the newest tagged release, but pinned
# nightly images exist, e.g.
#   vllm/vllm-openai:nightly-dd72658e7db0c6a674f473f6f9f0f5c2ebe7e523 (24/07)
#
# This probe is source-only: no install, no GPU, ~1 minute. Run it BEFORE
# spending pod time or a portal slot on a nightly image.
set -uo pipefail
SRC=${SRC:-/workspace/vllm-src}

if [[ ! -d "$SRC/.git" ]]; then
  echo "cloning vllm main (shallow) into $SRC ..."
  git clone --depth 1 https://github.com/vllm-project/vllm.git "$SRC" || exit 1
else
  git -C "$SRC" fetch --depth 1 origin main && git -C "$SRC" reset --hard origin/main
fi

echo
echo "commit: $(git -C "$SRC" rev-parse --short HEAD)  $(git -C "$SRC" log -1 --format=%cd)"
echo

echo "=== 1. the kv-cache-group assertion that blocked draft-model spec decode ==="
if grep -rn "same kv cache group" "$SRC/vllm" 2>/dev/null; then
  echo ">> STILL PRESENT. Inspect the guard: if it now allows a dense draft"
  echo ">> against a hybrid target, the draft/EAGLE track reopens. If it is an"
  echo ">> unconditional assert, draft-model spec decode stays dead."
else
  echo ">> GONE from the source tree. This is the single highest-value signal"
  echo ">> in the whole round: retest a dense draft against LFM2.5 immediately."
fi

echo
echo "=== 2. hybrid / mamba awareness in the spec-decode path ==="
grep -rln "is_hybrid\|mamba\|ShortConv" "$SRC/vllm/v1/spec_decode/" 2>/dev/null \
  || echo "(no hybrid-aware files under vllm/v1/spec_decode/)"

echo
echo "=== 3. LFM2 support surface ==="
ls "$SRC/vllm/model_executor/models/" 2>/dev/null | grep -i lfm || echo "(no lfm2 model file?)"

echo
echo "=== 4. spec-decode methods this build accepts ==="
grep -rn "SpeculativeMethod\|\"ngram\"\|'ngram'" "$SRC/vllm/config/speculative.py" 2>/dev/null \
  | head -20

echo
echo "=== 5. recent spec-decode / hybrid commits ==="
git -C "$SRC" log --oneline -25 -- vllm/v1/spec_decode vllm/config/speculative.py 2>/dev/null

cat <<'EOF'

NEXT STEP IF SECTION 1 SAYS THE DOOR IS OPEN
  Redeploy the pod on the pinned nightly image and rerun the draft battery:
    ONLY="D0 D10" bash tools/runpod/draft_battery_r2.sh
  D10 is the dense SmolLM2-135M draft that previously hit the assertion.
  Gates before any portal slot: 6/6 greedy equivalence, needle no worse than
  the matched FP8 control, and a repeatable TPOT win under the SM cap.
EOF
