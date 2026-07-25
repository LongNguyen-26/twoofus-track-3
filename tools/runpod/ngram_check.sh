#!/usr/bin/env bash
# Correctness gate for CPU ngram speculative decoding, the last untested lever.
#
# Why it is still open. The 17/07 battery rejected CPU ngram on a local replay
# whose prompts are synthetic random tokens -- prompt lookup can never accept
# anything there, so the measurement was pure overhead and says nothing about
# the real trace, which is multi-turn QA over ~4k tokens of context. If real
# acceptance runs ~1.4 tokens/step, TPOT 4.07 -> 2.9ms and ERS reaches ~71.
#
# Why it needs this gate first. submit_015 shipped ngram_gpu and the harness
# aborted with "long-context probe failed (0%) -- truncation / dual-path
# likely". Plain ngram is a different, older proposer, but a second such flag
# would look bad at hau kiem, so prove long context survives before spending a
# slot. Greedy speculative decoding is mathematically lossless, so any output
# difference is a bug and a hard stop.
#
# This deliberately does NOT run replay_r2.py: the synthetic corpus cannot
# measure ngram acceptance, and reporting a number from it would repeat the
# mistake that closed this track in the first place.
set -uo pipefail
unset VLLM_API_KEY
MODEL=/workspace/model
PORT=8001
URL="http://localhost:${PORT}"
CORES=${CORES:-0-2}
VRAM_MB=${VRAM_MB:-17100}
OUT="/workspace/repo/results_ngram_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
SERVER_PID=""
cd /workspace/repo

TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_FRAC=$(python3 -c "print(min(0.95, round(${VRAM_MB}/${TOTAL_MB}, 3)))")

BASE=(
  --model=$MODEL --served-model-name=LFM2.5-1.2B-Instruct
  --host=0.0.0.0 --port=$PORT
  --max-model-len=32768 --gpu-memory-utilization=$GPU_FRAC
  --tensor-parallel-size=1 --enable-prefix-caching --quantization=fp8
)
NGRAM='{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":4,"prompt_lookup_min":2}'

start() {
  local log=$1; shift
  # shellcheck disable=SC2086
  setsid taskset -c "$CORES" python3 -m vllm.entrypoints.openai.api_server \
    "${BASE[@]}" "$@" >"$log" 2>&1 &
  SERVER_PID=$!
  local s; s=$(date +%s)
  while (( $(date +%s) - s < 600 )); do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  !! died"; tail -15 "$log"; return 1; }
    curl -sf "${URL}/health" >/dev/null 2>&1 && { echo "  up after $(( $(date +%s) - s ))s"; return 0; }
    sleep 2
  done
  echo "  !! timeout"; return 1
}
stop() { [[ -n "$SERVER_PID" ]] && kill -- -"$SERVER_PID" 2>/dev/null; SERVER_PID=""; sleep 5; }
trap 'stop; exit 130' INT TERM

echo "=== 1/2 baseline (no spec): write greedy reference + needle ==="
start "$OUT/base_server.log" || exit 1
python3 tools/evaluation/greedy_equivalence.py --url "$URL" \
  --write-reference "$OUT/reference.json" | tee "$OUT/base_equiv.log"
python3 tools/evaluation/needle_test.py "$URL" | tee "$OUT/base_needle.log"
stop

echo
echo "=== 2/2 CPU ngram: equivalence vs reference + needle ==="
start "$OUT/ngram_server.log" --speculative-config "$NGRAM" || {
  echo "!! ngram failed to start -- track closed, do not spend a slot"; exit 1; }
grep -i "speculat\|ngram\|proposer" "$OUT/ngram_server.log" | tail -5 | sed 's/^/  /'
python3 tools/evaluation/greedy_equivalence.py --url "$URL" \
  --reference "$OUT/reference.json" | tee "$OUT/ngram_equiv.log"
python3 tools/evaluation/needle_test.py "$URL" | tee "$OUT/ngram_needle.log"
grep -i "acceptance\|accepted\|draft" "$OUT/ngram_server.log" | tail -8 | sed 's/^/  /'
stop

cat <<EOF

==================== VERDICT ====================
$(grep -H "RETRIEVAL" "$OUT"/*_needle.log | sed "s|$OUT/||")
$(grep -Hi "match\|identical\|differ" "$OUT"/*_equiv.log | sed "s|$OUT/||")

SPEND A PORTAL SLOT only if BOTH hold:
  * ngram needle retrieval is no worse than the baseline in the same run
  * greedy equivalence is exact -- lossless is a theorem here, so any
    mismatch is a vLLM bug and closes the track for good

Acceptance rate cannot be measured locally (synthetic prompts have nothing to
copy). That is precisely what the portal slot buys: the implied-TPOT statistic
recovered from ERS and TTFT p50 will show whether real acceptance exists.
Full logs in $OUT/
EOF
