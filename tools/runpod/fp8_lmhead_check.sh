#!/usr/bin/env bash
# F-series: does the FP8 lm_head patch (submit_033/034) actually pay?
#
# The patch quantizes the one tensor `--quantization=fp8` skips: LFM2.5's
# 65536x2048 tied embedding / output head, which stays BF16 because
# ParallelLMHead subclasses VocabParallelEmbedding rather than LinearBase.
# Byte accounting says 0.268 GB -> 0.134 GB per decode step, about -0.22 ms of
# the portal's 4.07 ms TPOT.
#
# Correctness is already gated inside the patch by a startup self-test, so what
# this script is for is the question the self-test cannot answer: whether
# `torch._scaled_mm` with row-wise scales is FASTER than a BF16 GEMV at batch 1
# on a slice-sized SM budget. That is a pure kernel question, which is exactly
# what the calibrated Hopper rig is good for.
#
# NOTE ON THE GATE. Do not reuse the greedy-equivalence gate here. That gate
# exists for speculative decoding, which is lossless by construction, so any
# divergence proves a bug (that is how ngram was caught duplicating tokens).
# Quantization is *expected* to move some tokens. The gates that apply are:
#   * the patch reports ACTIVE, not REJECTED, in the server log
#   * needle retrieval no worse than the F0 baseline (never 0/5 -- the harness
#     aborts a submission whose long-context probe returns 0%, see submit_015)
#   * output stays coherent: no duplicated tokens, no truncation
#   * TPOT drops in both prompt regimes
#
# Prereqs: bash tools/runpod/pod_setup_h100.sh   (model + MPS cap + deps)
# Usage:   bash tools/runpod/fp8_lmhead_check.sh
#          ONLY="F1" bash tools/runpod/fp8_lmhead_check.sh
set -uo pipefail
unset VLLM_API_KEY

MODEL=/workspace/model
PORT=8001
URL="http://localhost:${PORT}"
CORES=${CORES:-0-2}
VRAM_MB=${VRAM_MB:-17100}
SM_PCT=${SM_PCT:-12}
HOG_PCT=${HOG_PCT:-60}
HOG_GB=${HOG_GB:-4}
ONLY=${ONLY:-"F0 F1 F2"}
OUT="/workspace/repo/results_fp8head_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
cd /workspace/repo

TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_FRAC=$(python3 -c "print(min(0.95, round(${VRAM_MB}/${TOTAL_MB}, 3)))")

export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
if pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1; then
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SM_PCT
  SM_STATE="CAPPED at ${SM_PCT}%"
else
  SM_STATE="!! UNCAPPED -- an SM-starved kernel will look better here than on the slice"
fi
echo "SM cap: $SM_STATE" | tee "$OUT/rig.txt"

# Install the exact image payload into this pod's site-packages, which is the
# same interpreter and the same layout the published image patches.
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
install_payload() {  # $1 = 1 to enable the patch by default, 0 to ship it inert
  cp docker/lfm25-custom-serve/payload/lfm25_boot.py "$SITE/"
  cp docker/lfm25-custom-serve/payload/lfm25_patches.py "$SITE/"
  printf 'FP8_LMHEAD = %s\n' "$([ "$1" = 1 ] && echo True || echo False)" \
    > "$SITE/lfm25_defaults.py"
  printf 'import lfm25_boot\n' > "$SITE/lfm25_serve.pth"
  python3 -c 'import sys' 2>&1 | grep -q 'autopatch armed' \
    || { echo "FATAL: .pth did not run -- check $SITE"; exit 1; }
}
remove_payload() {
  rm -f "$SITE"/lfm25_boot.py "$SITE"/lfm25_patches.py \
        "$SITE"/lfm25_defaults.py "$SITE"/lfm25_serve.pth
}
trap 'remove_payload' EXIT

HOG_PID=""
start_hog() {
  [ "$HOG_PCT" = "0" ] && return
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$HOG_PCT \
    python3 tools/runpod/bw_hog.py --gb "$HOG_GB" >"$OUT/hog.log" 2>&1 &
  HOG_PID=$!
  sleep 5
}
stop_hog() { [ -n "$HOG_PID" ] && kill "$HOG_PID" 2>/dev/null; HOG_PID=""; }

SERVER_PID=""
start_server() {  # $1 = run tag
  local log="$OUT/$1.server.log"
  taskset -c "$CORES" python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name LFM2.5-1.2B-Instruct \
    --host 0.0.0.0 --port "$PORT" \
    --max-model-len 32768 --gpu-memory-utilization "$GPU_FRAC" \
    --tensor-parallel-size 1 --enable-prefix-caching \
    --quantization fp8 >"$log" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 120); do
    curl -sf "$URL/v1/models" >/dev/null 2>&1 && break
    sleep 5
  done
  # NOT anchored with ^: vLLM prefixes subprocess stderr with
  # "(EngineCore pid=NNNN) " and the engine process is the one that patches.
  grep -F '[lfm25]' "$log" | sort -u | tee "$OUT/$1.patch.log"
}
stop_server() {
  [ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null
  wait "$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  sleep 10
}

run_case() {  # $1 = tag, $2 = enable patch
  echo "=== $1 (patch=$2) ===" | tee -a "$OUT/summary.txt"
  install_payload "$2"
  start_hog
  start_server "$1"
  if ! curl -sf "$URL/v1/models" >/dev/null 2>&1; then
    echo "$1: SERVER DID NOT START -- see $OUT/$1.server.log" | tee -a "$OUT/summary.txt"
    stop_server; stop_hog; return
  fi
  if [ "$2" = 1 ] && ! grep -q 'fp8 lm_head: ACTIVE' "$OUT/$1.patch.log"; then
    echo "$1: PATCH NOT ACTIVE -- treat every number below as the stock baseline" \
      | tee -a "$OUT/summary.txt"
  fi
  for mode in fresh shared; do
    echo "--- replay mode=$mode ---"
    python3 tools/replay/replay_r2.py --url "$URL" --mode "$mode" \
      --tokenizer "$MODEL" | tee "$OUT/${1}_${mode}.log"
  done
  echo "--- needle test ($1) ---"
  python3 tools/evaluation/needle_test.py "$URL" | tee "$OUT/${1}_needle.log"
  # Coherence sample: read this by eye. ngram's corruption showed up here as
  # duplicated tokens while the substring-scored needle test still passed.
  curl -s "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
    "model":"LFM2.5-1.2B-Instruct","temperature":0,"max_tokens":160,
    "messages":[{"role":"user","content":"List the first 8 prime numbers, then write two sentences about why 2 is unusual among them."}]}' \
    | python3 -c "import json,sys; print('  sample:', json.load(sys.stdin)['choices'][0]['message']['content'][:400].replace(chr(10),' | '))" \
    | tee -a "$OUT/summary.txt"
  stop_server
  stop_hog
}

for tag in $ONLY; do
  case "$tag" in
    F0) run_case F0 0 ;;   # baseline: layer present but inert
    F1) run_case F1 1 ;;   # treatment
    F2) run_case F2 0 ;;   # drift control, closes the battery
  esac
done

echo
echo "==================== F SUMMARY ===================="
echo "SM cap: $SM_STATE"
grep -H "SCORED" "$OUT"/*_fresh.log "$OUT"/*_shared.log 2>/dev/null | sed "s|$OUT/||"
echo "--- tpot ---"
grep -H "tpot_ms:" "$OUT"/*_fresh.log "$OUT"/*_shared.log 2>/dev/null | sed "s|$OUT/||"
echo "--- needle ---"
grep -Hc "HIT" "$OUT"/*_needle.log 2>/dev/null | sed "s|$OUT/||"
echo "--- patch state ---"
grep -H "fp8 lm_head" "$OUT"/*.patch.log 2>/dev/null | sed "s|$OUT/||"
echo
echo "Ship the patch only if F1 beats the mean of F0/F2 on tpot in BOTH modes,"
echo "needle is no worse than F0, and the coherence samples read cleanly."
