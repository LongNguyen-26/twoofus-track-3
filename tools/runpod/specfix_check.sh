#!/usr/bin/env bash
# Does repairing vLLM's optimistic spec-decode placeholders make ngram_gpu
# correct, and is it then worth a portal slot?
#
# THE DIAGNOSIS BEING TESTED
# --------------------------
# gpu_model_runner.py inserts `[-1] * prev_num_draft_len` placeholders into a
# request's output_token_ids whenever async scheduling is on and the last step
# drafted. gpu_input_batch.update_async_output_token_ids() repairs them only if
# `sampling_metadata.output_token_ids` is populated, which needs penalties /
# bad_words / custom logitsprocs / thinking budget. The graded workload is
# greedy with none of those, so the repair never runs and -1 stays in the token
# history that feeds token_ids_cpu and the n-gram corpus.
#
# Async scheduling defaults to ON and is auto-disabled for spec decode EXCEPT
# Eagle, dspark and NgramGPUTypes = Literal["ngram_gpu"]. So ngram_gpu keeps
# async and hits the bug; CPU ngram silently loses async and is a DIFFERENT
# defect that this patch does not fix. Do not test CPU ngram here.
#
# THREE ARMS, and B1 is the one that makes this science rather than hope:
#   B0  no spec decode          -> greedy reference + baseline TPOT
#   B1  ngram_gpu, patch OFF    -> MUST FAIL equivalence. If it passes, the
#                                  diagnosis is wrong and B2 proves nothing.
#   B2  ngram_gpu, patch ON     -> MUST be 6/6. Anything less and we stop.
#
# Speculative decoding is lossless by construction, so 6/6 is not a preference,
# it is the definition of working. submit_015 was aborted online with
# "truncation / dual-path likely" for shipping this untested; a second such flag
# would be read badly at hậu kiểm.
#
# Prereqs: bash tools/runpod/pod_setup_h100.sh
# Usage:   bash tools/runpod/specfix_check.sh
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
NUM_SPEC=${NUM_SPEC:-3}
OUT="/workspace/repo/results_specfix_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
cd /workspace/repo

TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_FRAC=$(python3 -c "print(min(0.95, round(${VRAM_MB}/${TOTAL_MB}, 3)))")

export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
if pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1; then
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SM_PCT
  SM_STATE="CAPPED at ${SM_PCT}%"
else
  SM_STATE="!! UNCAPPED"
fi

BASE=(
  --model=$MODEL --served-model-name=LFM2.5-1.2B-Instruct
  --host=0.0.0.0 --port=$PORT
  --max-model-len=32768 --gpu-memory-utilization=$GPU_FRAC
  --tensor-parallel-size=1 --enable-prefix-caching --quantization=fp8
)
NGRAM_GPU="{\"method\":\"ngram_gpu\",\"num_speculative_tokens\":${NUM_SPEC},\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"

# Install the image payload into this pod's interpreter; arms toggle it by env.
SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp docker/lfm25-custom-serve/payload/lfm25_boot.py "$SITE/"
cp docker/lfm25-custom-serve/payload/lfm25_patches.py "$SITE/"
printf 'FP8_LMHEAD = False\nFIX_ASYNC_SPEC = False\n' > "$SITE/lfm25_defaults.py"
printf 'import lfm25_boot\n' > "$SITE/lfm25_serve.pth"
python3 -c 'import sys' 2>&1 | grep -q 'autopatch armed' \
  || { echo "FATAL: .pth did not run -- check $SITE"; exit 1; }
trap 'rm -f "$SITE"/lfm25_boot.py "$SITE"/lfm25_patches.py \
      "$SITE"/lfm25_defaults.py "$SITE"/lfm25_serve.pth' EXIT

HOG_PID=""
start_hog() {
  [ "$HOG_PCT" = "0" ] && return
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$HOG_PCT \
    python3 tools/runpod/bw_hog.py --gb "$HOG_GB" >"$OUT/hog.log" 2>&1 &
  HOG_PID=$!; sleep 5
}
stop_hog() { [ -n "$HOG_PID" ] && kill "$HOG_PID" 2>/dev/null; HOG_PID=""; }

SERVER_PID=""
start() {
  local log=$1; shift
  setsid taskset -c "$CORES" python3 -m vllm.entrypoints.openai.api_server \
    "${BASE[@]}" "$@" >"$log" 2>&1 &
  SERVER_PID=$!
  local s; s=$(date +%s)
  while (( $(date +%s) - s < 600 )); do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "  !! died"; tail -20 "$log"; return 1; }
    curl -sf "${URL}/health" >/dev/null 2>&1 && { echo "  up after $(( $(date +%s) - s ))s"; return 0; }
    sleep 2
  done
  echo "  !! timeout"; return 1
}
stop() { [[ -n "$SERVER_PID" ]] && kill -- -"$SERVER_PID" 2>/dev/null; SERVER_PID=""; sleep 8; }
trap 'stop; stop_hog; exit 130' INT TERM

report_env() {  # $1 = log
  grep -i "Asynchronous scheduling is" "$1" | tail -1 | sed 's/^/  /'
  # NOT anchored with ^: vLLM prefixes subprocess stderr with
  # "(EngineCore pid=NNNN) ", and the engine process is the one that installs
  # the patch. An anchored grep silently hides exactly the line that says
  # whether the patch armed -- which is what happened on the 26/07 run.
  grep -F '[lfm25]' "$1" | sort -u | sed 's/^/  /'
}

start_hog
echo "=== B0/3  baseline, no spec decode ==="
export LFM25_FIX_ASYNC_SPEC=0
start "$OUT/b0_server.log" || exit 1
report_env "$OUT/b0_server.log"
python3 tools/evaluation/greedy_equivalence.py --url "$URL" \
  --write-reference "$OUT/reference.json" | tee "$OUT/b0_equiv.log"
python3 tools/replay/replay_r2.py --url "$URL" --mode fresh --tokenizer "$MODEL" \
  | tee "$OUT/b0_fresh.log"
python3 tools/evaluation/needle_test.py "$URL" | tee "$OUT/b0_needle.log"
stop

echo
echo "=== B1/3  ngram_gpu WITHOUT the fix -- expected to FAIL ==="
export LFM25_FIX_ASYNC_SPEC=0
if start "$OUT/b1_server.log" --speculative-config "$NGRAM_GPU"; then
  report_env "$OUT/b1_server.log"
  python3 tools/evaluation/greedy_equivalence.py --url "$URL" \
    --reference "$OUT/reference.json" | tee "$OUT/b1_equiv.log"
  python3 tools/evaluation/needle_test.py "$URL" | tee "$OUT/b1_needle.log"
  stop
else
  echo "  ngram_gpu did not start unpatched; note it and continue" | tee "$OUT/b1_equiv.log"
fi

echo
echo "=== B2/3  ngram_gpu WITH the fix -- must be 6/6 ==="
export LFM25_FIX_ASYNC_SPEC=1
start "$OUT/b2_server.log" --speculative-config "$NGRAM_GPU" || {
  echo "!! patched ngram_gpu failed to start"; stop_hog; exit 1; }
report_env "$OUT/b2_server.log"
python3 tools/evaluation/greedy_equivalence.py --url "$URL" \
  --reference "$OUT/reference.json" | tee "$OUT/b2_equiv.log"
python3 tools/replay/replay_r2.py --url "$URL" --mode fresh --tokenizer "$MODEL" \
  | tee "$OUT/b2_fresh.log"
python3 tools/evaluation/needle_test.py "$URL" | tee "$OUT/b2_needle.log"
grep -iE "acceptance|accepted|draft.*rate" "$OUT/b2_server.log" | tail -8 | sed 's/^/  /'
stop
stop_hog

cat <<EOF

==================== VERDICT ====================
SM cap: $SM_STATE   num_speculative_tokens: $NUM_SPEC

--- async scheduling actually in effect ---
$(grep -ih "Asynchronous scheduling is" "$OUT"/b*_server.log | sed 's/^/  /')

--- equivalence (B1 must FAIL, B2 must be 6/6) ---
$(grep -iH "EQUIVALEN\|match\|mismatch" "$OUT"/b1_equiv.log "$OUT"/b2_equiv.log 2>/dev/null | sed "s|$OUT/||")

--- tpot / ERS, B0 vs B2 ---
$(grep -H "SCORED\|tpot_ms:" "$OUT"/b0_fresh.log "$OUT"/b2_fresh.log 2>/dev/null | sed "s|$OUT/||")

--- needle ---
$(grep -Hc "HIT" "$OUT"/b0_needle.log "$OUT"/b1_needle.log "$OUT"/b2_needle.log 2>/dev/null | sed "s|$OUT/||")

Read it like this:
  B1 fails, B2 6/6, B2 tpot < B0 tpot  -> ship it, this is the real lever
  B1 fails, B2 6/6, B2 tpot >= B0 tpot -> diagnosis right, acceptance too low
                                          on this corpus; do NOT spend a slot,
                                          and note the local corpus is
                                          synthetic so the portal may differ
  B1 PASSES                            -> diagnosis wrong, B2 proves nothing,
                                          stop and re-read the source
  B2 not 6/6                           -> a second bug remains; never submit
Full logs: $OUT
EOF
