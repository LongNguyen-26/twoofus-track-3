#!/usr/bin/env bash
# GPQA-diamond Δ measurement for the final-5 audit (round 2).
# The portal no longer reports accuracy_drop; BTC runs GPQA full post-online on
# each team's <=5 picked submissions. We measure Δ(bf16 vs fp8) ourselves.
# Absolute accuracy need not match BTC's harness — Δ between configs on the
# same method is the signal (free band: Δ <= 0.10).
#
# Prereq ONCE (user, typed in the pod web terminal so the token never touches
# chat or git; the HF account must have accepted the Idavidrein/gpqa gate):
#   printf '%s' 'hf_YOURTOKEN' > /workspace/.hf_token && chmod 600 /workspace/.hf_token
#
# Usage: bash tools/gpqa_r2.sh                    # G0 (fp8) + G1 (bf16), zeroshot MCQ
#        ONLY=G0 TASKS=gpqa_diamond_cot_zeroshot bash tools/gpqa_r2.sh
set -uo pipefail
unset VLLM_API_KEY
HF_TOKEN=$(cat /workspace/.hf_token)
export HF_TOKEN
export HF_HUB_DISABLE_XET=1
MODEL=/workspace/model
PORT=8001
URL="http://localhost:${PORT}"
CORES=${CORES:-0-2}
VRAM_MB=${VRAM_MB:-17100}
TASKS=${TASKS:-gpqa_diamond_zeroshot}
OUT="/workspace/repo/results_gpqa_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
SERVER_PID=""
cd /workspace/repo

python3 -c "import lm_eval" 2>/dev/null || pip install -q "lm_eval[api]" 2>&1 | tail -1

TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_FRAC=$(python3 -c "print(min(0.95, round(${VRAM_MB}/${TOTAL_MB}, 3)))")
echo "GPU total ${TOTAL_MB} MiB -> gpu-memory-utilization ${GPU_FRAC}"

BASE_FLAGS=(
  --model=$MODEL
  --served-model-name=LFM2.5-1.2B-Instruct
  --host=0.0.0.0
  --port=$PORT
  --max-model-len=32768
  --gpu-memory-utilization=$GPU_FRAC
  --tensor-parallel-size=1
  --enable-prefix-caching
)

declare -A CFG
CFG[G0]="--quantization=fp8"   # = submit_013/020 config
CFG[G1]=""                     # = submit_018 bf16 twin
ORDER=${ONLY:-"G0 G1"}

wait_health() {
  local start
  start=$(date +%s)
  echo "  waiting for /health (pid ${SERVER_PID}) ..."
  while (( $(date +%s) - start < 600 )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "  !! server died during startup"
      return 1
    fi
    if curl -sf "${URL}/health" >/dev/null 2>&1; then
      echo "  up after $(( $(date +%s) - start ))s"
      return 0
    fi
    sleep 2
  done
  echo "  !! not healthy in 600s"
  return 1
}

stop_server() {
  [[ -n "$SERVER_PID" ]] && kill -- -"$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  sleep 5
}

for id in $ORDER; do
  extra=${CFG[$id]}
  echo
  echo "############ GPQA $id : ${extra:-<bf16>} / tasks=$TASKS ############"
  # shellcheck disable=SC2086
  setsid taskset -c "$CORES" python3 -m vllm.entrypoints.openai.api_server \
    "${BASE_FLAGS[@]}" $extra > "$OUT/${id}_server.log" 2>&1 &
  SERVER_PID=$!
  if ! wait_health; then
    tail -5 "$OUT/${id}_server.log"
    stop_server
    continue
  fi
  lm_eval --model local-completions \
    --model_args "model=LFM2.5-1.2B-Instruct,base_url=${URL}/v1/completions,num_concurrent=8,max_retries=3,tokenized_requests=False" \
    --tasks "$TASKS" \
    --output_path "$OUT/${id}" 2>&1 | tee "$OUT/${id}_eval.log" | tail -12
  stop_server
done

echo
echo "==== GPQA SUMMARY (${TASKS}) ===="
grep -h -E "gpqa|acc" "$OUT"/*_eval.log | tail -12
echo "Full logs in $OUT/"
