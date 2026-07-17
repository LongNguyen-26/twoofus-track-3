#!/usr/bin/env bash
# ROUND-2 experiment battery (LFM2.5-1.2B-Instruct) for a RunPod pod running
# image vllm/vllm-openai:v0.25.1 (same pattern as round 1's pod_overnight.sh).
#
# HOST DRIVER: the vllm images are CUDA-13-based — on deploy use
# Filter -> CUDA Version >= 13.0, or the container dies at init.
#
# KEEPALIVE: the vllm image's entrypoint IS the api server, so give the pod a
# tiny model as its Container Start Command to keep it alive on port 8000:
#   --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 \
#       --gpu-memory-utilization 0.1 --max-model-len 2048
# This script runs the real configs on port 8001 in their own process groups.
#
# PREREQ (once, in the pod Web Terminal, after the keepalive is Ready):
#   cd /workspace
#   git clone https://github.com/LongNguyen-26/twoofus-track-3.git repo && cd repo
#   pip install -U huggingface_hub hf_transfer aiohttp
#   export HF_HUB_ENABLE_HF_TRANSFER=1
#   huggingface-cli download LiquidAI/LFM2.5-1.2B-Instruct --local-dir /workspace/model
#
# Then:  bash tools/pod_battery_r2.sh
# Env knobs:
#   MODES="shared fresh"   replay prompt regimes per config (default both)
#   ONLY="R1 R3"           run just these config IDs
#   CORES=0-2              taskset CPU set (3 cores = MiG sim)
#   VRAM_MB=17100          KV+weights budget (0.95 * 18GB slice)

set -uo pipefail
unset VLLM_API_KEY

MODEL=/workspace/model
PORT=8001
URL="http://localhost:${PORT}"
CORES=${CORES:-0-2}
MODES=${MODES:-"shared fresh"}
VRAM_MB=${VRAM_MB:-17100}
OUT="results_r2_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
SERVER_PID=""

# Scale --gpu-memory-utilization so total VRAM use ≈ the MiG slice's 0.95*18GB
TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_FRAC=$(python3 -c "print(min(0.95, round(${VRAM_MB}/${TOTAL_MB}, 3)))")
echo "GPU total ${TOTAL_MB} MiB -> --gpu-memory-utilization ${GPU_FRAC} (sim ${VRAM_MB} MiB)"

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

# Config battery. R0 = submit_013 (portal 58.68: ttft p50 66 / p95 88, tbt 4ms).
# Goal ranking: TPOT is the big remaining hole (s_tpot ~0.45 vs s_ttft ~0.73).
declare -A CFG
CFG[R0]="--quantization=fp8"
CFG[R1]="--quantization=fp8 --speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"
CFG[R2]="--quantization=fp8 --speculative-config {\"method\":\"suffix\"}"
CFG[R3]="--quantization=fp8 --compilation-config {\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"}"
CFG[R4]="--quantization=fp8 --async-scheduling"
CFG[R5]="--quantization=fp8 --max-num-batched-tokens=4096"
CFG[R6]=""  # BF16 reference (tie-break hedge + fp8 delta measurement)
CFG[R7]="--quantization=fp8 --async-scheduling --compilation-config {\"cudagraph_mode\":\"FULL_AND_PIECEWISE\"} --speculative-config {\"method\":\"ngram\",\"num_speculative_tokens\":4,\"prompt_lookup_max\":4,\"prompt_lookup_min\":2}"
ORDER=${ONLY:-"R0 R1 R2 R3 R4 R5 R6 R7"}

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
      echo "  up after $(( $(date +%s) - start ))s (BTC gate: 600s)"
      return 0
    fi
    sleep 2
  done
  echo "  !! not healthy in 600s — would FAIL the BTC gate"
  return 1
}

stop_server() {
  [[ -n "$SERVER_PID" ]] && kill -- -"$SERVER_PID" 2>/dev/null
  SERVER_PID=""
  sleep 5
}

for id in $ORDER; do
  extra=${CFG[$id]}
  log="$OUT/${id}_server.log"
  echo
  echo "############################################################"
  echo "# CONFIG $id : ${extra:-<bf16 baseline flags only>}"
  echo "############################################################"
  # shellcheck disable=SC2086
  setsid taskset -c "$CORES" python3 -m vllm.entrypoints.openai.api_server \
    "${BASE_FLAGS[@]}" $extra >"$log" 2>&1 &
  SERVER_PID=$!
  if ! wait_health; then
    echo "  config $id FAILED to start — last log lines:"
    tail -5 "$log"
    stop_server
    continue
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1
  for mode in $MODES; do
    echo "--- replay mode=$mode ---"
    python3 tools/replay_r2.py --url "$URL" --mode "$mode" --tokenizer "$MODEL" \
      | tee "$OUT/${id}_${mode}.log"
  done
  stop_server
done

echo
echo "==== SUMMARY (scored ERS lines) ===="
grep -H "SCORED" "$OUT"/*_*.log | sed "s|$OUT/||"
echo "Full logs in $OUT/"
