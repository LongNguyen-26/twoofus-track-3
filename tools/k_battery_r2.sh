#!/usr/bin/env bash
# K-series: KV-cache bandwidth battery. Decode on the slice is bandwidth-bound;
# at 4k context the KV read (~0.49GB bf16) is ~30% of per-step bandwidth and was
# left at bf16 in every prior round-2 config. fp8 KV halves it -> should lower
# TPOT IF the slice's fp8-dequant is cheap (round-1 Qwen hinted it wasn't; the
# LFM2.5 geometry + Hopper native fp8 may differ -> measure, don't assume).
# Each config runs fresh + shared replay + MANDATORY needle test (fp8 KV can
# corrupt long context -> the harness probe that killed 015).
set -uo pipefail
unset VLLM_API_KEY
MODEL=/workspace/model
PORT=8001
URL="http://localhost:${PORT}"
CORES=${CORES:-0-2}
MODES=${MODES:-"fresh shared"}
VRAM_MB=${VRAM_MB:-17100}
OUT="/workspace/repo/results_k_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
SERVER_PID=""
cd /workspace/repo

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
  --quantization=fp8
)

declare -A CFG
CFG[K0]=""                              # = submit_020 (fp8 weights, bf16 KV) reference
CFG[K1]="--kv-cache-dtype fp8"          # fp8 KV (e4m3 default)
CFG[K2]="--kv-cache-dtype fp8_e5m2"     # alt fp8 KV format
ORDER=${ONLY:-"K0 K1 K2"}

wait_health() {
  local start; start=$(date +%s)
  echo "  waiting for /health (pid ${SERVER_PID}) ..."
  while (( $(date +%s) - start < 600 )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "  !! server died"; return 1; fi
    if curl -sf "${URL}/health" >/dev/null 2>&1; then
      echo "  up after $(( $(date +%s) - start ))s"; return 0; fi
    sleep 2
  done
  echo "  !! not healthy in 600s"; return 1
}

stop_server() {
  [[ -n "$SERVER_PID" ]] && kill -- -"$SERVER_PID" 2>/dev/null
  SERVER_PID=""; sleep 5
}

for id in $ORDER; do
  extra=${CFG[$id]}
  log="$OUT/${id}_server.log"
  echo
  echo "############################################################"
  echo "# CONFIG $id : ${extra:-<K0: fp8 weights, bf16 KV = submit_020>}"
  echo "############################################################"
  # shellcheck disable=SC2086
  setsid taskset -c "$CORES" python3 -m vllm.entrypoints.openai.api_server \
    "${BASE_FLAGS[@]}" $extra >"$log" 2>&1 &
  SERVER_PID=$!
  if ! wait_health; then
    echo "  config $id FAILED to start - last log lines:"; tail -5 "$log"; stop_server; continue
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1
  for mode in $MODES; do
    echo "--- replay mode=$mode ---"
    python3 tools/replay_r2.py --url "$URL" --mode "$mode" --tokenizer "$MODEL" \
      | tee "$OUT/${id}_${mode}.log"
  done
  echo "--- needle test ($id) ---"
  python3 tools/needle_test.py "$URL" | tee "$OUT/${id}_needle.log"
  stop_server
done

echo
echo "==== K SUMMARY ===="
grep -H "SCORED" "$OUT"/*_*.log 2>/dev/null | sed "s|$OUT/||"
grep -H "tpot_ms:" "$OUT"/*_fresh.log 2>/dev/null | sed "s|$OUT/||"
grep -H "RETRIEVAL" "$OUT"/*_needle.log 2>/dev/null | sed "s|$OUT/||"
echo "Full logs in $OUT/"
