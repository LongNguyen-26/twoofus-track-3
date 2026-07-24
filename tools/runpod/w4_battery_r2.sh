#!/usr/bin/env bash
# Round-2 in-flight W4 battery for LFM2.5-1.2B-Instruct on RunPod.
#
# Default order brackets two independently started BitsAndBytes W4 runs with
# byte-identical FP8 baselines:
#   FP8A -> BNB4A -> BNB4B -> FP8B
#
# Cross-quantization output equality is informational only: legitimate weight
# quantization can change greedy outputs. The correctness gates are instead:
# repeatability across BNB4A/BNB4B, long-context retrieval relative to FP8,
# zero replay errors, and a later GPQA delta check.
#
# Prerequisite:
#   bash tools/runpod/pod_install_bitsandbytes.sh
#
# Usage:
#   bash tools/runpod/w4_battery_r2.sh
#   ONLY="FP8A BNB4A" bash tools/runpod/w4_battery_r2.sh
#   OUT=/workspace/repo/results_w4_existing ONLY="BNB4B FP8B" \
#     bash tools/runpod/w4_battery_r2.sh

set -uo pipefail
unset VLLM_API_KEY
# A suffix experiment may have installed Arctic into this live pod. Keep its
# legacy patch plugin disabled so every run uses unmodified vLLM execution.
export ARCTIC_INFERENCE_ENABLED=0

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"

MODEL=${MODEL:-/workspace/model}
PORT=${PORT:-8001}
URL="http://localhost:${PORT}"
CORES=${CORES:-0-2}
MODES=${MODES:-fresh}
VRAM_MB=${VRAM_MB:-17100}
OUT=${OUT:-"$REPO_DIR/results_w4_$(date +%Y%m%d_%H%M%S)"}
FP8_REFERENCE="$OUT/fp8_equivalence_reference.json"
BNB4_REFERENCE="$OUT/bnb4_equivalence_reference.json"
SERVER_PID=""
mkdir -p "$OUT"

declare -A CFG
CFG[FP8A]='--quantization=fp8'
CFG[BNB4A]='--quantization=bitsandbytes --load-format=bitsandbytes'
CFG[BNB4B]='--quantization=bitsandbytes --load-format=bitsandbytes'
CFG[FP8B]='--quantization=fp8'
ORDER=${ONLY:-"FP8A BNB4A BNB4B FP8B"}

for id in $ORDER; do
  if [[ -z "${CFG[$id]+defined}" ]]; then
    echo "ERROR: unknown config ID: $id" >&2
    exit 2
  fi
done

if [[ " $ORDER " == *" BNB4"* ]] \
  && ! python3 -c "import bitsandbytes" >/dev/null 2>&1; then
  echo "ERROR: BitsAndBytes is not importable." >&2
  echo "Run: bash tools/runpod/pod_install_bitsandbytes.sh" >&2
  exit 2
fi

{
  date -u +"utc=%Y-%m-%dT%H:%M:%SZ"
  python3 -c \
    "import torch, vllm; print('vllm='+vllm.__version__); print('torch='+torch.__version__); print('container_cuda='+str(torch.version.cuda))"
  if python3 -c "import bitsandbytes" >/dev/null 2>&1; then
    python3 -c \
      "import bitsandbytes as bnb; print('bitsandbytes='+bnb.__version__)"
  fi
  nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader
} | tee "$OUT/environment.txt"

TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_FRAC=$(python3 -c "print(min(0.95, round(${VRAM_MB}/${TOTAL_MB}, 3)))")
echo "GPU total ${TOTAL_MB} MiB -> gpu-memory-utilization ${GPU_FRAC}"
echo "Results: $OUT"

BASE_FLAGS=(
  --model="$MODEL"
  --served-model-name=LFM2.5-1.2B-Instruct
  --host=0.0.0.0
  --port="$PORT"
  --max-model-len=32768
  --gpu-memory-utilization="$GPU_FRAC"
  --tensor-parallel-size=1
  --enable-prefix-caching
)

wait_health() {
  local id=$1
  local start elapsed
  start=$(date +%s)
  echo "  waiting for /health (pid ${SERVER_PID}) ..."
  while (( $(date +%s) - start < 600 )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "  !! server died during startup"
      echo "died" >"$OUT/${id}_startup_status.txt"
      return 1
    fi
    if curl -sf "${URL}/health" >/dev/null 2>&1; then
      elapsed=$(( $(date +%s) - start ))
      echo "$elapsed" >"$OUT/${id}_startup_seconds.txt"
      echo "healthy" >"$OUT/${id}_startup_status.txt"
      echo "  up after ${elapsed}s (BTC gate: 600s)"
      return 0
    fi
    sleep 2
  done
  echo "timeout" >"$OUT/${id}_startup_status.txt"
  echo "  !! not healthy in 600s -- would fail the BTC startup gate"
  return 1
}

stop_server() {
  if [[ -n "$SERVER_PID" ]]; then
    kill -- -"$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
  sleep 5
}

run_equivalence() {
  local id=$1
  case "$id" in
    FP8A)
      python3 tools/evaluation/greedy_equivalence.py --url "$URL" \
        --write-reference "$FP8_REFERENCE" \
        | tee "$OUT/${id}_equivalence.log"
      ;;
    FP8B)
      if [[ -f "$FP8_REFERENCE" ]]; then
        python3 tools/evaluation/greedy_equivalence.py --url "$URL" \
          --reference "$FP8_REFERENCE" \
          | tee "$OUT/${id}_equivalence.log"
      else
        echo "EQUIVALENCE 0/0 (missing FP8A reference)" \
          | tee "$OUT/${id}_equivalence.log"
      fi
      ;;
    BNB4A)
      python3 tools/evaluation/greedy_equivalence.py --url "$URL" \
        --write-reference "$BNB4_REFERENCE" \
        | tee "$OUT/${id}_equivalence.log"
      ;;
    BNB4B)
      if [[ -f "$BNB4_REFERENCE" ]]; then
        python3 tools/evaluation/greedy_equivalence.py --url "$URL" \
          --reference "$BNB4_REFERENCE" \
          | tee "$OUT/${id}_equivalence.log"
      else
        echo "EQUIVALENCE 0/0 (missing BNB4A reference)" \
          | tee "$OUT/${id}_equivalence.log"
      fi
      ;;
  esac

  # Quantization may legitimately change output text. Record the difference
  # from FP8 for diagnosis, but never use it as a pass/fail gate.
  if [[ "$id" == BNB4* && -f "$FP8_REFERENCE" ]]; then
    python3 tools/evaluation/greedy_equivalence.py --url "$URL" \
      --reference "$FP8_REFERENCE" \
      2>&1 | tee "$OUT/${id}_cross_fp8.log" || true
  fi
}

trap stop_server EXIT INT TERM

for id in $ORDER; do
  extra=${CFG[$id]}
  server_log="$OUT/${id}_server.log"
  echo
  echo "############################################################"
  echo "# CONFIG $id : $extra"
  echo "############################################################"
  # Each value in CFG contains only whitespace-separated CLI options.
  # shellcheck disable=SC2086
  setsid taskset -c "$CORES" python3 -m vllm.entrypoints.openai.api_server \
    "${BASE_FLAGS[@]}" $extra >"$server_log" 2>&1 &
  SERVER_PID=$!
  if ! wait_health "$id"; then
    echo "  config $id FAILED to start -- last log lines:"
    tail -30 "$server_log"
    stop_server
    continue
  fi

  nvidia-smi --query-gpu=memory.used --format=csv,noheader \
    | head -1 | tee "$OUT/${id}_gpu_memory.txt"

  run_equivalence "$id"

  for mode in $MODES; do
    echo "--- replay mode=$mode (all 420 requests scored) ---"
    python3 tools/replay/replay_r2.py \
      --url "$URL" \
      --mode "$mode" \
      --tokenizer "$MODEL" \
      | tee "$OUT/${id}_${mode}.log"
  done

  echo "--- long-context needle ($id) ---"
  python3 tools/evaluation/needle_test.py "$URL" \
    | tee "$OUT/${id}_needle.log"
  stop_server
done

trap - EXIT INT TERM

echo
python3 tools/analysis/summarize_w4_battery.py "$OUT" \
  | tee "$OUT/summary.txt"
echo "Full logs: $OUT"
