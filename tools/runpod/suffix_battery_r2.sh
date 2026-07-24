#!/usr/bin/env bash
# Round-2 suffix-decoding battery for LFM2.5-1.2B-Instruct on RunPod.
#
# Default order brackets candidates with byte-identical baselines:
#   D0A -> S4 -> S8 -> S16 -> D0B
#
# Every replay uses the current 420-request scoring policy. Each config also
# runs exact greedy-output equivalence, the ~28k-token needle test, and saves
# vLLM speculative-decoding acceptance counters.
#
# Prerequisite on the stock vLLM image:
#   bash tools/runpod/pod_install_suffix.sh
#
# Useful overrides:
#   ONLY="D0A S8 D0B" MODES="fresh shared" \
#     bash tools/runpod/suffix_battery_r2.sh
#   CORES=0-2 VRAM_MB=17100 MODEL=/workspace/model \
#     bash tools/runpod/suffix_battery_r2.sh

set -uo pipefail
unset VLLM_API_KEY
export ARCTIC_INFERENCE_ENABLED=0

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
cd "$REPO_DIR"

MODEL=${MODEL:-/workspace/model}
PORT=${PORT:-8001}
URL="http://localhost:${PORT}"
CORES=${CORES:-0-2}
MODES=${MODES:-"fresh shared"}
VRAM_MB=${VRAM_MB:-17100}
OUT=${OUT:-"$REPO_DIR/results_suffix_$(date +%Y%m%d_%H%M%S)"}
REFERENCE="$OUT/equivalence_reference.json"
SERVER_PID=""
mkdir -p "$OUT"

if ! python3 -c \
  "from arctic_inference.suffix_decoding import SuffixDecodingCache" \
  >/dev/null 2>&1; then
  echo "ERROR: suffix extension is unavailable." >&2
  echo "Run: bash tools/runpod/pod_install_suffix.sh" >&2
  exit 2
fi

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
  --quantization=fp8
)

declare -A CFG
CFG[D0A]=""
CFG[S4]='--speculative-config {"method":"suffix","num_speculative_tokens":4,"suffix_decoding_max_tree_depth":24}'
CFG[S8]='--speculative-config {"method":"suffix","num_speculative_tokens":8,"suffix_decoding_max_tree_depth":24}'
CFG[S16]='--speculative-config {"method":"suffix","num_speculative_tokens":16,"suffix_decoding_max_tree_depth":24}'
CFG[D0B]=""
ORDER=${ONLY:-"D0A S4 S8 S16 D0B"}

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

save_spec_metrics() {
  local id=$1
  local metrics="$OUT/${id}_metrics.txt"
  curl -sf "${URL}/metrics" >"$metrics" || true
  grep -E '^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)(_total)?[{ ]' \
    "$metrics" || true
}

trap stop_server EXIT INT TERM

for id in $ORDER; do
  if [[ -z "${CFG[$id]+defined}" ]]; then
    echo "ERROR: unknown config ID: $id" >&2
    exit 2
  fi
  extra=${CFG[$id]}
  server_log="$OUT/${id}_server.log"
  echo
  echo "############################################################"
  echo "# CONFIG $id : ${extra:-<FP8 baseline>}"
  echo "############################################################"
  # JSON contains no spaces internally, so the intentional expansion below
  # becomes exactly two CLI arguments: --speculative-config and its JSON.
  # shellcheck disable=SC2086
  setsid taskset -c "$CORES" python3 -m vllm.entrypoints.openai.api_server \
    "${BASE_FLAGS[@]}" $extra >"$server_log" 2>&1 &
  SERVER_PID=$!
  if ! wait_health; then
    echo "  config $id FAILED to start -- last log lines:"
    tail -20 "$server_log"
    stop_server
    continue
  fi

  nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1

  if [[ "$id" == "D0A" ]]; then
    python3 tools/evaluation/greedy_equivalence.py --url "$URL" \
      --write-reference "$REFERENCE" | tee "$OUT/${id}_equivalence.log"
  elif [[ -f "$REFERENCE" ]]; then
    python3 tools/evaluation/greedy_equivalence.py --url "$URL" \
      --reference "$REFERENCE" | tee "$OUT/${id}_equivalence.log"
  else
    echo "  !! equivalence skipped: run D0A first to create $REFERENCE"
  fi

  for mode in $MODES; do
    echo "--- replay mode=$mode (all 420 requests scored) ---"
    python3 tools/replay/replay_r2.py \
      --url "$URL" \
      --mode "$mode" \
      --tokenizer "$MODEL" \
      | tee "$OUT/${id}_${mode}.log"
  done

  echo "--- long-context needle ($id) ---"
  python3 tools/evaluation/needle_test.py "$URL" | tee "$OUT/${id}_needle.log"
  echo "--- speculative metrics ($id) ---"
  save_spec_metrics "$id"
  stop_server
done

trap - EXIT INT TERM

echo
echo "==== ERS SUMMARY (all 420 requests scored) ===="
grep -H "ALL REQUESTS SCORED" "$OUT"/*_*.log 2>/dev/null | sed "s|$OUT/||" || true
echo
echo "==== CORRECTNESS SUMMARY ===="
grep -H -E "EQUIVALENCE [0-9]+/[0-9]+|RETRIEVAL [0-9]+/[0-9]+" \
  "$OUT"/*_*.log 2>/dev/null | sed "s|$OUT/||" || true
echo
echo "==== ACCEPTANCE SUMMARY ===="
grep -H -E '^vllm:spec_decode_num_(drafts|draft_tokens|accepted_tokens)(_total)?[{ ]' \
  "$OUT"/*_metrics.txt 2>/dev/null | sed "s|$OUT/||" || true
echo
python3 tools/analysis/summarize_suffix_battery.py "$OUT" \
  | tee "$OUT/summary.txt"
echo "Full logs: $OUT"
