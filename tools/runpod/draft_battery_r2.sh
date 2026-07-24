#!/usr/bin/env bash
# D-series: draft-model speculative decoding battery (LFM2.5-1.2B target, fp8)
set -uo pipefail
unset VLLM_API_KEY
MODEL=/workspace/model
PORT=8001
URL="http://localhost:${PORT}"
CORES=${CORES:-0-2}
MODES=${MODES:-"fresh shared"}
VRAM_MB=${VRAM_MB:-17100}
OUT="/workspace/repo/results_draft_$(date +%Y%m%d_%H%M%S)"
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
CFG[D0]=""
CFG[D1]="--speculative-config {\"model\":\"/workspace/draft230\",\"num_speculative_tokens\":3,\"quantization\":\"fp8\"}"
CFG[D2]="--speculative-config {\"model\":\"/workspace/draft350\",\"num_speculative_tokens\":3,\"quantization\":\"fp8\"}"
CFG[D3]="--speculative-config {\"model\":\"/workspace/draft230\",\"num_speculative_tokens\":2,\"quantization\":\"fp8\"}"
CFG[D4]="--speculative-config {\"model\":\"/workspace/draft230\",\"num_speculative_tokens\":5,\"quantization\":\"fp8\"}"
CFG[D5]="--speculative-config {\"model\":\"/workspace/draft350\",\"num_speculative_tokens\":2,\"quantization\":\"fp8\"}"
# DEAD ENDS on v0.25.1 (kept for the record): D1/D2/D6/D8 — draft quantization hits
# "hf_overrides must be a dict" (internal wrap, CLI {} doesn't help); D7 — hybrid drafts
# rejected ("All drafting layers should belong to the same kv cache group") ⇒ no
# LFM2-family draft is possible. Remaining tracks: suffix (D9), heterogeneous-vocab
# dense draft via TLI (D10, SmolLM2-135M; requires hf download to /workspace/draft_smol).
# D9 requires: pip install arctic-inference==0.1.1
CFG[D6]="--hf-overrides {} --speculative-config {\"model\":\"/workspace/draft230\",\"num_speculative_tokens\":3,\"quantization\":\"fp8\"}"
CFG[D7]="--speculative-config {\"model\":\"/workspace/draft230\",\"num_speculative_tokens\":3}"
CFG[D8]="--hf-overrides {} --speculative-config {\"model\":\"/workspace/draft350\",\"num_speculative_tokens\":3,\"quantization\":\"fp8\"}"
CFG[D9]="--speculative-config {\"method\":\"suffix\"}"
CFG[D10]="--speculative-config {\"model\":\"/workspace/draft_smol\",\"num_speculative_tokens\":3,\"use_heterogeneous_vocab\":true}"
ORDER=${ONLY:-"D0 D1 D2"}

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
  log="$OUT/${id}_server.log"
  echo
  echo "############################################################"
  echo "# CONFIG $id : ${extra:-<fp8 baseline>}"
  echo "############################################################"
  # shellcheck disable=SC2086
  setsid taskset -c "$CORES" python3 -m vllm.entrypoints.openai.api_server \
    "${BASE_FLAGS[@]}" $extra >"$log" 2>&1 &
  SERVER_PID=$!
  if ! wait_health; then
    echo "  config $id FAILED to start - last log lines:"
    tail -5 "$log"
    stop_server
    continue
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1
  for mode in $MODES; do
    echo "--- replay mode=$mode ---"
    python3 tools/replay/replay_r2.py --url "$URL" --mode "$mode" --tokenizer "$MODEL" \
      | tee "$OUT/${id}_${mode}.log"
  done
  echo "--- needle test ($id) ---"
  python3 tools/evaluation/needle_test.py "$URL" | tee "$OUT/${id}_needle.log"
  stop_server
done

echo
echo "==== SUMMARY ===="
grep -H "SCORED" "$OUT"/*_*.log 2>/dev/null | sed "s|$OUT/||"
grep -H "RETRIEVAL" "$OUT"/*_needle.log 2>/dev/null | sed "s|$OUT/||"
echo "Full logs in $OUT/"
