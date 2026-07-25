#!/usr/bin/env bash
# H-series: first round-2 battery on a Hopper pod with a REAL SM cap.
#
# Why this rig replaces the 4090 one:
#   * The BTC slice is MiG 1g.18gb of an H200 = ~16 of 132 SMs on Hopper.
#   * The old rig capped CPU cores and VRAM only, leaving 128 Ada SMs, so it
#     over-valued anything spending SM cycles to save bytes. That is precisely
#     how fp8 KV (local -6% TPOT, portal -2.4 points) and BitsAndBytes W4 both
#     produced false positives.
#   * Hopper also selects different fp8/int4 kernels than Ada, so Ada could
#     never answer the INT4 question either.
#
# H0 and H1 are CALIBRATION ANCHORS, not experiments. The portal has already
# measured both configs, so the rig is only trustworthy if it reproduces them:
#   H0 (= submit_020, fp8) -> portal TPOT 4.07ms, TTFT p50 42ms
#   H1 (= submit_018, bf16) -> portal TPOT 5.81ms
# Reproducing the 1.74ms gap between them is the acceptance test for this pod.
# If H0/H1 land far off, treat every later delta as unproven and say so.
#
# Portal TPOT is not reported directly; it is recovered from the result page as
#   ERS ~ (1-fail) * 0.5 * [s_ttft(TTFT_p50) + s_tpot]
# which reproduces submit_018's integer tbt_median of 6ms exactly.
#
# Prereqs: bash tools/runpod/pod_setup_h100.sh   (model + MPS cap + deps)
# Usage:   bash tools/runpod/h_battery_r2.sh
#          ONLY="H0 H2" bash tools/runpod/h_battery_r2.sh
set -uo pipefail
unset VLLM_API_KEY

MODEL=/workspace/model
PORT=8001
URL="http://localhost:${PORT}"
CORES=${CORES:-0-2}
VRAM_MB=${VRAM_MB:-17100}
SM_PCT=${SM_PCT:-12}
OUT="/workspace/repo/results_h_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
SERVER_PID=""
cd /workspace/repo

TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_FRAC=$(python3 -c "print(min(0.95, round(${VRAM_MB}/${TOTAL_MB}, 3)))")

# Cap SMs for every server we launch. Without the MPS daemon this is inert, so
# record whether it is actually active -- an uncapped run is still useful data
# but must never be compared against a capped one.
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
if pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1; then
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SM_PCT
  SM_STATE="CAPPED at ${SM_PCT}% (~$(python3 -c "print(round(132*${SM_PCT}/100))") of 132 SMs)"
else
  SM_STATE="!! UNCAPPED -- MPS daemon not running; SM-hungry options will look better than they are"
fi

{
  echo "date: $(date -Is)"
  echo "gpu: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
  echo "sm_cap: $SM_STATE"
  echo "cores: $CORES   vram_cap_mb: $VRAM_MB   gpu_frac: $GPU_FRAC"
  echo "vllm: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"
  echo "torch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null)"
} | tee "$OUT/environment.txt"

BASE_FLAGS=(
  --model=$MODEL
  --served-model-name=LFM2.5-1.2B-Instruct
  --host=0.0.0.0
  --port=$PORT
  --max-model-len=32768          # never lower: the harness runs a long-context probe
  --gpu-memory-utilization=$GPU_FRAC
  --tensor-parallel-size=1
  --enable-prefix-caching        # load-bearing: killing it cost 15.8 points (submit_017)
  --quantization=fp8
)

declare -A CFG ENVV SHARED
# --- calibration anchors -----------------------------------------------------
CFG[H0]=""                                        # = submit_020, portal 4.07ms
SHARED[H0]=1
CFG[H1]="--dtype bfloat16"                        # = submit_018, portal 5.81ms
# --- untested cheap levers ---------------------------------------------------
# 014 shipped FULL_AND_PIECEWISE and moved nothing, but FULL_DECODE_ONLY is a
# different mode and hybrid models have historically fallen back silently.
CFG[H2]='--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}'
# Three cores shared by the API server, the engine core and torch's OMP pool.
CFG[H3]="--async-scheduling"
ENVV[H3]="OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
# --- drift control: rerun the anchor last ------------------------------------
CFG[H4]=""

ORDER=${ONLY:-"H0 H1 H2 H3 H4"}

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
trap 'stop_server; exit 130' INT TERM

echo
echo "SM cap: $SM_STATE"

for id in $ORDER; do
  extra=${CFG[$id]:-}
  envv=${ENVV[$id]:-}
  log="$OUT/${id}_server.log"
  echo
  echo "############################################################"
  echo "# CONFIG $id : ${extra:-<fp8 baseline = submit_020>} ${envv}"
  echo "############################################################"
  # shellcheck disable=SC2086
  setsid env $envv taskset -c "$CORES" python3 -m vllm.entrypoints.openai.api_server \
    "${BASE_FLAGS[@]}" $extra >"$log" 2>&1 &
  SERVER_PID=$!
  if ! wait_health; then
    echo "  config $id FAILED to start - last log lines:"; tail -15 "$log"; stop_server; continue
  fi

  # Prove the compilation/cudagraph mode actually took effect instead of
  # assuming it did -- 014's null result was never verified this way.
  echo "  --- cudagraph / compile lines from the server log ---"
  grep -i "cudagraph\|capturing\|torch.compile" "$log" | tail -6 | sed 's/^/  /'
  nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1

  modes="fresh"
  [[ -n "${SHARED[$id]:-}" ]] && modes="fresh shared"
  for mode in $modes; do
    echo "--- replay mode=$mode ---"
    python3 tools/replay/replay_r2.py --url "$URL" --mode "$mode" --tokenizer "$MODEL" \
      | tee "$OUT/${id}_${mode}.log"
  done
  echo "--- needle test ($id) ---"
  python3 tools/evaluation/needle_test.py "$URL" | tee "$OUT/${id}_needle.log"
  stop_server
done

echo
echo "==================== H SUMMARY ===================="
echo "SM cap: $SM_STATE"
grep -H "SCORED" "$OUT"/*_fresh.log "$OUT"/*_shared.log 2>/dev/null | sed "s|$OUT/||"
echo "--- tpot ---"
grep -H "tpot_ms:" "$OUT"/*_fresh.log 2>/dev/null | sed "s|$OUT/||"
echo "--- ttft ---"
grep -H "ttft_ms:" "$OUT"/*_fresh.log 2>/dev/null | sed "s|$OUT/||"
echo "--- needle (judge RELATIVE to H0; only 0/5 is disqualifying) ---"
grep -H "RETRIEVAL" "$OUT"/*_needle.log 2>/dev/null | sed "s|$OUT/||"
cat <<EOF

ACCEPTANCE TEST FOR THIS RIG
  H0 fresh tpot p50 should land near the portal's 4.07ms and H1 near 5.81ms.
  The gap matters more than the absolutes: reproducing ~1.74ms means the rig
  finally models the slice's decode bandwidth and later deltas can be trusted.
  H4 minus H0 is the local noise floor -- no experiment smaller than that gap
  is real.

Full logs in $OUT/
EOF
