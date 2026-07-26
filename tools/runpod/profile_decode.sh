#!/usr/bin/env bash
# Profile one decode step under slice-like conditions.
#
# Answers the last open question in the cost model. The budget is
#   TPOT 4.07ms = 1.73 (fp8 linear weights) + 0.45 (BF16 head) + ~1.9 (???)
# where the first two are byte counts confirmed against the portal to 1% and the
# third has only ever been obtained by subtraction. It is now the largest term
# and the only place a lever bigger than +2 points can still be.
#
# Two arms:
#   P1  CUDA graphs on  -- the real serving configuration
#   P2  eager           -- attribution cross-check; if P1 and P2 disagree about
#                          the shares, the graph is hiding work and P2 is the
#                          honest picture of what the kernels cost
#
# Read SHARES, not milliseconds. This rig runs ~1.9x the portal's TPOT and
# overstates bandwidth ~1.4x, so absolute figures do not transfer -- but the
# ranking does, and the ranking is the whole point.
#
# Prereqs: bash tools/runpod/pod_setup_h100.sh   (model + MPS cap)
# Usage:   bash tools/runpod/profile_decode.sh
#          ONLY=P1 bash tools/runpod/profile_decode.sh
set -uo pipefail
unset VLLM_API_KEY

MODEL=/workspace/model
CORES=${CORES:-0-2}
VRAM_MB=${VRAM_MB:-17100}
SM_PCT=${SM_PCT:-12}
HOG_PCT=${HOG_PCT:-60}
HOG_GB=${HOG_GB:-4}
CONTEXT=${CONTEXT:-4000}
STEPS=${STEPS:-200}
ONLY=${ONLY:-"P1 P2"}
OUT="/workspace/repo/results_profile_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
cd /workspace/repo

TOTAL_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
GPU_FRAC=$(python3 -c "print(min(0.95, round(${VRAM_MB}/${TOTAL_MB}, 3)))")

export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
if pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1; then
  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SM_PCT
  SM_STATE="CAPPED at ${SM_PCT}%"
else
  SM_STATE="!! UNCAPPED -- shares will be wrong; a 132-SM GPU hides small-kernel cost"
fi
echo "SM cap: $SM_STATE" | tee "$OUT/rig.txt"

# Keeps the engine in-process so the profiler sees the model forward directly
# instead of across the EngineCore process boundary.
export VLLM_ENABLE_V1_MULTIPROCESSING=0

HOG_PID=""
if [ "$HOG_PCT" != "0" ]; then
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$HOG_PCT \
    python3 tools/runpod/bw_hog.py --gb "$HOG_GB" >"$OUT/hog.log" 2>&1 &
  HOG_PID=$!
  sleep 5
fi
cleanup() { [ -n "$HOG_PID" ] && kill "$HOG_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

run() {  # $1 = tag, rest = extra args
  local tag=$1; shift
  echo
  echo "==================== $tag ===================="
  taskset -c "$CORES" python3 tools/runpod/decode_profile.py \
    --model "$MODEL" --context "$CONTEXT" --steps "$STEPS" \
    --quantization fp8 --gpu-memory-utilization "$GPU_FRAC" \
    "$@" 2>&1 | tee "$OUT/${tag}.log"
}

for tag in $ONLY; do
  case "$tag" in
    P1) run P1 ;;
    P2) run P2 --eager ;;
  esac
done

echo
echo "==================== SUMMARY ===================="
echo "SM cap: $SM_STATE   context: $CONTEXT   steps: $STEPS"
for f in "$OUT"/P*.log; do
  [ -e "$f" ] || continue
  echo
  echo "--- $(basename "$f" .log) ---"
  sed -n '/category/,/TOTAL GPU/p' "$f"
  grep -E "GPU kernel time attributed|host \(CPU\) time|clean wall clock" "$f"
done
cat <<EOF

What to do with this:
  * One category above ~25% of the step is a target. Anything under ~5% is not
    worth an image, however easy it looks.
  * If "unclassified" is large, read the top-kernel list -- the bucket names are
    heuristic and a big unnamed kernel is exactly what we are hunting.
  * If host (CPU) time exceeds the GPU total, the step is CPU bound and no
    kernel work helps; that would redirect everything toward the serving path.
  * P1 vs P2 disagreeing means CUDA graphs are hiding work; trust P2's shares.
Full logs: $OUT
EOF
