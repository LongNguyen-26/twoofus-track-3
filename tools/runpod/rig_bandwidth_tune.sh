#!/usr/bin/env bash
# Find the bandwidth-hog setting that makes this pod behave like the MiG slice.
#
# The bare MPS cap failed its acceptance test: with 14 SMs but the whole HBM,
# bf16 and fp8 decoded at an identical 3.17ms while the portal pays 1.74ms for
# that difference. Adding a competing bandwidth consumer restores the missing
# constraint. Target: the capped client should see ~600 GB/s, matching the real
# 1g.18gb slice, instead of the 871 GB/s it sees on an idle GPU.
#
# Usage: bash tools/runpod/rig_bandwidth_tune.sh
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
SM_PCT=${SM_PCT:-12}
HOG_PID=""

cleanup() { [[ -n "$HOG_PID" ]] && kill "$HOG_PID" 2>/dev/null; sleep 2; }
trap cleanup EXIT

pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1 || {
  echo "!! MPS is not running; start it with tools/runpod/mps_smcap.sh start"; exit 1; }

echo "=== baseline: no hog ==="
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SM_PCT python3 "$HERE/sm_cap_verify.py"

for spec in "60 4" "80 8" "88 12"; do
  set -- $spec
  pct=$1; gb=$2
  echo
  echo "=== hog at ${pct}% SMs, ${gb}GB buffers ==="
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$pct HOG_GB=$gb python3 "$HERE/bw_hog.py" &
  HOG_PID=$!
  sleep 12
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SM_PCT python3 "$HERE/sm_cap_verify.py"
  kill "$HOG_PID" 2>/dev/null; wait "$HOG_PID" 2>/dev/null; HOG_PID=""
  sleep 3
done

cat <<'EOF'

TUNE BY THE RATIO, NOT BY ABSOLUTE BANDWIDTH. The hog steals SMs as well as
bandwidth, so chasing the slice's 600 GB/s on its own just over-starves compute.
What has to match is gemm_tflops / bw_gbs:

    MiG 1g.18gb on H200   ~115 TFLOPS / ~600 GB/s = 0.19
    bare 12% cap, no hog    98.6 / 868            = 0.114   (SM starved)
    hog 60% / 4GB           54.2 / 263            = 0.206   <- best match
    hog 80% / 8GB           42.1 / 212            = 0.199
    hog 88% / 12GB          34.9 / 203            = 0.172

The right setting makes the pod a uniformly slower scale model of the slice, so
latency RATIOS transfer even though absolute milliseconds do not. Prefer the
lightest hog that matches, since it costs the least wall-clock noise.

Then confirm the rig is finally valid before trusting it for anything:

  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=<pct> HOG_GB=<gb> python3 tools/runpod/bw_hog.py &
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=12 python3 tools/runpod/int4_microbench.py --marlin

The acceptance test is the bf16/fp8 column: it must climb from 1.13 toward the
portal's 1.43. If it does, rerun ONLY="H0 H1" h_battery_r2.sh with the hog
running and check the TPOT ratio too. If it stays near 1.0, no local rig models
this slice -- stop paying for pods and settle the remaining questions with
portal slots, which are free and are the actual measurement.
EOF
