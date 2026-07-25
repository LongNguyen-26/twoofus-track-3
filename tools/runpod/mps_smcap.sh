#!/usr/bin/env bash
# Start CUDA MPS and cap visible SMs so the pod approximates the BTC MiG slice.
#
# WHY: the BTC eval runs on a MiG 1g.18gb partition of an H200 = ~16 of 132 SMs
# plus ~1/8 of memory bandwidth. Every previous local rig (RTX 4090) had 128 SMs
# and therefore over-reported the value of anything that trades SM cycles for
# bytes -- which is exactly why fp8 KV (submit_021) and BitsAndBytes W4 both
# looked fine locally and lost on the portal.
#
# MPS is the only privilege-free way to cap SMs for a *separate* process: without
# it, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is silently ignored and kernels from
# different processes time-slice instead of sharing the GPU. So this script
# verifies the cap rather than trusting it.
#
# Usage:
#   bash tools/runpod/mps_smcap.sh start     # start daemon + measure capped/uncapped
#   bash tools/runpod/mps_smcap.sh verify    # re-measure only
#   bash tools/runpod/mps_smcap.sh stop
#
# After 'start', every process that should be capped must be launched with
# SM_CAP_ENV exported; h_battery_r2.sh does this for you.
set -uo pipefail

# H200 MiG 1g.18gb is ~16 of 132 SMs = 12.1%.
SM_PCT=${SM_PCT:-12}
export CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}
export CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

start_mps() {
  mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
  if pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "MPS control daemon already running"
    return 0
  fi
  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "!! nvidia-cuda-mps-control not present in this container."
    echo "!! SM capping is UNAVAILABLE -- results will overstate SM-hungry"
    echo "!! options (INT4 dequant, spec decode) exactly like the 4090 rig did."
    return 1
  fi
  nvidia-cuda-mps-control -d && echo "MPS control daemon started" && return 0
  echo "!! failed to start MPS daemon (needs GPU compute mode DEFAULT)"
  return 1
}

verify() {
  echo "--- uncapped baseline ---"
  env -u CUDA_MPS_ACTIVE_THREAD_PERCENTAGE python3 "$HERE/sm_cap_verify.py"
  echo "--- capped to ${SM_PCT}% ---"
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SM_PCT python3 "$HERE/sm_cap_verify.py"
  cat <<EOF

Read the two SMCAP_VERIFY lines above:
  * gemm_tflops must drop by roughly 132/16 ~ 8x. If it barely moves, the cap
    is NOT in force and every number from this pod is untrustworthy for
    kernel-level decisions -- say so in the writeup instead of shipping.
  * the bw_gbs ratio is the INT4 verdict in miniature. Capped bw_gbs near the
    real slice's ~600 GB/s means decode is still bandwidth bound and fewer
    weight bytes help. If bw_gbs collapses in proportion to SM count, decode is
    SM bound and INT4 buys nothing no matter how good the kernel is.
EOF
}

case "${1:-start}" in
  start)
    start_mps || exit 1
    verify
    echo
    echo "Export this for any process that must be capped:"
    echo "  export CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
    echo "  export CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$SM_PCT"
    ;;
  verify) verify ;;
  stop)
    echo quit | nvidia-cuda-mps-control 2>/dev/null || true
    echo "MPS stopped"
    ;;
  *) echo "usage: $0 {start|verify|stop}"; exit 2 ;;
esac
