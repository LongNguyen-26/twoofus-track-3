#!/usr/bin/env bash
# Overnight experiment battery for the Viettel AI Race pod (RunPod RTX 4090).
#
# HOST DRIVER REQUIREMENT: vllm/vllm-openai:v0.22.1 is built on CUDA 13.0, so
# the RunPod host needs driver R580+ (CUDA >= 13.0). Cheap 4090 hosts on older
# drivers fail at container init with "nvidia-container-cli: unsatisfied
# condition: cuda>=13.0". On deploy, use Filter -> CUDA Version >= 13.0.
#
# Runs each vLLM config in turn, simulating the MiG H200 eval slice
# (18GB VRAM cap via --gpu-memory-utilization, 3 CPU cores via taskset),
# and for each: prints cold-vs-warm TTFT (prefix-cache probe) + a full
# 2-pass replay (primer + scored) with local ERS.
#
# KEEPALIVE: the vllm image's entrypoint IS the server, so the container can't
# host a shell on its own. Set the pod's Container Start Command to a tiny
# keepalive server so the container stays up and Web Terminal can attach:
#   --model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 \
#       --gpu-memory-utilization 0.1 --max-model-len 2048
# That keepalive owns port 8000; THIS script runs the real configs on port 8001
# as children in its own process group, so it never kills the keepalive (PID 1).
#
# PREREQ (run once in the pod Web Terminal, after keepalive is Ready):
#   cd /workspace
#   git clone https://github.com/LongNguyen-26/twoofus-track-3.git repo && cd repo
#   pip install -U huggingface_hub hf_transfer aiohttp
#   export HF_HUB_ENABLE_HF_TRANSFER=1
#   huggingface-cli download Qwen/Qwen3.5-2B --local-dir /workspace/model
#
# Then from the repo root:  bash tools/pod_overnight.sh
#
# NOTE: this covers all configs that share ONE image (v0.22.1). To test
# v0.24.0, redeploy the pod with image vllm/vllm-openai:v0.24.0 and rerun.

set -uo pipefail

# The RunPod template may inject VLLM_API_KEY into the container env; a test
# server inheriting it would 401 every request (exactly what a stale run saw).
unset VLLM_API_KEY

MODEL=/workspace/model
PORT=8001                         # 8000 is held by the keepalive server (PID 1)
URL="http://localhost:${PORT}"
OUT="results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
SERVER_PID=""                     # process-group leader of the current test server

# Flags shared by every config — ONLY the ones that mimic the eval box or are
# hard requirements (see CLAUDE.md):
#  - max-model-len 40960 : prompts reach 27,398 tokens
#  - gpu-mem-util 0.71   : 0.71*24GB = 17.1GB = 0.95*18GB (mimics MiG cap)
# Scheduler knobs (chunked prefill, batch budget, seqs) are deliberately NOT
# here — they are the experiment variables.
COMMON=(
  --model "$MODEL"
  --served-model-name Qwen3.5-2B
  --host 0.0.0.0 --port "$PORT"
  --max-model-len 40960
  --gpu-memory-utilization 0.71
  --tensor-parallel-size 1
)

wait_health() {
  # Abort early if the server process dies (bad config) instead of
  # burning the full 600s window.
  echo "  waiting for /health (pid ${SERVER_PID}) ..."
  local start
  start=$(date +%s)
  while (( $(date +%s) - start < 600 )); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "  !! server process died during startup"
      return 1
    fi
    if curl -sf "${URL}/health" >/dev/null 2>&1; then
      echo "  server up after $(( $(date +%s) - start ))s"
      return 0
    fi
    sleep 2
  done
  echo "  !! server not healthy in 600s (would FAIL the BTC 600s gate)"
  return 1
}

stop_server() {
  # Self-healing teardown. Runs even with no tracked SERVER_PID so it also
  # clears orphans left by an earlier crashed run.

  # 1) kill this test server's process group, if we launched one
  if [ -n "$SERVER_PID" ]; then
    kill -TERM -"$SERVER_PID" 2>/dev/null
    for i in $(seq 1 20); do
      kill -0 -"$SERVER_PID" 2>/dev/null || break
      sleep 1
      [ "$i" -eq 10 ] && kill -9 -"$SERVER_PID" 2>/dev/null
    done
    SERVER_PID=""
  fi

  # 2) the vLLM V1 EngineCore has no model path in its cmdline, so pattern
  #    kills miss it. Kill it by GPU footprint instead: anything using >5GB is
  #    a leftover test engine; the keepalive (0.6B) sits ~3.5GB and is spared.
  local hogs
  hogs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
         | tr -d ',' | awk '$2>5000{print $1}')
  [ -n "$hogs" ] && kill -9 $hogs 2>/dev/null

  # 3) free our port too, in case an APIServer is still bound to it
  local pport
  pport=$(ss -ltnHp "sport = :${PORT}" 2>/dev/null | grep -oP 'pid=\K[0-9]+')
  [ -n "$pport" ] && kill -9 $pport 2>/dev/null

  # 4) wait for the GPU to settle back down to roughly the keepalive footprint
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 5000 ] && break
    sleep 2
  done
  echo "  gpu now ${used:-?} MiB used (keepalive only if ~3500)"
}

run_cfg() {
  local name="$1"; shift
  echo ""
  echo "############################################################"
  echo "# CONFIG: ${name}"
  echo "# extra flags: $*"
  echo "############################################################"

  stop_server

  # Refuse to run if something already answers on our port — otherwise the
  # health check "passes" instantly and we benchmark the wrong server.
  if curl -sf "${URL}/health" >/dev/null 2>&1; then
    echo "  !! port ${PORT} already answering /health (stale server?) — aborting battery"
    exit 1
  fi

  # setsid puts the server in its own process group so stop_server can kill it
  # (and its vLLM worker children) without touching the keepalive on port 8000.
  # 3-core CPU cap mimics the eval slice; server log captured for grepping
  # (look for 'prefix', 'chunked', 'not supported', 'speculative').
  setsid taskset -c 0-2 python3 -m vllm.entrypoints.openai.api_server \
    "${COMMON[@]}" "$@" > "${OUT}/server_${name}.log" 2>&1 &
  SERVER_PID=$!

  if ! wait_health; then
    echo "  --> skipping benchmark for ${name} (see ${OUT}/server_${name}.log)"
    tail -n 30 "${OUT}/server_${name}.log"
    stop_server
    return
  fi

  # Identity check: make sure the thing answering on $PORT is OUR model,
  # not the keepalive or a stale server.
  if ! curl -s "${URL}/v1/models" | grep -q '"Qwen3.5-2B"'; then
    echo "  !! server on ${PORT} is not serving Qwen3.5-2B — see ${OUT}/server_${name}.log"
    stop_server
    return
  fi

  {
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    echo "=== PROBE (cold vs warm TTFT on request 0) ==="
    python3 tools/replay_trace.py --url "$URL" --probe 0
    echo ""
    echo "=== FULL 2-PASS REPLAY (primer + scored, wave-timed) ==="
    python3 tools/replay_trace.py --url "$URL" --passes 2
    echo ""
    echo "=== prefix-cache stats from server log ==="
    grep -i "hit rate\|prefix" "${OUT}/server_${name}.log" | tail -5
  } 2>&1 | tee "${OUT}/bench_${name}.txt"

  stop_server
}

# --- The battery (v0.22.1 image) ------------------------------------------
# E0  replica004 : exact flags of submit_004 (portal: 14.49 on 09/07 03:07).
#     Local ERS here = the calibration anchor between this pod and BTC's MiG.
run_cfg replica004 --enable-prefix-caching

# E1  sched      : submit_006 candidate — scheduler tuning, no quantization.
run_cfg sched      --enable-prefix-caching --enable-chunked-prefill \
                   --max-num-batched-tokens 16384 --max-num-seqs 32

# E2  sched_kvfp8: E1 + kv fp8 (what submit_005 would have scored; also tests
#     whether fp8 KV hurts/breaks the prefix-cache warm hit).
run_cfg sched_kvfp8 --enable-prefix-caching --enable-chunked-prefill \
                    --max-num-batched-tokens 16384 --max-num-seqs 32 \
                    --kv-cache-dtype fp8

# E3  sched_mtp  : E1 + MTP speculative decoding -> does TBT drop from ~59ms?
run_cfg sched_mtp  --enable-prefix-caching --enable-chunked-prefill \
                   --max-num-batched-tokens 16384 --max-num-seqs 32 \
                   --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":2}'

echo ""
echo "ALL DONE. Results in ${OUT}/  (bench_*.txt = numbers, server_*.log = boot logs)"
echo "Key questions: (1) does warm TTFT collapse to ~ms in E0/E1?"
echo "               (2) does E2 lose the warm hit vs E1?  (3) E3 TBT vs E1?"
echo "REMINDER: STOP THE POD now to stop billing."
