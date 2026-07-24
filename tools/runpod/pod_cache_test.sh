#!/usr/bin/env bash
# In-pod test for the semantic-cache patch (image/*.py) BEFORE building the
# Docker image. The pod already runs vllm/vllm-openai:v0.22.1 — the same base
# as the final image — so installing the patch into this pod's site-packages
# reproduces the final image's behavior exactly.
#
# Usage (pod Web Terminal, from the repo root, keepalive on 8000 already up):
#   cd /workspace/repo && git pull && bash tools/runpod/pod_cache_test.sh
#   QUANT=0 bash tools/runpod/pod_cache_test.sh
#
# Success criteria (printed as VERDICT lines at the end):
#   1) byte-identity: the 2nd identical curl returns byte-identical JSON
#      (same "id"/"created" -> proves it came from the cache, not re-inference)
#   2) probe warm TTFT ~ a few ms (cache HIT; APC-only was ~44ms)
#   3) 2-pass replay: pass 2 ERS >= 0.98
set -uo pipefail
unset VLLM_API_KEY

MODEL=/workspace/model
PORT=8001                        # 8000 is held by the keepalive server (PID 1)
URL="http://localhost:${PORT}"
OUT="results_cache_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT"
SERVER_PID=""
QUANT=${QUANT:-1}

# ---- 1. install the patch into this pod's python (mirrors the Dockerfile) ---
SP=$(python3 -c "import site; print(site.getsitepackages()[0])")
cp image/vllm_exact_cache.py      "$SP/vllm_exact_cache.py"
cp image/vllm_exact_cache_boot.py "$SP/vllm_exact_cache_boot.py"
cp image/exact_cache.pth          "$SP/exact_cache.pth"
cp image/sitecustomize.py         "$SP/sitecustomize.py"
python3 -c "import vllm_exact_cache_boot" || { echo "patch import FAILED"; exit 1; }
# The .pth is the real mechanism: confirm it auto-runs at interpreter startup
# (this is exactly what failed before — sitecustomize was shadowed).
if ! python3 -c "pass" 2>&1 | grep -q "exact-cache] FastAPI autopatch installed"; then
  echo "!! .pth autoloader did NOT run at startup — patch would not load in the server"
  echo "   sys.path / sitecustomize diagnostics:"
  python3 -c "import site,sys; print('getsitepackages:', site.getsitepackages()); print('sys.path[:6]:', sys.path[:6])"
  exit 1
fi
echo "patch installed into $SP and .pth autoload confirmed"

wait_health() {
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
  if [ -n "$SERVER_PID" ]; then
    kill -TERM -"$SERVER_PID" 2>/dev/null
    for i in $(seq 1 20); do
      kill -0 -"$SERVER_PID" 2>/dev/null || break
      sleep 1
      [ "$i" -eq 10 ] && kill -9 -"$SERVER_PID" 2>/dev/null
    done
    SERVER_PID=""
  fi
  local hogs
  hogs=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits \
         | tr -d ',' | awk '$2>5000{print $1}')
  [ -n "$hogs" ] && kill -9 $hogs 2>/dev/null
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 5000 ] && break
    sleep 2
  done
  echo "  gpu now ${used:-?} MiB used"
}

# ---- 2. start the submit_009 candidate: 008 flags (best portal baseline) ----
stop_server
if curl -sf "${URL}/health" >/dev/null 2>&1; then
  echo "!! port ${PORT} already answering — stale server, aborting"
  exit 1
fi

EXTRA=()
[ "$QUANT" = "1" ] && EXTRA+=(--quantization fp8)
setsid taskset -c 0-2 python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name Qwen3.5-2B \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len 40960 --gpu-memory-utilization 0.71 \
  --tensor-parallel-size 1 --enable-prefix-caching \
  "${EXTRA[@]}" > "${OUT}/server_cache.log" 2>&1 &
SERVER_PID=$!

if ! wait_health; then
  tail -n 30 "${OUT}/server_cache.log"
  stop_server
  exit 1
fi
if ! curl -s "${URL}/v1/models" | grep -q '"Qwen3.5-2B"'; then
  echo "!! wrong server on ${PORT}"; stop_server; exit 1
fi
if ! grep -q "exact-cache" "${OUT}/server_cache.log"; then
  echo "!! server log has no [exact-cache] lines — patch did not load"; stop_server; exit 1
fi

{
  # ---- 3. byte-identity check (identical id/created == served from cache) ---
  REQ='{"model":"Qwen3.5-2B","messages":[{"role":"user","content":"Count from 1 to 5."}],"max_tokens":32,"temperature":0,"seed":42}'
  curl -s "${URL}/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ" -o "${OUT}/r1.json"
  curl -s "${URL}/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ" -o "${OUT}/r2.json"
  if cmp -s "${OUT}/r1.json" "${OUT}/r2.json"; then
    echo "VERDICT byte-identity: PASS"
  else
    echo "VERDICT byte-identity: FAIL"
    head -c 400 "${OUT}/r1.json"; echo; head -c 400 "${OUT}/r2.json"; echo
  fi

  # ---- 4. probe + full 2-pass replay (client runs unpatched) ----------------
  echo "=== PROBE (cold vs warm; warm should be ~ms now, was ~44ms APC-only) ==="
  VLLM_EXACT_CACHE=0 python3 tools/replay/replay_trace.py --url "$URL" --probe 0
  echo ""
  echo "=== FULL 2-PASS REPLAY (pass 2 should be ERS >= 0.98) ==="
  VLLM_EXACT_CACHE=0 python3 tools/replay/replay_trace.py --url "$URL" --passes 2
  echo ""
  echo "=== cache counters from server log ==="
  echo "STOREs: $(grep -c 'exact-cache] STORE' "${OUT}/server_cache.log")"
  echo "HITs  : $(grep -c 'exact-cache] HIT'   "${OUT}/server_cache.log")"
} 2>&1 | tee "${OUT}/bench_cache.txt"

stop_server
echo ""
echo "Results saved in ${OUT}/. If all three verdicts pass, build & push the"
echo "image from the Windows machine (see image/Dockerfile header), then fill"
echo "submissions/submit_009/docker-compose.yml with your Docker Hub image."
echo "REMINDER: STOP THE POD to stop billing."
