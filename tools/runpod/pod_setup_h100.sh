#!/usr/bin/env bash
# One-shot setup for the Hopper (H100 SXM) round-2 pod.
#
# Run this once in the pod Web Terminal after the keepalive server is Ready:
#   cd /workspace && git clone https://github.com/LongNguyen-26/twoofus-track-3.git repo
#   cd repo && bash tools/runpod/pod_setup_h100.sh
#
# Budget note: at $2.99/hr a $9.60 balance is ~3 hours. This script is the only
# unavoidable overhead (~8 minutes); everything after it is measurement.
set -uo pipefail
cd /workspace/repo

echo "==== 1. deps ===="
pip install -q -U huggingface_hub aiohttp 2>&1 | tail -2

echo
echo "==== 2. model ===="
# HF_HUB_DISABLE_XET=1 is mandatory: on a network volume the xet path deadlocks
# on .lock files and hangs in "Reconstructing". Killed downloads leave stale
# locks -- recover with: pkill -9 -f "hf download" && rm -rf /workspace/model
# `huggingface-cli` was removed in the 2026 hub releases and now exits with a
# deprecation stub, so prefer `hf` and fall back to the python API.
if [[ -f /workspace/model/config.json ]]; then
  echo "already present at /workspace/model"
elif command -v hf >/dev/null 2>&1; then
  HF_HUB_DISABLE_XET=1 hf download LiquidAI/LFM2.5-1.2B-Instruct \
    --local-dir /workspace/model
else
  HF_HUB_DISABLE_XET=1 python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("LiquidAI/LFM2.5-1.2B-Instruct", local_dir="/workspace/model")
PY
fi
[[ -f /workspace/model/config.json ]] || { echo "!! model download failed"; exit 1; }
python3 - <<'PY'
import json
c = json.load(open("/workspace/model/config.json"))
print("arch:", c.get("architectures"), "hidden:", c.get("hidden_size"),
      "layers:", c.get("num_hidden_layers"), "vocab:", c.get("vocab_size"))
PY

echo
echo "==== 3. SM cap (the entire point of moving to Hopper) ===="
bash tools/runpod/mps_smcap.sh start

echo
echo "==== 4. ready ===="
cat <<'EOF'
Run, in this order:

  # ~2 min, no GPU: has the spec-decode door reopened on vLLM main?
  bash tools/runpod/nightly_source_probe.sh

  # ~5 min: is batch-1 decode bandwidth bound or SM bound at 16 SMs?
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=12 \
    python3 tools/runpod/int4_microbench.py --marlin | tee /workspace/repo/int4_micro.log

  # ~45 min: calibration anchors + cheap untested levers
  bash tools/runpod/h_battery_r2.sh 2>&1 | tee /workspace/repo/h_battery.log

Stop the pod the moment the battery finishes -- billing is per millisecond.
EOF
