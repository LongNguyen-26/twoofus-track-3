#!/usr/bin/env bash
# Install the locally built Arctic suffix extension into the current RunPod
# container so the suffix battery can run before a custom image is published.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
WHEEL_DIR=${WHEEL_DIR:-/workspace/arctic-wheels}
export WHEEL_DIR

bash "$SCRIPT_DIR/pod_build_arctic_wheel.sh"

shopt -s nullglob
wheels=("$WHEEL_DIR"/arctic_inference-0.1.1-*.whl)
shopt -u nullglob
if (( ${#wheels[@]} != 1 )); then
  echo "ERROR: expected exactly one Arctic wheel, found ${#wheels[@]}" >&2
  exit 1
fi

export ARCTIC_INFERENCE_ENABLED=0
python3 -m pip install --no-cache-dir --no-deps --force-reinstall "${wheels[0]}"
python3 -c \
  "from importlib.metadata import version; from arctic_inference.suffix_decoding import SuffixDecodingCache; c=SuffixDecodingCache(max_tree_depth=24,max_cached_requests=8); print('arctic-inference',version('arctic-inference'),'suffix import OK')"

echo "Live pod is ready for: bash tools/runpod/suffix_battery_r2.sh"
