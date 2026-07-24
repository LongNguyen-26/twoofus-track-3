#!/usr/bin/env bash
# Build arctic-inference 0.1.1 against the Python/PyTorch already present in
# vllm/vllm-openai:v0.25.1. The package's normal isolated build pins torch
# 2.7.0, so --no-build-isolation is intentional and required.
#
# Run as tools/runpod/pod_build_arctic_wheel.sh in a disposable RunPod based
# on vllm/vllm-openai:v0.25.1.
# Output defaults to /workspace/arctic-wheels.

set -euo pipefail

ARCTIC_VERSION=${ARCTIC_VERSION:-0.1.1}
WHEEL_DIR=${WHEEL_DIR:-/workspace/arctic-wheels}
BUILD_VENV=${BUILD_VENV:-/workspace/arctic-build-venv}
BUILD_JOBS=${BUILD_JOBS:-3}

if [[ "$ARCTIC_VERSION" != "0.1.1" ]]; then
  echo "ERROR: vLLM 0.25.1 explicitly expects arctic-inference==0.1.1" >&2
  exit 2
fi

mkdir -p "$WHEEL_DIR"
shopt -s nullglob
existing=("$WHEEL_DIR"/arctic_inference-0.1.1-*.whl)
shopt -u nullglob
if (( ${#existing[@]} > 0 )) && [[ "${FORCE_REBUILD:-0}" != "1" ]]; then
  echo "Reusing wheel: ${existing[0]}"
  exit 0
fi

if ! command -v g++ >/dev/null 2>&1; then
  echo "Installing the C++ compiler needed by the suffix-tree extension ..."
  apt-get update
  apt-get install -y --no-install-recommends build-essential
fi

# A system-site-packages venv sees the image's matching torch build, while
# build-only pins such as protobuf and nanobind remain isolated from vLLM.
python3 -m venv --clear --system-site-packages "$BUILD_VENV"
"$BUILD_VENV/bin/python" -m pip install --no-cache-dir --upgrade \
  pip setuptools wheel ninja cmake nanobind==2.9.2 \
  protobuf==5.29.5 grpcio-tools

"$BUILD_VENV/bin/python" -c \
  "import torch; print('building against torch', torch.__version__)"

export ARCTIC_INFERENCE_ENABLED=0
export CMAKE_BUILD_PARALLEL_LEVEL="$BUILD_JOBS"
"$BUILD_VENV/bin/python" -m pip wheel \
  --no-cache-dir \
  --no-build-isolation \
  --no-deps \
  --wheel-dir "$WHEEL_DIR" \
  "arctic-inference==$ARCTIC_VERSION"

shopt -s nullglob
built=("$WHEEL_DIR"/arctic_inference-0.1.1-*.whl)
shopt -u nullglob
if (( ${#built[@]} != 1 )); then
  echo "ERROR: expected exactly one Arctic wheel, found ${#built[@]}" >&2
  exit 1
fi
sha256sum "${built[0]}"
echo "Wheel ready: ${built[0]}"
