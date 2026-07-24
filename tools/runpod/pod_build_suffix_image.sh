#!/usr/bin/env bash
# Append the tested arctic-inference wheel as a clean site-packages layer on
# vllm/vllm-openai:v0.25.1, then push the resulting image with crane.
#
# No Docker daemon is required. Run from a disposable RunPod:
#   crane auth login index.docker.io -u <dockerhub-user>
#   TAG=<dockerhub-user>/lfm25-suffix:v1 \
#     bash tools/runpod/pod_build_suffix_image.sh
#
# The Docker Hub repository MUST be public before a BTC submission.

set -euo pipefail

TAG=${TAG:?set TAG=<dockerhub-user>/lfm25-suffix:<tag>}
BASE=${BASE:-vllm/vllm-openai:v0.25.1}
WHEEL_DIR=${WHEEL_DIR:-/workspace/arctic-wheels}
BUILD_ROOT=${BUILD_ROOT:-/workspace/suffix-image-build}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export WHEEL_DIR

case "$BUILD_ROOT" in
  /workspace/*) ;;
  *)
    echo "ERROR: BUILD_ROOT must stay under /workspace: $BUILD_ROOT" >&2
    exit 2
    ;;
esac

bash "$SCRIPT_DIR/pod_build_arctic_wheel.sh"

shopt -s nullglob
wheels=("$WHEEL_DIR"/arctic_inference-0.1.1-*.whl)
shopt -u nullglob
if (( ${#wheels[@]} != 1 )); then
  echo "ERROR: expected exactly one Arctic wheel, found ${#wheels[@]}" >&2
  exit 1
fi

mkdir -p "$BUILD_ROOT"
STAGE="$BUILD_ROOT/stage"
if [[ -e "$STAGE" ]]; then
  resolved_stage=$(realpath "$STAGE")
  case "$resolved_stage" in
    "$BUILD_ROOT"/*) rm -rf -- "$resolved_stage" ;;
    *)
      echo "ERROR: refusing to clear unexpected stage path: $resolved_stage" >&2
      exit 2
      ;;
  esac
fi

SITE_DIR=$(python3 -c "import site; print(site.getsitepackages()[0])")
SITE_REL=${SITE_DIR#/}
STAGE_SITE="$STAGE/$SITE_REL"
mkdir -p "$STAGE_SITE"

python3 -m pip install \
  --no-cache-dir \
  --no-compile \
  --no-deps \
  --target "$STAGE_SITE" \
  "${wheels[0]}"

PYTHONPATH="$STAGE_SITE" ARCTIC_INFERENCE_ENABLED=0 python3 -c \
  "from arctic_inference.suffix_decoding import SuffixDecodingCache; SuffixDecodingCache(max_tree_depth=24,max_cached_requests=8); print('staged suffix import OK')"

LAYER="$BUILD_ROOT/arctic-suffix-layer.tar"
tar --owner=0 --group=0 --numeric-owner --mode=u=rwX,go=rX \
  -C "$STAGE" -cf "$LAYER" "$SITE_REL"
sha256sum "$LAYER"

if ! command -v crane >/dev/null 2>&1; then
  CRANE_VERSION=${CRANE_VERSION:-v0.20.2}
  curl -fsSL -o "$BUILD_ROOT/crane.tgz" \
    "https://github.com/google/go-containerregistry/releases/download/${CRANE_VERSION}/go-containerregistry_Linux_x86_64.tar.gz"
  tar -xzf "$BUILD_ROOT/crane.tgz" -C "$BUILD_ROOT" crane
  install "$BUILD_ROOT/crane" /usr/local/bin/crane
fi

DIGEST=$(crane digest "$BASE")
BASE_REPO=${BASE%:*}
echo "Base digest: $DIGEST"
crane append -b "${BASE_REPO}@${DIGEST}" -f "$LAYER" -t "$TAG"

echo "== Published image config =="
crane config "$TAG" | python3 -c \
  "import json,sys; c=json.load(sys.stdin)['config']; print('Entrypoint:',c.get('Entrypoint')); print('Cmd:',c.get('Cmd'))"
echo "Pushed: $TAG"
echo "Next: make the Docker Hub repository PUBLIC, redeploy this image, and rerun a smoke battery."
