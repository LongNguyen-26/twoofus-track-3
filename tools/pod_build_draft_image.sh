#!/usr/bin/env bash
# Build the round-2 draft-model spec-decode image WITHOUT docker, using crane,
# directly from the pod (datacenter bandwidth). Appends the draft model weights
# as a layer at /draft-model onto vllm/vllm-openai:v0.25.1 and pushes to Docker Hub.
# Same crane workflow as round 1's cache image (see CLAUDE.md).
#
# Usage (inside the pod, AFTER the D-series test picks a winner):
#   1) crane auth login index.docker.io -u <dockerhub-user>   # paste an access token, NOT your password
#   2) DRAFT_DIR=/workspace/draft230 TAG=<dockerhub-user>/lfm25-draft:v1 bash tools/pod_build_draft_image.sh
#
# The compose then uses:  image: <TAG>
#   --speculative-config={"model":"/draft-model","num_speculative_tokens":N,"quantization":"fp8"}
set -euo pipefail
DRAFT_DIR=${DRAFT_DIR:-/workspace/draft230}
TAG=${TAG:?set TAG=user/repo:tag}
BASE=vllm/vllm-openai:v0.25.1
WORK=/workspace/imgbuild
mkdir -p "$WORK" && cd "$WORK"

if ! command -v crane >/dev/null; then
  curl -fsSL -o crane.tgz https://github.com/google/go-containerregistry/releases/download/v0.20.2/go-containerregistry_Linux_x86_64.tar.gz
  tar -xzf crane.tgz crane && install crane /usr/local/bin/
fi

rm -rf stage && mkdir -p stage/draft-model
cp "$DRAFT_DIR"/*.json "$DRAFT_DIR"/*.safetensors stage/draft-model/
cp "$DRAFT_DIR"/tokenizer* "$DRAFT_DIR"/*.jinja stage/draft-model/ 2>/dev/null || true
tar --owner=0 --group=0 --numeric-owner --mode=u=rw,go=r -C stage -cf layer.tar draft-model
echo "layer:"; ls -la layer.tar

DIGEST=$(crane digest "$BASE")
echo "base digest: $DIGEST"
crane append -b "${BASE%%:*}@${DIGEST}" -f layer.tar -t "$TAG"

echo "== verify =="
crane config "$TAG" | python3 -c "import json,sys; c=json.load(sys.stdin)['config']; print('Entrypoint:', c.get('Entrypoint')); print('Cmd:', c.get('Cmd'))"
echo "pushed: $TAG  (make the Docker Hub repo PUBLIC before submitting)"
