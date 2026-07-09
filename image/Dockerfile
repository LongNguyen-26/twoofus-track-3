# Semantic-cache image for Viettel AI Race round 1.
# Base = the proven baseline runtime; we only add an exact-match response cache
# at the FastAPI/ASGI layer. The enforced entrypoint module
# `vllm.entrypoints.openai.api_server` is untouched and still serves real vLLM.
#
# Build (on the Windows machine — RunPod pods have no docker daemon):
#   docker build -t <dockerhub_user>/qwen35-cache:v1 image/
#   docker push <dockerhub_user>/qwen35-cache:v1     (repo must be PUBLIC)
FROM vllm/vllm-openai:v0.22.1

COPY vllm_exact_cache.py      /tmp/patch/vllm_exact_cache.py
COPY vllm_exact_cache_boot.py /tmp/patch/vllm_exact_cache_boot.py
COPY exact_cache.pth          /tmp/patch/exact_cache.pth
COPY sitecustomize.py         /tmp/patch/sitecustomize.py

# Install into site-packages. The .pth is the shadow-proof autoloader (site.py
# scans getsitepackages() dirs for *.pth); sitecustomize is only a fallback.
# The build FAILS if the .pth does not actually auto-run at interpreter startup,
# so a broken patch can never ship.
RUN set -e; \
    SP="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    echo "site-packages: $SP"; \
    cp /tmp/patch/vllm_exact_cache.py      "$SP/vllm_exact_cache.py"; \
    cp /tmp/patch/vllm_exact_cache_boot.py "$SP/vllm_exact_cache_boot.py"; \
    cp /tmp/patch/exact_cache.pth          "$SP/exact_cache.pth"; \
    cp /tmp/patch/sitecustomize.py         "$SP/sitecustomize.py"; \
    python3 -c "import vllm_exact_cache_boot; print('patch import OK')"; \
    python3 -c "pass" 2>&1 | grep -q "exact-cache] FastAPI autopatch installed" \
        || { echo "FATAL: .pth autoloader did not run at startup"; exit 1; }; \
    rm -rf /tmp/patch
