# Semantic-cache image for Viettel AI Race round 1.
# Base = the proven baseline runtime; we only add an exact-match response cache
# at the FastAPI/ASGI layer. The enforced entrypoint module
# `vllm.entrypoints.openai.api_server` is untouched and still serves real vLLM.
#
# Build (on the Windows machine — RunPod pods have no docker daemon):
#   docker build -t <dockerhub_user>/qwen35-cache:v1 image/
#   docker push <dockerhub_user>/qwen35-cache:v1     (repo must be PUBLIC)
FROM vllm/vllm-openai:v0.22.1

COPY vllm_exact_cache.py /tmp/patch/vllm_exact_cache.py
COPY sitecustomize.py    /tmp/patch/sitecustomize.py

RUN set -e; \
    SP="$(python3 -c 'import site; print(site.getsitepackages()[0])')"; \
    echo "site-packages: $SP"; \
    cp /tmp/patch/vllm_exact_cache.py "$SP/vllm_exact_cache.py"; \
    if [ -f "$SP/sitecustomize.py" ]; then \
        cat /tmp/patch/sitecustomize.py >> "$SP/sitecustomize.py"; \
    else \
        cp /tmp/patch/sitecustomize.py "$SP/sitecustomize.py"; \
    fi; \
    python3 -c "import vllm_exact_cache; print('patch import OK')"; \
    rm -rf /tmp/patch
