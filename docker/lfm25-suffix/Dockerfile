# Reproducible Docker-build alternative to
# tools/runpod/pod_build_suffix_image.sh.
# The builder uses vLLM's existing PyTorch instead of Arctic's incompatible
# isolated-build pin (torch==2.7.0). The final image contains only the wheel.

FROM vllm/vllm-openai:v0.25.1 AS suffix-builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv --system-site-packages /tmp/arctic-build \
    && /tmp/arctic-build/bin/python -m pip install --no-cache-dir --upgrade \
       pip setuptools wheel ninja cmake nanobind==2.9.2 \
       protobuf==5.29.5 grpcio-tools \
    && ARCTIC_INFERENCE_ENABLED=0 CMAKE_BUILD_PARALLEL_LEVEL=3 \
       /tmp/arctic-build/bin/python -m pip wheel \
       --no-cache-dir --no-build-isolation --no-deps \
       --wheel-dir /tmp/wheels arctic-inference==0.1.1

FROM vllm/vllm-openai:v0.25.1

COPY --from=suffix-builder /tmp/wheels /tmp/arctic-wheels
RUN python3 -m pip install --no-cache-dir --no-deps \
       /tmp/arctic-wheels/arctic_inference-0.1.1-*.whl \
    && rm -rf /tmp/arctic-wheels \
    && ARCTIC_INFERENCE_ENABLED=0 python3 -c \
       "from arctic_inference.suffix_decoding import SuffixDecodingCache; SuffixDecodingCache(max_tree_depth=24,max_cached_requests=8); print('suffix import OK')"

# Keep Arctic's old vLLM patch plugin disabled. vLLM 0.25.1 uses its own
# built-in SuffixDecodingProposer and imports only the suffix-tree extension.
ENV ARCTIC_INFERENCE_ENABLED=0
