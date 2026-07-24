# Suffix decoding: RunPod test and image workflow

This workflow tests suffix decoding first in a disposable container, then
publishes the exact tested extension as one layer on
`vllm/vllm-openai:v0.25.1`. No Docker daemon is required.

## 1. Deploy the test pod

Use:

- Image: `vllm/vllm-openai:v0.25.1`
- CUDA filter: `>= 13.0`
- GPU: RTX 4090 is sufficient for ordering; H100/H200 is preferred for the
  final confirmation because FP8 behavior transfers better.
- Network volume mounted at `/workspace`.
- Container start command:

```text
--model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.1 --max-model-len 2048
```

The small server is only a keepalive. The battery launches the real model on
port 8001, restricted to three CPU cores and approximately 17.1 GB VRAM.

## 2. Prepare repository, model, and suffix extension

In the RunPod web terminal:

```bash
cd /workspace
git clone https://github.com/LongNguyen-26/twoofus-track-3.git repo
cd /workspace/repo

python3 -m pip install --no-cache-dir huggingface_hub aiohttp
export HF_HUB_DISABLE_XET=1
hf download LiquidAI/LFM2.5-1.2B-Instruct \
  --local-dir /workspace/model

bash tools/runpod/pod_install_suffix.sh
```

If the repository already exists:

```bash
cd /workspace/repo
git pull --ff-only
bash tools/runpod/pod_install_suffix.sh
```

`tools/runpod/pod_install_suffix.sh` builds Arctic 0.1.1 with
`--no-build-isolation`.
This is deliberate: Arctic's published build metadata pins PyTorch 2.7.0,
while vLLM 0.25.1 already carries the matching newer PyTorch runtime.

## 3. Run the battery

Full battery, approximately one hour plus server startup time:

```bash
cd /workspace/repo
bash tools/runpod/suffix_battery_r2.sh
```

Default order:

```text
D0A -> S4 -> S8 -> S16 -> D0B
```

Each config runs both `fresh` and `shared` 420-request replays, exact greedy
output comparison, the 28k-token needle check, and saves suffix acceptance
counters. At the end, `summary.txt` reports points/TPOT relative to the mean
of D0A and D0B, correctness, acceptance rate, and baseline drift.

For a quick functional smoke test first:

```bash
ONLY="D0A S8" MODES="fresh" \
  bash tools/runpod/suffix_battery_r2.sh
```

If D0A already exists in a previous output directory, do not reuse its
equivalence file across code/model changes. Run a new D0A.

Candidate gate:

- `EQUIVALENCE 6/6`
- Needle result no worse than D0A; never submit a 0/5 result
- No startup or HTTP errors
- Improvement appears in both prompt regimes
- Target at least 15% TPOT reduction, ideally TPOT <= 3.2 ms in the
  portal-calibrated regime
- D0A and D0B reveal how much host drift occurred during the battery

## 4. Publish the custom image

Create a public Docker Hub repository, then authenticate with an access token:

```bash
crane auth login index.docker.io -u DOCKERHUB_USER
```

Build and append the tested wheel:

```bash
cd /workspace/repo
TAG=DOCKERHUB_USER/lfm25-suffix:v1 \
  bash tools/runpod/pod_build_suffix_image.sh
```

The script pins the base by digest before appending the layer. Make the Docker
Hub repository public after the push.

For environments with Docker BuildKit, the reproducible alternative is:

```bash
docker build \
  -t DOCKERHUB_USER/lfm25-suffix:v1 \
  -f docker/lfm25-suffix/Dockerfile .
docker push DOCKERHUB_USER/lfm25-suffix:v1
```

## 5. Validate the published image

Deploy a new pod using `DOCKERHUB_USER/lfm25-suffix:v1` with the same
keepalive command, clone/pull the repository, download or reuse the model, and
run:

```bash
cd /workspace/repo
ONLY="D0A S8" MODES="fresh" \
  bash tools/runpod/suffix_battery_r2.sh
```

Replace `S8` with the winning limit. This catches a missing layer or incorrect
public image before using a portal quota.

## 6. Prepare the portal compose

Copy `docker/lfm25-suffix/docker-compose.example.yml` into the next submission
directory. Replace:

- `DOCKERHUB_USER/lfm25-suffix:v1` with the exact public tag
- `num_speculative_tokens` with the S4/S8/S16 winner

Do not reduce `max-model-len`, disable prefix caching, add FP8 KV cache, or
change the first four BTC-controlled arguments.
