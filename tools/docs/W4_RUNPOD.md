# In-flight W4 BitsAndBytes battery on RunPod

This is the current round-2 candidate workflow after suffix decoding failed
its correctness and latency gates. It tests official vLLM in-flight
BitsAndBytes 4-bit weight quantization directly from the mounted BF16 model.
It does not package or alter a target checkpoint.

## 1. Reuse the current pod

A new pod is not required. The suffix installation can remain in the
container: the W4 battery exports `ARCTIC_INFERENCE_ENABLED=0`, and the clean
FP8 baseline already proved that the installed extension does not change
normal vLLM execution.

Pull the new scripts:

```bash
cd /workspace/repo
git pull --ff-only
```

The expected environment remains:

- Base image: `vllm/vllm-openai:v0.25.1`
- Model: `/workspace/model`
- RTX 4090 or faster GPU
- Three simulated CPU cores (`0-2`)
- Approximately 17.1 GB usable VRAM

## 2. Install or verify BitsAndBytes

```bash
cd /workspace/repo
bash tools/runpod/pod_install_bitsandbytes.sh
```

The helper first reuses an importable installation. If none exists, it
installs `bitsandbytes>=0.46.1` with `--no-deps`, so it cannot replace the
image's PyTorch, CUDA, or vLLM packages. The exact installed versions are
saved by the battery in `environment.txt`.

To test a pinned wheel explicitly:

```bash
BNB_SPEC='bitsandbytes==VERSION' FORCE_REINSTALL=1 \
  bash tools/runpod/pod_install_bitsandbytes.sh
```

Do not change the package version during one bracket.

## 3. Run the full bracket

```bash
cd /workspace/repo
bash tools/runpod/w4_battery_r2.sh
```

Default order:

```text
FP8A -> BNB4A -> BNB4B -> FP8B
```

Each server is started independently. Every run performs:

- startup timing against the 600-second BTC health gate;
- greedy-output capture/comparison;
- the current all-420 `fresh` replay;
- the approximately 28k-token needle test;
- GPU-memory and environment capture.

`BNB4A` writes a W4 reference and `BNB4B` compares against it across a full
server restart. Comparison between BNB4 and FP8 is also recorded, but it is
informational only: legitimate weight quantization may change greedy output.

The full bracket normally takes about 35–45 minutes on the existing 4090 pod.
Results are written to:

```text
/workspace/repo/results_w4_YYYYMMDD_HHMMSS/
```

The final decision is in `summary.txt`.

## 4. Resume an interrupted bracket

Reuse the exact output directory so the equivalence references and previous
logs remain available:

```bash
OUT=/workspace/repo/results_w4_YYYYMMDD_HHMMSS \
ONLY="BNB4B FP8B" \
  bash tools/runpod/w4_battery_r2.sh
```

Do not reuse references after changing the model, BitsAndBytes version, vLLM
image, or battery code. Start a new output directory in those cases.

For startup-only diagnosis, run the first pair, but it is not enough for a
candidate decision:

```bash
ONLY="FP8A BNB4A" \
  bash tools/runpod/w4_battery_r2.sh
```

## 5. Candidate gates

The summarizer requires all of the following:

- both BNB4 runs start in less than 600 seconds;
- zero replay errors;
- BNB4 is deterministic across restart;
- needle retrieval is not worse than the bracketed FP8 baselines;
- both BNB4 runs improve fresh TPOT by at least 15%;
- both BNB4 runs improve synthetic ERS over the FP8 bracket mean.

The current local FP8 TPOT is approximately 2.49 ms, so the minimum useful
BNB4 target is **2.12 ms**; **1.9–2.0 ms** is preferable. Small gains are not
portal-worthy because dequantization costs transfer poorly from a 4090 to the
CPU-starved H200 slice.

## 6. Accuracy check only after a local pass

W4 is not expected to be byte-identical to FP8. A local pass therefore moves
to BF16-vs-BNB4 GPQA:

```bash
ONLY="G1 G2" \
  bash tools/evaluation/gpqa_r2.sh
```

`G1` is BF16 and `G2` is in-flight BitsAndBytes W4. Require
`accuracy_bf16 - accuracy_bnb4 <= 0.10` before preparing a custom image or
using portal quota.

If the W4 battery reports `LOCAL REJECT`, stop the track. Do not compensate by
combining W4 with scheduler, KV-FP8, suffix, or speculative-decoding flags.
