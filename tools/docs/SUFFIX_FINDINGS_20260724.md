# Suffix-decoding findings — 24/07/2026

## Verdict

**Rejected. Do not run S16, build/publish the suffix image, or spend a portal
submission on suffix decoding with vLLM 0.25.1 and LFM2.5-1.2B-Instruct.**

The final bracket was run on an RTX 4090 with three CPU cores and a 17.1 GB
VRAM cap. Both clean FP8 baselines were byte-identical and nearly drift-free,
so the suffix regression is causal rather than pod noise.

| Config | Fresh ERS | TTFT p50/p95 | TPOT p50 | Equivalence | Needle |
|---|---:|---:|---:|---:|---:|
| D0A | 70.20 | 61/125 ms | 2.49 ms | 6/6 | 2/5 |
| S4 | 65.45 | 65/108 ms | 3.02 ms | 0/6 | 1/5 |
| S8 | 65.49 | 65/116 ms | 3.04 ms | 0/6 | 1/5 |
| D0B | 70.08 | 61/125 ms | 2.49 ms | 6/6 | 2/5 |

FP8 bracket drift was only **−0.12 points**, with identical 2.49 ms TPOT and
61/125 ms TTFT percentiles.

## Why it loses

Prometheus counters:

| Config | Draft proposals | Draft tokens | Accepted tokens | Acceptance |
|---|---:|---:|---:|---:|
| S4 | 53,410 | 114,727 | 26,525 | 23.1% |
| S8 | 53,836 | 131,012 | 27,158 | 20.7% |

S4 accepts `26,525 / 53,410 = 0.497` speculative tokens per proposal; S8
accepts `27,158 / 53,836 = 0.504`. Raising the maximum from 4 to 8 therefore
adds almost no useful accepted work while increasing draft and verification
overhead. TPOT becomes 21–22% slower and ERS falls by approximately 4.7
points.

S4 and S8 both produced `EQUIVALENCE 0/6`, including corrupted sequence and
JSON answers, while D0B reproduced D0A at 6/6 after a full restart. This is a
hybrid-target correctness incompatibility, not ordinary GPU nondeterminism or
an acceptable quantization difference. Needle retrieval also regressed from
2/5 to 1/5.

The apparently lower S4 TTFT p95 is not a win: p50 and mean worsened, a
roughly 900 ms outlier appeared, TPOT regressed sharply, and the scoring
output fell.

## Build lesson

`arctic-inference==0.1.1` can be built against the PyTorch already shipped in
`vllm/vllm-openai:v0.25.1` using `--no-build-isolation` and
`ARCTIC_INFERENCE_ENABLED=0`.

The first automated build failed at `running build_ext` with:

```text
No such file or directory: 'cmake'
```

The Python `cmake` wheel was present in the build venv, but its `bin`
directory was not on `PATH` for the setuptools subprocess. The build helper
now exports the venv `bin` directory and verifies `cmake --version` before
building. Recreating the pod would not have fixed the old script.

## Archived reproduction

The scripts remain for reproducibility:

```bash
bash tools/runpod/pod_install_suffix.sh
MODES=fresh bash tools/runpod/suffix_battery_r2.sh
```

Detailed result directory from the live pod:

```text
/workspace/repo/results_suffix_20260724_113728
```

The next active track is official vLLM in-flight BitsAndBytes W4:

```bash
bash tools/runpod/pod_install_bitsandbytes.sh
bash tools/runpod/w4_battery_r2.sh
```

See `tools/docs/W4_RUNPOD.md`.
