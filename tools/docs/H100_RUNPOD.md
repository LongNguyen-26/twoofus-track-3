# Hopper pod runbook (round 2, from 25/07/2026)

Replaces the RTX 4090 rig. Read `tools/runpod/mps_smcap.sh` for why.

## Why H100 SXM and not something else

| Option | $/hr | Verdict |
|---|---|---|
| **H100 SXM 80GB** | **2.99** | **Use this.** Hopper (native fp8 matmul like the H200 slice), 132 SMs — the same SM count as H200, so a 12% MPS cap models MiG 1g.18gb directly. 8 vCPU is plenty for a 3-core `taskset`. |
| H200 SXM | 4.39 | Closest to the real slice, but 1.5× the price buys only more memory bandwidth per SM — and we cap SMs anyway. Not worth losing an hour of runway on a $9.60 balance. |
| A100 SXM | 1.49 | Ampere: **no native fp8**. Cannot answer any question we have. |
| RTX PRO 6000 / 5090 | 0.99–1.99 | Blackwell/Ada consumer parts — different kernel selection from the H200 slice. This is the exact mistake the 4090 rig made. |

At $2.99/hr a $9.60 balance is **~3.2 hours**. The plan below uses ~1 hour;
the rest is reserve for the nightly follow-up.

## Deploy settings

- **GPU**: H100 SXM, count 1
- **Template**: vLLM Latest — image `vllm/vllm-openai:v0.25.1` (keep this; it is
  the control that every portal result was measured against)
- **Filter → CUDA Version ≥ 13.0** (the image fails container init on older
  host drivers with `unsatisfied condition: cuda>=13.0`)
- **Container Disk**: 60 GB (image ~12 GB + model 2.4 GB + vLLM source clone)
- **Volume Disk**: 20 GB mounted at `/workspace` — a network volume also works
  but constrains the datacenter, so prefer a plain volume unless H100 SXM is
  available where `ka4m1yetr4` lives
- **Container Start Command** — the image entrypoint *is* the server, so the pod
  needs a tiny keepalive to stay attachable. Paste exactly:

```
--model Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 --gpu-memory-utilization 0.05 --max-model-len 2048 --enforce-eager
```

The keepalive owns port 8000 as PID 1; every battery server runs on port 8001.

## Bring-up

```bash
cd /workspace && git clone https://github.com/LongNguyen-26/twoofus-track-3.git repo && cd repo && bash tools/runpod/pod_setup_h100.sh
```

That installs deps, downloads the model with `HF_HUB_DISABLE_XET=1`, starts the
MPS daemon and prints the capped-vs-uncapped verification.

**Read the two `SMCAP_VERIFY` lines before doing anything else.** `gemm_tflops`
must fall roughly 8×. If it does not, MPS is not in force and this pod can only
produce the same untrustworthy numbers the 4090 did — record that fact rather
than shipping conclusions from it.

### Measured 25/07 on H100 80GB HBM3 (cap confirmed working)

| | uncapped | capped 12% |
|---|---|---|
| SMs visible to torch | 132 | **14** |
| bf16 GEMM | 801 TFLOPS | **98** (8.2× ✓) |
| bulk 512MB read | 2936 GB/s | **871** (3.4×) |
| batch-1 GEMV 8192×2048 | 10.6 µs @ 3171 GB/s | **33.2 µs @ 1011 GB/s** |

Two conclusions:

1. **Batch-1 decode is bandwidth bound, not SM bound, at a MiG-sized SM budget.**
   The GEMV achieves *more* bandwidth (1011 GB/s) than the bulk reduction, so
   14 SMs are nowhere near saturated by a batch-1 projection. Halving weight
   bytes really does halve the weight-read term — the INT4 thesis survives its
   first gate. Whether Marlin's dequant eats the win is a separate question that
   only the real kernel probe answers.
2. **The rig still overstates bandwidth ~1.5×.** MPS partitions SMs but not the
   memory subsystem, while a real MiG 1g.18gb slice gets ~600 GB/s. So expect
   local TPOT below the portal's and a compressed bf16→fp8 gap. Judge by the
   ratio (portal H1/H0 = 5.81/4.07 = **1.43**), never the absolute ms.

`SM_PCT=12` yields 14 SMs; a real 1g slice is ~16, so this is slightly
conservative. `SM_PCT=13` gives ~17 if you want to bracket it.

## Experiments, in priority order

### 1. Nightly source probe — ~2 min, no GPU

```bash
bash tools/runpod/nightly_source_probe.sh
```

Greps vLLM `main` for the `All drafting layers should belong to the same kv
cache group` assertion that killed every draft-model route on v0.25.1.
Speculative decoding is the only lever with a ceiling above ~72 points, so this
cheap check outranks everything else.

### 2. INT4 kernel gate — ~5 min

```bash
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=12 python3 tools/runpod/int4_microbench.py --marlin | tee int4_micro.log
```

Answers whether batch-1 decode at 16 SMs is bandwidth bound (INT4 helps) or SM
bound (INT4 is dead regardless of kernel quality). The int4-floor column is a
version-proof upper bound, so a negative verdict here closes the track without
writing a single line of loader patch.

### 3. Calibration + cheap levers — ~45 min

```bash
bash tools/runpod/h_battery_r2.sh 2>&1 | tee h_battery.log
```

| id | config | role |
|---|---|---|
| H0 | fp8 = submit_020 | anchor — portal TPOT **4.07ms**, TTFT p50 42ms |
| H1 | bf16 = submit_018 | anchor — portal TPOT **5.81ms** |
| H2 | `cudagraph_mode=FULL_DECODE_ONLY` | untested mode (014 used FULL_AND_PIECEWISE) |
| H3 | `--async-scheduling` + `OMP_NUM_THREADS=1` | 3-core contention |
| H4 | fp8 rerun | local noise floor |

**The rig is only usable if H0/H1 reproduce the ~1.74ms portal gap.** H4−H0 sets
the minimum delta any later experiment must exceed to be believed.

### 4. Conditional: nightly draft-model spec decode — ~40 min

Only if step 1 says the assertion is gone. Redeploy the pod on the pinned
nightly image `vllm/vllm-openai:nightly-dd72658e7db0c6a674f473f6f9f0f5c2ebe7e523`
(same keepalive command), re-run `pod_setup_h100.sh`, then:

```bash
ONLY="D0 D10" bash tools/runpod/draft_battery_r2.sh
```

Gates before any portal slot: 6/6 greedy equivalence, needle no worse than the
matched control, and a repeatable TPOT win under the SM cap.

## Shutdown

Stop the pod as soon as the battery prints its summary — billing is per
millisecond and the logs are already on the volume under `results_h_*`.

```bash
tar czf /workspace/results_h.tgz /workspace/repo/results_h_* /workspace/repo/*.log
```
