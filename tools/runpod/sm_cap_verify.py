"""Verify that the CUDA MPS SM cap is actually in force.

The round-2 local rig (RTX 4090, 128 SMs) mispredicted every kernel-level
experiment (fp8 KV, BitsAndBytes W4) because it capped CPU cores and VRAM but
NOT streaming multiprocessors. The real BTC slice is a MiG 1g.18gb partition of
an H200: ~16 of 132 SMs. This script measures three throughputs so the battery
can prove the cap is real instead of assuming it:

  gemm   - compute bound  -> should fall ~8x when capped to ~12%
  bwread - bandwidth bound -> falls less; the ratio is the whole INT4 question
  gemv   - a real LFM2.5 MLP-shaped batch-1 projection

Run uncapped and capped and compare (tools/runpod/mps_smcap.sh does this).
Output is one machine-readable line prefixed SMCAP_VERIFY.
"""
import os
import time

import torch


def bench(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device")
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)

    # 1. compute-bound GEMM
    n = 4096
    a = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
    b = torch.randn(n, n, dtype=torch.bfloat16, device=dev)
    t_gemm = bench(lambda: torch.mm(a, b))
    tflops = (2 * n ** 3) / t_gemm / 1e12

    # 2. bandwidth-bound streaming read (512 MiB)
    nelem = 512 * 1024 * 1024 // 2
    x = torch.ones(nelem, dtype=torch.bfloat16, device=dev)
    t_bw = bench(lambda: x.sum(), warmup=3, iters=15)
    gbs = (nelem * 2) / t_bw / 1e9

    # 3. batch-1 GEMV at an LFM2.5 MLP shape (2048 -> 8192)
    w = torch.randn(8192, 2048, dtype=torch.bfloat16, device=dev)
    v = torch.randn(1, 2048, dtype=torch.bfloat16, device=dev)
    t_gemv = bench(lambda: torch.nn.functional.linear(v, w), iters=200)
    gemv_gbs = (8192 * 2048 * 2) / t_gemv / 1e9

    cap = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "unset")
    pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "unset")
    print(
        f"SMCAP_VERIFY gpu={props.name!r} sm_count={props.multi_processor_count} "
        f"cap_pct={cap} pipe={pipe} "
        f"gemm_tflops={tflops:.1f} bw_gbs={gbs:.0f} "
        f"gemv_us={t_gemv * 1e6:.1f} gemv_gbs={gemv_gbs:.0f}"
    )


if __name__ == "__main__":
    main()
