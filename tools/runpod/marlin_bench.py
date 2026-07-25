"""The INT4 gate: does Marlin reach memory speed on the real LFM2.5 shapes?

The cost model is already closed. At 600 GB/s the slice's 1.036B linear params
cost 3.45ms in bf16 and 1.73ms in fp8, and the portal's submit_018 - submit_020
gap is 1.74ms -- a 1% match, so linears run at exactly memory speed there with
no residual kernel tax. INT4 would cost 0.87ms, which projects to ERS 69.7
(linears only) or 72.6 (including the tied embedding). The one unknown left is
whether a Marlin W4 kernel actually achieves that bandwidth instead of paying a
dequant tax, and that is what this script measures.

Read it as a CONSERVATIVE gate. On the hogged pod the cutlass fp8 path only
reaches ~291 GB/s versus the slice's ~600, so the rig handicaps SM-hungry
kernels; INT4 needs more SM work per byte than fp8, so it is handicapped harder
here than it would be on the portal. A Marlin win on this rig therefore implies
a win on the slice. A narrow loss here is inconclusive.

Two variants are benchmarked, both from the ops this v0.25.1 build exposes:
  W4A16 - int4 weights, bf16 activations
  W4A8  - int4 weights, fp8 activations (marlin_int4_fp8_preprocess +
          gptq_marlin_repack(is_a_8bit=True)); native fp8 on Hopper, so this is
          the variant most likely to clear the gate

Run under the tuned rig:
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=60 HOG_GB=4 python3 tools/runpod/bw_hog.py &
  CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=12 python3 tools/runpod/marlin_bench.py
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from int4_microbench import bench, bench_bf16, bench_fp8, linear_shapes  # noqa: E402

GROUP = 128
BATCHES = (1, 2, 4, 8)


def quantize_gptq(w, group=GROUP):
    """Symmetric groupwise int4 (uint4b8) packing of a (K, N) weight.

    Calibration-free round-to-nearest, exactly what an online startup-time
    quantizer would do -- BTC forbids offline/prepacked target weights, so the
    benchmark must reflect the legal path.
    """
    K, N = w.shape
    wf = w.float().reshape(K // group, group, N)
    scale = wf.abs().amax(dim=1, keepdim=True) / 7.0
    scale = torch.clamp(scale, min=1e-8)
    q = torch.clamp(torch.round(wf / scale), -8, 7).to(torch.int32) + 8
    q = q.reshape(K, N)

    pack = 32 // 4
    qweight = torch.zeros((K // pack, N), dtype=torch.int32, device=w.device)
    qb = q.reshape(K // pack, pack, N)
    for i in range(pack):
        qweight |= (qb[:, i, :] & 0xF) << (4 * i)
    return qweight, scale.reshape(K // group, N).to(w.dtype)


def make_marlin(w, dtype, a_8bit):
    """Repack a (K, N) weight into Marlin layout. Returns (b_q, s, workspace)."""
    from vllm import _custom_ops as vops
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu

    K, N = w.shape
    dev = w.device
    qweight, scales = quantize_gptq(w)
    perm = torch.empty(0, dtype=torch.int, device=dev)
    b_q = vops.gptq_marlin_repack(qweight, perm, K, N, 4, a_8bit)
    if a_8bit:
        b_q = vops.marlin_int4_fp8_preprocess(b_q, None, False)
    s = mu.marlin_permute_scales(scales.to(dtype), K, N, GROUP)
    try:
        ws = mu.marlin_make_workspace_new(dev)
    except TypeError:
        ws = torch.zeros(N // 64 * 16, dtype=torch.int, device=dev)
    return b_q, s


def marlin_call(a, b_q, s, a_scales, N, K, type_id, m):
    from vllm import _custom_ops as vops
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu

    try:
        ws = mu.marlin_make_workspace_new(a.device)
    except TypeError:
        ws = torch.zeros(N // 64 * 16, dtype=torch.int, device=a.device)
    return lambda: torch.ops._C.marlin_gemm(
        a, None, b_q, None, s, a_scales, None, None, None, None, ws,
        type_id, m, N, K, True, False, True, False)


def bench_marlin(out_f, in_f, m, a_8bit, dtype=torch.bfloat16):
    from vllm.scalar_type import scalar_types

    dev = torch.device("cuda")
    K, N = in_f, out_f                      # GPTQ stores (K, N)
    w = torch.randn(K, N, dtype=dtype, device=dev) * 0.02
    b_q, s = make_marlin(w, dtype, a_8bit)
    type_id = scalar_types.uint4b8.id

    if a_8bit:
        a16 = torch.randn(m, K, dtype=dtype, device=dev)
        a = a16.to(torch.float8_e4m3fn)
        a_scales = torch.ones(1, dtype=torch.float32, device=dev)
    else:
        a16 = torch.randn(m, K, dtype=dtype, device=dev)
        a = a16
        a_scales = None

    fn = marlin_call(a, b_q, s, a_scales, N, K, type_id, m)
    out = fn()

    # A fast wrong kernel is worthless: check against the dequantized reference.
    ref = a16.float() @ w.float()
    err = (out.float() - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
    return bench(fn), err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/model")
    args = ap.parse_args()

    shapes, embed_params = linear_shapes(args.model)
    props = torch.cuda.get_device_properties(0)
    cap = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "UNCAPPED")
    print(f"# gpu={props.name} sm_count={props.multi_processor_count} sm_cap={cap}")
    try:
        from vllm.model_executor.layers.quantization.utils import marlin_utils as mu

        print(f"# supported marlin types: {mu.query_marlin_supported_quant_types(False)}")
    except Exception as e:
        print(f"# query_marlin_supported_quant_types failed: {e}")
    print()

    hdr = (f"{'name':<12}{'out':>7}{'in':>7}{'n':>4}{'m':>4}"
           f"{'fp8 us':>9}{'W4A16 us':>10}{'W4A8 us':>9}{'W4A8 GB/s':>11}{'relerr':>10}")
    print(hdr)
    print("-" * len(hdr))

    totals = {b: {"fp8": 0.0, "w4a16": 0.0, "w4a8": 0.0} for b in BATCHES}
    failures = []
    for name, out_f, in_f, count in shapes:
        for m in BATCHES:
            t_fp8, _ = bench_fp8(out_f, in_f, m)
            totals[m]["fp8"] += t_fp8 * count
            row = {}
            for key, a8 in (("w4a16", False), ("w4a8", True)):
                try:
                    t, err = bench_marlin(out_f, in_f, m, a8)
                    totals[m][key] += t * count
                    row[key] = (t, err)
                except Exception as e:
                    failures.append(f"{name} m={m} {key}: {type(e).__name__}: {e}")
                    totals[m][key] = float("nan")
                    row[key] = None
            if m == 1:
                a16 = row.get("w4a16")
                a8 = row.get("w4a8")
                nb = out_f * in_f / 2
                print(f"{name:<12}{out_f:>7}{in_f:>7}{count:>4}{m:>4}"
                      f"{t_fp8 * 1e6:>9.1f}"
                      f"{(a16[0] * 1e6 if a16 else float('nan')):>10.1f}"
                      f"{(a8[0] * 1e6 if a8 else float('nan')):>9.1f}"
                      f"{(nb / a8[0] / 1e9 if a8 else float('nan')):>11.0f}"
                      f"{(a8[1] if a8 else float('nan')):>10.4f}")

    if failures:
        print()
        print("=== failures ===")
        for f in failures[:12]:
            print(f"  {f}")

    print()
    print("=== per-decode-step totals over all linear layers (ms) ===")
    print(f"{'batch':>6}{'fp8':>10}{'W4A16':>10}{'W4A8':>10}"
          f"{'fp8/W4A16':>12}{'fp8/W4A8':>11}")
    for m in BATCHES:
        t = totals[m]
        r16 = t["fp8"] / t["w4a16"] if t["w4a16"] else float("nan")
        r8 = t["fp8"] / t["w4a8"] if t["w4a8"] else float("nan")
        print(f"{m:>6}{t['fp8'] * 1000:>10.2f}{t['w4a16'] * 1000:>10.2f}"
              f"{t['w4a8'] * 1000:>10.2f}{r16:>12.2f}{r8:>11.2f}")

    print()
    print("=== verdict ===")
    print("Gate: the best W4 variant must be >=25% faster than fp8 (ratio >=1.25)")
    print("at batch 1-4 AND relerr must stay small (a wrong kernel is worthless).")
    print("Because this rig handicaps SM-hungry kernels, a clear win here implies")
    print("a win on the portal, where INT4 linears project to ERS 69.7 and INT4")
    print("including the tied embedding projects to 72.6.")


if __name__ == "__main__":
    main()
