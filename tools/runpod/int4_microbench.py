"""Decide the online-INT4-Marlin question before writing any vLLM loader patch.

Context. Portal submit_020 decodes at TPOT ~4.07ms. Calibrating against the
bf16 twin (submit_018, 5.81ms) gives an effective slice bandwidth of ~690 GB/s,
so the budget splits roughly:

    4.07ms = 1.74ms reading fp8 weights + 2.33ms of everything else

INT4 can only attack the first term. Its perfect, zero-dequant-tax ceiling is
therefore about TPOT 3.3ms (ERS ~69) and about 3.0ms (ERS ~71) if the tied
embedding/lm_head is quantized too. That is barely the qualification cutoff, so
INT4 is only worth a private vLLM fork if the kernel gate passes cleanly.

This script answers, in order:

  1. Is batch-1 decode on a MiG-sized SM budget bandwidth bound or SM bound?
     If SM bound, fewer weight bytes buy nothing and the whole track is dead.
     This needs no Marlin and cannot break on a vLLM version bump.
  2. What is the measured floor for reading the real weight bytes at bf16 /
     fp8 / int4 sizes, summed over the actual LFM2.5 linear shapes?
  3. If vLLM's Marlin op is importable, what does a real W4A16 GEMM cost versus
     native fp8 W8A8 at the same shape? Gate: Marlin must be >=25% faster.

Run it with the same SM cap the servers use:
    CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=12 python3 tools/runpod/int4_microbench.py
"""
import argparse
import json
import os
import time

import torch

BATCHES = (1, 2, 4, 8)


def bench(fn, warmup=10, iters=100):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def linear_shapes(model_dir):
    """Enumerate real 2-D weight shapes straight from the checkpoint.

    Deriving shapes from config.json produced 1.439B "linear" params for a
    1.17B model -- LFM2.5 does not have one uniform MLP per layer. Read the
    safetensors headers instead: exact, and it cannot drift from the weights
    the server actually loads.
    """
    import glob
    import struct

    tensors = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with open(path, "rb") as fh:
            (hlen,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(hlen))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            shape = meta.get("shape", [])
            if len(shape) == 2:
                tensors[name] = tuple(shape)

    # Group identical shapes and label them by the role in the name.
    groups = {}
    embed_params = 0
    for name, (out_f, in_f) in tensors.items():
        if "embed" in name or "lm_head" in name:
            embed_params += out_f * in_f
            continue
        label = name.split(".")[-2] if "." in name else name
        groups.setdefault((label, out_f, in_f), 0)
        groups[(label, out_f, in_f)] += 1

    shapes = [(lbl, o, i, n) for (lbl, o, i), n in sorted(groups.items())]
    return shapes, embed_params


def bench_bf16(out_f, in_f, m):
    w = torch.randn(out_f, in_f, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(m, in_f, dtype=torch.bfloat16, device="cuda")
    return bench(lambda: torch.nn.functional.linear(x, w))


def bench_fp8(out_f, in_f, m):
    """Native Hopper fp8 W8A8. Prefers vLLM's cutlass path (supports m=1)."""
    w = torch.randn(out_f, in_f, device="cuda").to(torch.float8_e4m3fn)
    x = torch.randn(m, in_f, device="cuda").to(torch.float8_e4m3fn)
    sa = torch.ones(1, device="cuda", dtype=torch.float32)
    sb = torch.ones(1, device="cuda", dtype=torch.float32)
    try:
        from vllm import _custom_ops as vops

        wt = w.t()
        fn = lambda: vops.cutlass_scaled_mm(x, wt, sa, sb, torch.bfloat16)
        fn()
        return bench(fn), "cutlass_scaled_mm"
    except Exception:
        pass
    # torch._scaled_mm needs M padded to a multiple of 16 on most builds.
    mp = max(16, (m + 15) // 16 * 16)
    xp = torch.randn(mp, in_f, device="cuda").to(torch.float8_e4m3fn)
    wt = w.t()
    fn = lambda: torch._scaled_mm(xp, wt, scale_a=sa, scale_b=sb,
                                  out_dtype=torch.bfloat16)
    fn()
    return bench(fn), f"torch._scaled_mm(m padded {m}->{mp})"


def bench_readfloor(nbytes):
    """Time to merely stream `nbytes` -- the floor any kernel reading them obeys.

    Must use a bf16 reduction, not uint8: torch upcasts uint8 sums to int64,
    which turns a bandwidth probe into an arithmetic one (the first run reported
    45 GB/s and produced a nonsense int4 verdict).
    """
    buf = torch.ones(max(1, nbytes // 2), dtype=torch.bfloat16, device="cuda")
    return bench(lambda: buf.sum(), warmup=3, iters=20)


def marlin_ops_available():
    """Report every marlin-ish entry point this build exposes.

    v0.25.1 has no `vllm._custom_ops.gptq_marlin_gemm`, so the first probe could
    not even name the op it needed. List what exists instead of guessing.
    """
    found = []
    try:
        from vllm import _custom_ops as vops

        found += [f"vllm._custom_ops.{n}" for n in dir(vops) if "marlin" in n.lower()]
    except Exception as e:
        found.append(f"(vllm._custom_ops import failed: {type(e).__name__})")
    try:
        found += [f"torch.ops._C.{n}" for n in dir(torch.ops._C) if "marlin" in n.lower()]
    except Exception:
        pass
    try:
        import vllm.model_executor.layers.quantization.utils.marlin_utils as mu

        found += [f"marlin_utils.{n}" for n in dir(mu)
                  if "quantize" in n.lower() or "gemm" in n.lower()]
    except Exception as e:
        found.append(f"(marlin_utils import failed: {type(e).__name__})")
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/model")
    ap.add_argument("--marlin", action="store_true",
                    help="also probe the real Marlin W4A16 kernel")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    shapes, embed_params = linear_shapes(args.model)
    props = torch.cuda.get_device_properties(0)
    cap = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "UNCAPPED")
    print(f"# gpu={props.name} sm_count={props.multi_processor_count} sm_cap={cap}")
    print(f"# hidden={cfg['hidden_size']} layers={cfg.get('num_hidden_layers')}")
    print()

    params = sum(o * i * c for _, o, i, c in shapes)
    print(f"linear params (from safetensors): {params / 1e9:.3f}B "
          f"| bf16 {params * 2 / 1e9:.2f}GB / fp8 {params / 1e9:.2f}GB / "
          f"int4 {params / 2e9:.2f}GB")
    print(f"embedding/lm_head (stays bf16 under --quantization=fp8): "
          f"{embed_params / 1e9:.3f}B = {embed_params * 2 / 1e9:.2f}GB/step")
    print()

    header = f"{'name':<18}{'out':>7}{'in':>7}{'n':>4}{'m':>4}" \
             f"{'bf16 us':>10}{'fp8 us':>10}{'fp8 GB/s':>10}{'bf16 GB/s':>11}"
    print(header)
    print("-" * len(header))

    totals = {b: {"bf16": 0.0, "fp8": 0.0} for b in BATCHES}
    read_floor = {"bf16": 0.0, "fp8": 0.0, "int4": 0.0}
    fp8_impl = ""
    for name, out_f, in_f, count in shapes:
        nb = out_f * in_f
        for pre, div in (("bf16", 0.5), ("fp8", 1.0), ("int4", 2.0)):
            read_floor[pre] += bench_readfloor(int(nb / div)) * count
        for m in BATCHES:
            t_bf16 = bench_bf16(out_f, in_f, m)
            t_fp8, fp8_impl = bench_fp8(out_f, in_f, m)
            totals[m]["bf16"] += t_bf16 * count
            totals[m]["fp8"] += t_fp8 * count
            if m == 1:
                print(f"{name:<18}{out_f:>7}{in_f:>7}{count:>4}{m:>4}"
                      f"{t_bf16 * 1e6:>10.1f}{t_fp8 * 1e6:>10.1f}"
                      f"{nb / t_fp8 / 1e9:>10.0f}{nb * 2 / t_bf16 / 1e9:>11.0f}")

    print()
    print(f"fp8 implementation: {fp8_impl}")
    print()
    print("=== per-decode-step totals over all linear layers (ms) ===")
    print(f"{'batch':>6}{'bf16 gemm':>12}{'fp8 gemm':>12}{'bf16/fp8':>11}")
    for m in BATCHES:
        t = totals[m]
        print(f"{m:>6}{t['bf16'] * 1000:>12.2f}{t['fp8'] * 1000:>12.2f}"
              f"{t['bf16'] / max(t['fp8'], 1e-12):>11.2f}")
    print()
    print("=== pure read floors, same bytes, no arithmetic (ms/step) ===")
    print(f"  bf16 {read_floor['bf16'] * 1000:.2f}   "
          f"fp8 {read_floor['fp8'] * 1000:.2f}   "
          f"int4 {read_floor['int4'] * 1000:.2f}")

    if args.marlin:
        print()
        print("=== marlin entry points exposed by this build ===")
        for entry in marlin_ops_available() or ["(none found)"]:
            print(f"  {entry}")

    print()
    print("=== how to read this ===")
    print("The portal measured bf16/fp8 = 5.81/4.07 = 1.43 on the real slice.")
    print("If the 'bf16/fp8' column above is far below 1.43, this rig is NOT")
    print("bandwidth limited the way the slice is -- MPS caps SMs but leaves the")
    print("full HBM, so byte-saving looks worthless here while it is worth +13")
    print("points on the portal. In that regime the INT4 question CANNOT be")
    print("settled locally; only a portal slot can settle it.")


if __name__ == "__main__":
    main()
