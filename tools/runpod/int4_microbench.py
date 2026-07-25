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


def linear_shapes(cfg):
    """Enumerate (name, out_features, in_features, count) for LFM2.5."""
    h = cfg["hidden_size"]
    inter = cfg.get("intermediate_size") or cfg.get("block_ff_dim") or 4 * h
    n_layers = cfg.get("num_hidden_layers", 16)
    n_heads = cfg.get("num_attention_heads", 32)
    n_kv = cfg.get("num_key_value_heads", 8)
    head_dim = cfg.get("head_dim") or h // n_heads
    # LFM2.5 = full-attention layers at fixed indices, ShortConv elsewhere.
    attn_idx = cfg.get("full_attn_idxs") or cfg.get("layer_types") or []
    if isinstance(attn_idx, list) and attn_idx and isinstance(attn_idx[0], str):
        n_attn = sum(1 for t in attn_idx if "attention" in t)
    elif isinstance(attn_idx, list) and attn_idx:
        n_attn = len(attn_idx)
    else:
        n_attn = 6
    n_conv = n_layers - n_attn
    return [
        ("mlp.gate_up", 2 * inter, h, n_layers),
        ("mlp.down", h, inter, n_layers),
        ("attn.qkv", (n_heads + 2 * n_kv) * head_dim, h, n_attn),
        ("attn.o", h, n_heads * head_dim, n_attn),
        ("conv.in_proj", 3 * h, h, n_conv),
        ("conv.out_proj", h, h, n_conv),
    ]


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


def bench_int4_readfloor(out_f, in_f, m):
    """Upper bound for ANY int4 kernel: time to merely stream the int4 bytes.

    Version-proof stand-in for Marlin. If this floor does not beat the measured
    fp8 GEMM, no packing scheme or kernel can rescue the track.
    """
    nbytes = out_f * in_f // 2
    buf = torch.ones(nbytes, dtype=torch.uint8, device="cuda")
    return bench(lambda: buf.sum())


def try_marlin(out_f, in_f, m):
    """Real W4A16 Marlin GEMM if this vLLM build exposes a usable helper."""
    try:
        from vllm.model_executor.layers.quantization.utils import marlin_utils_test as mut
        from vllm import _custom_ops as vops
        from vllm.scalar_type import scalar_types
    except Exception as e:
        return None, f"unavailable ({type(e).__name__})"
    try:
        w = torch.randn(in_f, out_f, dtype=torch.float16, device="cuda")
        quant = mut.marlin_quantize(w, scalar_types.uint4b8, 128, act_order=False)
        q_w, scales = quant[1], quant[2]
        workspace = mut.MarlinWorkspace(out_f, 64, 256) if hasattr(mut, "MarlinWorkspace") else None
        x = torch.randn(m, in_f, dtype=torch.float16, device="cuda")
        fn = lambda: vops.gptq_marlin_gemm(
            x, q_w, scales, None, None, workspace.scratch, scalar_types.uint4b8,
            m, out_f, in_f, True, False, False)
        fn()
        return bench(fn), "gptq_marlin_gemm"
    except Exception as e:
        return None, f"probe failed ({type(e).__name__}: {e})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/model")
    ap.add_argument("--marlin", action="store_true",
                    help="also probe the real Marlin W4A16 kernel")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model, "config.json")))
    shapes = linear_shapes(cfg)
    props = torch.cuda.get_device_properties(0)
    cap = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "UNCAPPED")
    print(f"# gpu={props.name} sm_count={props.multi_processor_count} sm_cap={cap}")
    print(f"# hidden={cfg['hidden_size']} layers={cfg.get('num_hidden_layers')}")
    print()

    params = sum(o * i * c for _, o, i, c in shapes)
    print(f"total linear params: {params / 1e9:.3f}B "
          f"(bf16 {params * 2 / 1e9:.2f}GB / fp8 {params / 1e9:.2f}GB / "
          f"int4 {params / 2e9:.2f}GB)")
    print()

    header = f"{'shape':<16}{'out':>7}{'in':>7}{'n':>4}{'m':>4}" \
             f"{'bf16 us':>10}{'fp8 us':>10}{'int4floor':>11}{'fp8 GB/s':>10}"
    print(header)
    print("-" * len(header))

    totals = {b: {"bf16": 0.0, "fp8": 0.0, "int4": 0.0} for b in BATCHES}
    fp8_impl = ""
    for name, out_f, in_f, count in shapes:
        if count == 0:
            continue
        for m in BATCHES:
            t_bf16 = bench_bf16(out_f, in_f, m)
            t_fp8, fp8_impl = bench_fp8(out_f, in_f, m)
            t_i4 = bench_int4_readfloor(out_f, in_f, m)
            totals[m]["bf16"] += t_bf16 * count
            totals[m]["fp8"] += t_fp8 * count
            totals[m]["int4"] += t_i4 * count
            gbs = (out_f * in_f) / t_fp8 / 1e9
            print(f"{name:<16}{out_f:>7}{in_f:>7}{count:>4}{m:>4}"
                  f"{t_bf16 * 1e6:>10.1f}{t_fp8 * 1e6:>10.1f}"
                  f"{t_i4 * 1e6:>11.1f}{gbs:>10.0f}")

    print()
    print(f"fp8 implementation: {fp8_impl}")
    print()
    print("=== per-decode-step totals over all linear layers (ms) ===")
    print(f"{'batch':>6}{'bf16':>10}{'fp8':>10}{'int4 floor':>12}{'fp8->int4':>12}")
    for m in BATCHES:
        t = totals[m]
        gain = (t["fp8"] - t["int4"]) * 1000
        print(f"{m:>6}{t['bf16'] * 1000:>10.2f}{t['fp8'] * 1000:>10.2f}"
              f"{t['int4'] * 1000:>12.2f}{gain:>12.2f}")

    if args.marlin:
        print()
        print("=== real Marlin W4A16 probe (gate: >=25% faster than fp8) ===")
        for name, out_f, in_f, count in shapes[:2]:
            for m in (1, 4):
                t_m, note = try_marlin(out_f, in_f, m)
                t_fp8, _ = bench_fp8(out_f, in_f, m)
                if t_m is None:
                    print(f"{name} m={m}: marlin {note}")
                else:
                    print(f"{name} m={m}: marlin {t_m * 1e6:.1f}us vs fp8 "
                          f"{t_fp8 * 1e6:.1f}us -> "
                          f"{(t_fp8 / t_m - 1) * 100:+.1f}%")

    b1 = totals[1]
    print()
    print("=== verdict ===")
    print(f"measured fp8 weight-read cost at batch 1: {b1['fp8'] * 1000:.2f}ms")
    print(f"perfect-int4 ceiling saving:              "
          f"{(b1['fp8'] - b1['int4']) * 1000:.2f}ms")
    print("Portal TPOT is 4.07ms. Subtract the saving above to get the best case")
    print("an ideal zero-tax INT4 kernel could reach, then score it:")
    print("  s_tpot = ((10 - TPOT)/9)^2 ; ERS ~ 0.988 * 0.5 * (0.8426 + s_tpot)")
    print("Proceed to a vLLM fork ONLY if that lands comfortably above 70 AND")
    print("the real Marlin probe clears +25%.")


if __name__ == "__main__":
    main()
