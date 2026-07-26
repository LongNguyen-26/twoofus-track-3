#!/usr/bin/env python3
"""Where does a decode step actually spend its time?

The portal-calibrated budget for LFM2.5 on the MiG slice is

    TPOT 4.07ms = 1.73 (fp8 linear weights) + 0.45 (BF16 head) + ~1.9 (???)

The first two terms are byte counts confirmed against the portal to 1%. The
third has never been measured -- it is what is left after subtraction, described
only as "roughly 150 small kernels". It is now the largest single term and the
only place a lever bigger than +2 points could still be hiding. This script
enumerates it.

Method
------
Two profiled runs over the same ~4k-token prompt:

    A: max_tokens=1     -> one prefill forward, no decode steps
    B: max_tokens=N+1   -> one prefill forward, N decode steps

Per-kernel CUDA totals are subtracted (B - A) and divided by N, which cancels
prefill exactly and leaves the per-decode-step cost of every kernel. A separate
unprofiled pass measures clean wall-clock timing, because the profiler itself
adds overhead and the attribution must be read against a known scale factor.

Runs in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the profiler sees the
model runner directly rather than across the EngineCore process boundary.

Read the output as PROPORTIONS, not absolutes. The rig overstates bandwidth
about 1.4x and its TPOT runs about 1.9x the portal's, so a kernel that is 30% of
the step here is the thing to attack, but its millisecond figure is not the
slice's.
"""

import argparse
import os
import time

# Must precede the vllm import: keeps the engine in this process so the
# profiler can see the model forward.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

CATEGORIES = (
    # (label, substrings) -- checked in order, first match wins
    ("fp8/GEMM (weights)", ("cutlass", "scaled_mm", "gemm", "nvjet", "s16816",
                            "wgmma", "tensorop", "gemv")),
    ("attention", ("flash", "attn", "paged", "mha", "fmha")),
    ("shortconv / state", ("conv", "causal", "short", "mamba", "ssm")),
    ("rmsnorm", ("rms", "layernorm", "layer_norm")),
    ("rope / positional", ("rope", "rotary")),
    ("activation / elementwise", ("silu", "swiglu", "gelu", "elementwise",
                                  "vectorized_elementwise", "mul", "add")),
    ("embedding / logits gather", ("embedding", "index_select", "gather",
                                   "indexselect")),
    ("sampling", ("argmax", "softmax", "topk", "top_k", "sort", "multinomial",
                  "cumsum", "randperm")),
    ("quantize (act fp8)", ("quant", "fp8_quant", "scaled_fp8")),
    ("copy / memset", ("memcpy", "memset", "copy_", "cat", "clone", "pad")),
)


def classify(name):
    low = name.lower()
    for label, keys in CATEGORIES:
        if any(k in low for k in keys):
            return label
    return "unclassified"


def device_time(evt):
    """torch renamed self_cuda_time_total -> self_device_time_total."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        val = getattr(evt, attr, None)
        if val is not None:
            return float(val)
    return 0.0


def cpu_time(evt):
    return float(getattr(evt, "self_cpu_time_total", 0.0) or 0.0)


def kernel_totals(prof):
    """{kernel name: (cuda_us, count)} for device-side work only."""
    out = {}
    for evt in prof.key_averages():
        dt = device_time(evt)
        if dt <= 0:
            continue
        name = evt.key
        cur = out.get(name, (0.0, 0))
        out[name] = (cur[0] + dt, cur[1] + int(getattr(evt, "count", 0) or 0))
    return out


def host_total(prof):
    return sum(cpu_time(e) for e in prof.key_averages())


def make_prompt(llm, n_tokens):
    """Real token ids, so tokenizer and embedding behave normally."""
    tok = llm.get_tokenizer()
    base = tok.encode(
        "The quick brown fox jumps over the lazy dog while the committee "
        "reviews the quarterly inference latency report in detail. "
    )
    ids = (base * (n_tokens // max(1, len(base)) + 2))[:n_tokens]
    return {"prompt_token_ids": ids}


def generate(llm, prompt, max_tokens, sampling_cls):
    sp = sampling_cls(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    return llm.generate(prompts=[prompt], sampling_params=sp, use_tqdm=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/workspace/model")
    ap.add_argument("--context", type=int, default=4000,
                    help="prompt tokens; the graded trace is ~4000")
    ap.add_argument("--steps", type=int, default=200,
                    help="decode steps to profile; the trace pins 200")
    ap.add_argument("--quantization", default="fp8")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.20)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--eager", action="store_true",
                    help="disable CUDA graphs; noisier but attributes per-op")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    print("[profile] loading %s (quantization=%s, eager=%s)"
          % (args.model, args.quantization, args.eager))
    llm = LLM(
        model=args.model,
        quantization=args.quantization or None,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        enforce_eager=args.eager,
        tensor_parallel_size=1,
    )
    prompt = make_prompt(llm, args.context)

    # Warm up: triggers torch.compile and CUDA graph capture, which would
    # otherwise land inside the first profiled run and dominate it.
    print("[profile] warmup")
    generate(llm, prompt, 8, SamplingParams)
    generate(llm, prompt, args.steps + 1, SamplingParams)
    torch.cuda.synchronize()

    # ---- clean wall clock, no profiler ----------------------------------
    t0 = time.perf_counter()
    generate(llm, prompt, 1, SamplingParams)
    torch.cuda.synchronize()
    t_prefill = time.perf_counter() - t0

    t0 = time.perf_counter()
    generate(llm, prompt, args.steps + 1, SamplingParams)
    torch.cuda.synchronize()
    t_full = time.perf_counter() - t0

    clean_step_ms = (t_full - t_prefill) * 1000.0 / args.steps
    print("\n[clean, no profiler]  prefill %.1f ms   per decode step %.3f ms"
          % (t_prefill * 1000.0, clean_step_ms))

    # ---- profiled A (prefill only) and B (prefill + N decode) -----------
    acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    print("[profile] run A: prefill only")
    with profile(activities=acts) as prof_a:
        generate(llm, prompt, 1, SamplingParams)
        torch.cuda.synchronize()
    print("[profile] run B: prefill + %d decode steps" % args.steps)
    with profile(activities=acts) as prof_b:
        generate(llm, prompt, args.steps + 1, SamplingParams)
        torch.cuda.synchronize()

    a, b = kernel_totals(prof_a), kernel_totals(prof_b)
    per_step = {}
    for name, (us_b, cnt_b) in b.items():
        us_a, cnt_a = a.get(name, (0.0, 0))
        delta_us = us_b - us_a
        delta_cnt = cnt_b - cnt_a
        if delta_us <= 0:
            continue  # prefill-only kernel, or noise
        per_step[name] = (delta_us / args.steps, delta_cnt / args.steps)

    total_us = sum(v[0] for v in per_step.values())
    host_ms = (host_total(prof_b) - host_total(prof_a)) / args.steps / 1000.0

    # ---- report ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("PER DECODE STEP, GPU kernel time attributed  (context %d, %d steps)"
          % (args.context, args.steps))
    print("=" * 78)

    buckets = {}
    for name, (us, cnt) in per_step.items():
        label = classify(name)
        cur = buckets.get(label, [0.0, 0.0])
        buckets[label] = [cur[0] + us, cur[1] + cnt]
    print("\n%-28s %10s %8s %8s" % ("category", "us/step", "share", "kernels"))
    print("-" * 78)
    for label, (us, cnt) in sorted(buckets.items(), key=lambda kv: -kv[1][0]):
        print("%-28s %10.1f %7.1f%% %8.1f"
              % (label, us, 100.0 * us / max(total_us, 1e-9), cnt))
    print("-" * 78)
    print("%-28s %10.1f %7.1f%% %8.1f"
          % ("TOTAL GPU", total_us, 100.0,
             sum(v[1] for v in per_step.values())))

    print("\ntop %d kernels" % args.top)
    print("-" * 78)
    print("%-52s %10s %8s" % ("kernel", "us/step", "n/step"))
    for name, (us, cnt) in sorted(per_step.items(), key=lambda kv: -kv[1][0])[: args.top]:
        print("%-52s %10.2f %8.1f" % (name[:52], us, cnt))

    print("\n" + "=" * 78)
    print("scale check -- attribution is only meaningful against these")
    print("=" * 78)
    print("  GPU kernel time attributed : %7.3f ms/step" % (total_us / 1000.0))
    print("  host (CPU) time            : %7.3f ms/step" % host_ms)
    print("  clean wall clock           : %7.3f ms/step" % clean_step_ms)
    print("""
  Async scheduling overlaps host with GPU, so host > 0 is not additive; what
  matters is whether host EXCEEDS the GPU total, which would mean the step is
  CPU bound and no kernel work can fix it.

  Profiler overhead inflates the wall clock; compare the GPU total against the
  clean figure, not against the profiled one.

  This rig runs ~1.9x the portal's TPOT and overstates bandwidth ~1.4x. Divide
  to estimate the slice, but act on the SHARES -- they are what transfers.""")


if __name__ == "__main__":
    main()
