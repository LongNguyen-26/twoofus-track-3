# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Competition workspace for **Viettel AI Race 2026 — LLM Inference Optimization Challenge**, Phase 1 (Vòng 1 Sơ loại, 02/07–30/07/2026). There is no application code, build, lint, or test suite. The deliverable is a `docker-compose.yml` submitted to the organizer's (BTC) portal; it must reference a **public Docker Hub image** that serves **Qwen/Qwen3.5-2B** as an OpenAI-compatible endpoint. BTC pulls the image, runs it on their hardware, and benchmarks it against a fixed trace.

Communicate with the user in Vietnamese (English technical terms are fine).

## Evaluation environment (fixed by BTC)

- 1× MiG H200 instance: **18GB VRAM, 3 CPU cores, 8GB RAM**, Ubuntu 22.04, CUDA 12.x
- Model weights are provided by BTC (fixed HF hash) mounted at `/model`; served model name must be `Qwen3.5-2B` on port 8000
- Baseline image: `vllm/vllm-openai:v0.22.1`
- Healthcheck wait: 600s — server startup (model load + any on-the-fly quantization/warmup) must fit inside this window
- Submission limit: **5/day**; results + logs return in ~15 min

In the sample compose, the `entrypoint` lines and the first four `command` args (`--model=/model`, `--served-model-name=Qwen3.5-2B`, `--host=0.0.0.0`, `--port=8000`) are marked `#Don't change this to vllm-server` — keep them verbatim when using the vLLM image; tune everything after them. Rules allow any framework (vLLM, SGLang, TensorRT-LLM, custom), but how the "don't change" constraint applies outside vLLM is unverified — check the forum before switching.

## Scoring

`Score = 100 × ERS × f(Δ)`

- **ERS** = mean over 120 requests of `0.5·s_ttft + 0.5·s_tpot`; a request that errors, times out, or returns 0 tokens scores **0**.
  - `s_ttft = clamp((1500 − TTFT_ms) / 1400, 0, 1)²` — full marks at ≤100ms, zero at ≥1500ms
  - `s_tpot = clamp((45 − TPOT_mean_ms) / 25, 0, 1)²` — full marks at ≤20ms, zero at ≥45ms
  - γ=2 makes the curve quadratic: latency near the ceiling is punished hard.
- **Accuracy gate f(Δ)**: 100 fixed GPQA Diamond questions vs BF16 reference accuracy 0.4. `Δ = 0.4 − team_accuracy`. `f = 1.0` if Δ ≤ 0.10 (i.e. GPQA ≥ 0.30 costs nothing), linear from 1→0 over 0.10 < Δ < 0.16, `f = 0` if Δ ≥ 0.16. Aggressive quantization is viable but a GPQA drop below 0.24 zeroes the entire score.

## The trace (input/trace-round1.jsonl) — analyzed facts

Exact numbers from the real Qwen3.5-2B tokenizer + chat template (`python tools/analyze_trace.py`):

- 120 requests = **20 parallel conversations × 6 turns**. Waves at t = 0, 5, 10, 15, 20, 25s; each wave sends the next turn of all 20 conversations. Each request is an exact string-prefix extension of that conversation's previous request.
- Prompt tokens: wave 0 avg 12,949 → wave 5 avg 27,372; absolute max **27,398**. So `--max-model-len=32768` suffices; we use 40960 for margin. Requests longer than max-model-len get HTTP 400 → score 0 (killed submit_001).
- Shared system prompt: **6,396 tokens**, byte-identical across all 120 requests. Each turn adds ~2,885 tokens.
- Naive total prefill 2.42M tokens; perfect prefix reuse leaves **426k unique tokens** (5.7× less).
- Every body: `max_tokens=200`, `temperature=0`, `seed=42`, no `stream` flag (replay with streaming locally to measure TTFT).
- **Primer**: the harness runs a primer pass before the scored run (submit_001's error text: "primer: 120/120 transport errors"). If the primer replays the same 120 prompts, a server whose prefix cache works and retains everything enters the scored run ~100% warm — the likely explanation for top scores ≈94. **Cache hit rate across primer → scored run is the whole game; TTFT then measures cache lookup, not prefill.**

## Model architecture (Qwen/Qwen3.5-2B) — from HF config.json

- **Hybrid linear-attention** (Qwen3-Next style), NOT a plain dense transformer: 24 layers = 18 `linear_attention` + **6 `full_attention`** (every 4th). `Qwen3_5ForConditionalGeneration`, multimodal repo (vision tower ships in the 4.55GB BF16 weights); trace is text-only. `max_position_embeddings` 262144.
- KV cache exists only for the 6 full-attn layers: 2 KV heads × head_dim 256 → **12 KB/token BF16** (6 KB fp8). The 18 linear layers instead keep a **fixed ~21MB fp32 recurrent state per sequence** (`mamba_ssm_dtype: float32`).
- KV pool on 18GB at util 0.95 with BF16 weights ≈ 11GB → **~900k tokens in BF16**. The full 426k-token working set fits ~2× over **without any quantization** → VRAM capacity is NOT the constraint; whether the serving stack can prefix-cache/checkpoint the hybrid linear state is.
- Has an **MTP head** (`mtp_num_hidden_layers: 1`) → native speculative decoding for TPOT if the stack supports it for this arch.
- Working hypothesis for submit_002's 9.17: prefix caching ineffective for this hybrid arch on vLLM v0.22.1 and/or broken by `--kv-cache-dtype=fp8` → every request re-prefilled 13k–27k tokens. Verify on a rented GPU (replay the same request twice; warm TTFT should collapse to ~ms) before trusting any config.
- Image versions as of 08/07/2026: baseline `vllm/vllm-openai:v0.22.1`; newest stable `vllm/vllm-openai:v0.24.0`, `lmsysorg/sglang:v0.5.14` — newer stacks likely have better Qwen3.5 hybrid support.

## Repo layout & conventions

- `input/` — organizer-provided files (trace, baseline compose). Tracked in git.
- `submissions/submit_NNN/` — one dir per portal submission: the exact `docker-compose.yml` sent, plus a screenshot of the portal result (screenshots are git-ignored, local only — record the outcome as text in this file instead). Never edit an old submission dir; create the next `submit_NNN`.
- `requirements/` — scraped competition pages (đề bài, scoring formulas). **Git-ignored, exists only on this machine**; the portal is the source of truth.
- `context_support/` — reference notes (e.g. RunPod GPU prices). Also git-ignored, local only.
- `.gitignore` deliberately blocks model weights (`*.safetensors`, `models/`) — never commit weights.
- No local NVIDIA GPU is assumed on this Windows machine; GPU experiments run on rented cloud pods (RunPod).

## Submission history — append outcomes here

- **submit_001** (08/07/2026): baseline + `--kv-cache-dtype=fp8`, `--enable-chunked-prefill`, `--max-model-len=8192` → **FAILED**: "primer: 120/120 transport errors — contestant server unscoreable". Cause: max-model-len 8192 < prompt lengths (13k–30k tokens), so vLLM rejected every request. Lesson above.
- **submit_002** (08/07/2026 05:29): same as 001 but `--max-model-len=40960` → **9.17** (graded OK). Server works; latency near the score floor — TTFT/TPOT mostly ≥ ceilings under 20-way concurrent long prefills. Baseline to beat with prefix-reuse/quantization/scheduler tuning.
- Leaderboard reference (08/07/2026): #1 pipilabu 94.02, #10 61.15.

## Useful commands

Analyze the trace with the real tokenizer (token counts, waves, prefix sharing):

```powershell
python tools\analyze_trace.py
```

Validate a compose file before submitting:

```powershell
docker compose -f submissions\submit_002\docker-compose.yml config
```

Smoke-test an endpoint (on a GPU pod) the way the benchmark sees it:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.5-2B","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
```
