# `nguyenlong26/lfm25-custom-serve`

A custom serving image for round 2. It is a stock vLLM image plus one ~15 KB
layer of Python that patches vLLM at runtime. It is still vLLM — the framework
lock is respected, the enforced entrypoint
`python3 -m vllm.entrypoints.openai.api_server` is untouched, and the base is a
published `vllm/vllm-openai` tag pinned by digest.

## Why an image at all

Every stock-flag lever is closed (see the lever table in `CLAUDE.md`): quant,
scheduler, backend, cudagraph, fusion and speculative-decoding families were all
measured and rejected. The remaining ideas need code, not flags.

## Published tags

Base A = `vllm/vllm-openai:v0.25.1` @ `sha256:e4f88a83…`
Base B = `vllm/vllm-openai:nightly-0ba2aa35a81dcc3246b26291368b53fa2389c7d7` @ `sha256:eb8ef841…`

| tag | base | `FP8_LMHEAD` | `FIX_ASYNC_SPEC` |
|---|---|---|---|
| `v1-0251` | A | off | off |
| `v1-0251-fp8head` | A | **on** | off |
| `v1-n0ba2aa3` | B | off | off |
| `v1-n0ba2aa3-fp8head` | B | **on** | off |
| `v2-0251-specfix` | A | off | **on** |
| `v2-0251-fp8head-specfix` | A | **on** | **on** |
| `v2-n0ba2aa3-specfix` | B | off | **on** |

Variants differ only in `lfm25_defaults.py`; everything else is byte-identical,
which is what makes the 2×2 in submissions 031–035 a clean comparison.

**`v1-*` tags are frozen.** Submissions have been graded against them, and BTC
bans swapping a Docker image after submitting. New payload work always gets a
new version prefix — never re-push a tag that has been submitted. Verify with
`crane digest` after any push session.

Mirroring the nightly here also removes a real risk: Docker Hub prunes
commit-pinned nightlies after about nine days, and a final-5 pick whose tag
disappears before hậu kiểm is unservable.

## Why the default is baked into the tag, not passed as an env var

It is not known whether the grading harness forwards `environment:` from
`docker-compose.yml`. The harness already rewrites the entrypoint and renames
the container, so assuming it preserves anything else costs a submission slot to
find out. Baking the default makes each tag self-describing; the matching
`LFM25_*` variable still overrides it locally, which is what
`tools/runpod/fp8_lmhead_check.sh` uses.

## What the layer contains

| file | role |
|---|---|
| `lfm25_serve.pth` | one line, `import lfm25_boot`, run at interpreter start |
| `lfm25_boot.py` | installs a post-import hook; never imports vLLM itself |
| `lfm25_patches.py` | the actual patch: FP8 lm_head |
| `lfm25_defaults.py` | per-tag defaults, generated at build time |

A `.pth` rather than `sitecustomize.py`: the vLLM base image already ships a
`sitecustomize` earlier on `sys.path`, which silently shadows ours. That cost a
day in round 1.

The payload is written into six candidate site-packages directories so the build
does not have to know the base image's Python version. Python only processes
`.pth` files inside genuine site directories, so the rest are inert.

### The patch: FP8 lm_head

`Fp8Config.get_quant_method` returns an FP8 method only for `LinearBase`, and
`ParallelLMHead` subclasses `VocabParallelEmbedding`, so `--quantization=fp8`
leaves the 65536×2048 tied output head in BF16. Reading it costs 0.268 GB on
every decode step — about 0.45 ms of the portal's measured 4.07 ms TPOT at the
slice's ~600 GB/s. The patch quantizes it to `float8_e4m3fn` with per-row scales
into a separate buffer and swaps `lm_head.quant_method.apply` for a
`torch._scaled_mm` path.

Quantization is on the fly at startup from the mounted BF16 weights, the form
BTC sanctioned on 25/07. The original BF16 tensor is kept, so the tied *input*
embedding lookup stays bit-identical to stock; the cost is 134 MB of VRAM, taken
before vLLM profiles memory, so the KV pool shrinks by under 1%.

Failure handling, in order:

1. the hook wraps `GPUModelRunner.load_model`, so a signature change means the
   patch never arms rather than crashing the server;
2. a startup self-test compares FP8 against stock BF16 on real rows of the head,
   on random hidden states, and at batch 1, and rolls back unless cosine ≥ 0.999,
   max relative error ≤ 2%, and the argmax matches;
3. any exception in the fast path permanently reverts to BF16.

No request-dependent branching: after startup either every token goes through
FP8 or every token goes through BF16. That distinction matters — BTC's
anti-cheat rules name "dual-path mechanisms" explicitly, and submission 015 was
aborted online with a "truncation / dual-path likely" flag.

Grep the container log for `[lfm25]`. `fp8 lm_head: ACTIVE` means it is live;
`REJECTED by self-test` means it fell back and the run equals its stock sibling.

## Rebuild and publish

No Docker daemon and no local copy of the ~10 GB base is needed. `crane append`
pushes the small layer and cross-repo-mounts the base layers.

```bash
python docker/lfm25-custom-serve/build_layer.py
crane auth login index.docker.io -u nguyenlong26     # once, access token
crane append -b vllm/vllm-openai@sha256:<base-digest> \
  -f docker/lfm25-custom-serve/build/layer_fp8head.tar \
  -t nguyenlong26/lfm25-custom-serve:<tag>
```

A `docker build` on the Windows workstation is not an option — one attempt in
round 1 filled the system drive and took the engine down with it.

After a first push to a new repository, check on Docker Hub that it is public;
BTC pulls anonymously. Verify with an anonymous token rather than trusting the
UI:

```bash
TOK=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:nguyenlong26/lfm25-custom-serve:pull" | python -c "import json,sys;print(json.load(sys.stdin)['token'])")
curl -s -o /dev/null -w "%{http_code}\n" -H "Authorization: Bearer $TOK" \
  -H "Accept: application/vnd.oci.image.manifest.v1+json" \
  https://registry-1.docker.io/v2/nguyenlong26/lfm25-custom-serve/manifests/v1-n0ba2aa3-fp8head
```

### The patch: async spec-decode placeholder repair

vLLM inserts and repairs optimistic speculative-decode placeholders under two
guards that do not agree. Insertion (`gpu_model_runner.py:1317`) happens
whenever async scheduling is on and the last step drafted:

```python
req_state.output_token_ids.extend([-1] * optimistic_num_accepted)
```

Repair (`gpu_input_batch.py:1019`) returns immediately unless
`sampling_metadata.output_token_ids` is populated, which requires penalties,
bad words, a custom logits processor, or a thinking budget. The graded workload
is greedy with none of those, so the repair never runs and the `-1` entries stay
in the request's token history — the same list that feeds `token_ids_cpu` and
the n-gram corpus.

Async scheduling is on by default and is auto-disabled for speculative decoding
*except* for Eagle types, `dspark`, and `NgramGPUTypes = Literal["ngram_gpu"]`.
So `ngram_gpu` keeps async and hits this bug — that is submit_015, aborted
online with "long-context probe failed (0%)" — while CPU `ngram` silently loses
async scheduling and fails a **different** way (token duplication, measured
25/07). This patch does not fix CPU ngram; do not run it expecting one.

The fix lets the existing repair run when there is anything to repair. It is
self-limiting: a request with no `-1` tail is skipped before any GPU sync, so
with speculative decoding off it costs a loop over a few list tails per step and
does not damage async scheduling.

Unlike the FP8 head this cannot be self-tested at startup, so the contract is
covered by unit tests instead (overshoot shrinks the list, undershoot clamps,
the no-placeholder path performs no sync, the stock path still delegates,
internal failure logs `OUTPUT MAY BE CORRUPT` rather than failing silently).
The ship gate is exact greedy equivalence on a pod.

## Validate before spending slots

```bash
bash tools/runpod/fp8_lmhead_check.sh          # FP8 head, on a prepared H100 pod
bash tools/runpod/specfix_check.sh             # spec-decode fix
```

**The two patches need different gates, and using the wrong one has already
cost this team a submission.**

`fp8_lmhead_check.sh` runs F0 inert → F1 patched → F2 drift control. Its gate is
*not* exact greedy equivalence — quantization is expected to move some tokens.
What must hold is that needle retrieval does not regress, output stays coherent,
and TPOT drops.

`specfix_check.sh` runs B0 reference → B1 `ngram_gpu` unpatched → B2 patched.
Its gate **is** exact greedy equivalence, 6/6, because speculative decoding is
lossless by construction and any divergence proves a bug. B1 exists to keep the
result honest: it must FAIL, otherwise the diagnosis is wrong and B2 proves
nothing.

Both scripts install the image payload into the pod's own site-packages, so what
is measured is the same code the published image ships.

## Tests that run without a GPU

```bash
python docker/lfm25-custom-serve/tests/test_boot.py docker/lfm25-custom-serve/payload
python docker/lfm25-custom-serve/tests/test_async_spec_fix.py docker/lfm25-custom-serve/payload
```

`test_boot.py` builds a stand-in `vllm.v1.worker.gpu_model_runner` on disk and
checks that the post-import hook fires on it deterministically, that unrelated
imports keep working while the finder is installed, and that a missing model, a
missing defaults module and `LFM25_PATCH=0` all degrade quietly.

`test_async_spec_fix.py` covers the repair's full contract against fake batches —
optimistic overshoot shrinks the list, undershoot clamps, a request with no
placeholders is skipped **without forcing a GPU sync**, the stock path still
delegates, removed slots and empty rows do not raise, and an internal failure is
logged loudly. Run both after touching anything in `payload/`; they are cheap and
they cover the failure modes that would otherwise cost a submission slot to find.

## Adding another patch

Put it in `lfm25_patches.py` behind its own flag, add the flag to `VARIANTS` in
`build_layer.py`, and publish a new tag. Keep the same three-layer failure
handling — arm-or-skip, startup self-test, runtime fallback — because a patch
that cannot be trusted to fail quietly is worth less than the slot it costs.
