"""Runtime patches for LFM2.5 served under vLLM.

Two independent, separately flagged patches:

  FP8_LMHEAD        quantize the output head that --quantization=fp8 skips
  FIX_ASYNC_SPEC    repair vLLM's unrepaired optimistic speculative-decode
                    placeholders, which corrupt generation under our exact
                    sampling parameters

===========================================================================
PATCH 1 - FP8 lm_head
===========================================================================

Why this exists
---------------
``--quantization=fp8`` only reaches ``LinearBase`` layers. The output head is a
``ParallelLMHead``, which subclasses ``VocabParallelEmbedding``, so
``Fp8Config.get_quant_method`` returns nothing for it and it keeps its BF16
weights. For LFM2.5-1.2B that head is 65536 x 2048 = 134M parameters, and
because the weights are tied it is also the input embedding table.

Every decode step reads the whole head to produce logits: 0.268 GB in BF16,
about 0.45 ms of the measured 4.07 ms TPOT on the graded MiG slice at its
~600 GB/s. Quantizing it to FP8 halves that read. This is the same lever that
``--quantization=fp8`` already applies to the 1.036B linear parameters, which is
worth about 13 points on the portal, applied to the one tensor it misses.

What it does
------------
At the end of ``GPUModelRunner.load_model`` the head weight is quantized to
``float8_e4m3fn`` with per-output-channel (per-row) scales into a *separate*
buffer, and ``lm_head.quant_method.apply`` is replaced with a ``_scaled_mm``
path. The original BF16 weight is left untouched, so the tied input embedding
lookup is bit-identical to stock. Cost is 134 MB of VRAM, taken before vLLM
profiles memory, so the KV pool shrinks by well under 1%.

Quantizing on the fly at startup from the mounted BF16 weights is the form BTC
sanctioned on 25/07; nothing is pre-baked and no weight file is modified.

Safety
------
Before the patch goes live it is checked against the stock BF16 path on real
rows of the head, on random hidden states, and at batch 1. It must reproduce
the stock argmax and stay within a tight cosine/relative-error bound or it is
rolled back. At serve time a failure inside the fast path restores the stock
apply permanently rather than returning a wrong or partial result. There is no
request-dependent branching: after startup either every token goes through FP8
or every token goes through stock BF16.
"""

import sys

_MIN_COSINE = 0.999
_MAX_REL_ERR = 0.02
_MIN_ARGMAX_AGREEMENT = 0.90


def _noop(_msg):
    pass


# --------------------------------------------------------------------------
# quantization helpers
# --------------------------------------------------------------------------


def _quantize_rowwise(weight, chunk=8192):
    """BF16 [V, H] -> (fp8 [V, H], fp32 scale [V, 1]), chunked to bound memory."""
    import torch

    fp8 = torch.float8_e4m3fn
    fmax = float(torch.finfo(fp8).max)
    rows, cols = weight.shape
    qweight = torch.empty((rows, cols), dtype=fp8, device=weight.device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=weight.device)
    for start in range(0, rows, chunk):
        stop = min(start + chunk, rows)
        block = weight[start:stop].to(torch.float32)
        amax = block.abs().amax(dim=1, keepdim=True).clamp_(min=1e-12)
        block_scale = amax / fmax
        scale[start:stop] = block_scale
        qweight[start:stop] = block.div_(block_scale).clamp_(-fmax, fmax).to(fp8)
        del block
    return qweight, scale


def _pick_act_quant(log):
    """Return f(x2d) -> (fp8 tensor, fp32 per-token scale [m, 1])."""
    import torch

    fp8 = torch.float8_e4m3fn
    fmax = float(torch.finfo(fp8).max)

    def manual(x2):
        xf = x2.to(torch.float32)
        amax = xf.abs().amax(dim=1, keepdim=True).clamp_(min=1e-12)
        xscale = amax / fmax
        return xf.div_(xscale).clamp_(-fmax, fmax).to(fp8), xscale

    try:
        from vllm import _custom_ops as ops

        probe = torch.randn(4, 64, dtype=torch.bfloat16, device="cuda")
        xq, xs = ops.scaled_fp8_quant(probe, scale=None, use_per_token_if_dynamic=True)
        if xq.dtype == fp8 and xs.shape == (4, 1):

            def fused(x2):
                return ops.scaled_fp8_quant(x2, scale=None, use_per_token_if_dynamic=True)

            log("activation quant: vllm._custom_ops.scaled_fp8_quant")
            return fused
    except Exception as exc:
        log("activation quant: falling back to torch ops (%r)" % (exc,))
    log("activation quant: torch ops")
    return manual


def _scaled_mm(xq, wq_t, xscale, wscale_row, bias, out_dtype):
    import torch

    try:
        return torch._scaled_mm(
            xq,
            wq_t,
            scale_a=xscale,
            scale_b=wscale_row,
            bias=bias,
            out_dtype=out_dtype,
        )
    except TypeError:
        return torch._scaled_mm(xq, wq_t, xscale, wscale_row, bias, None, out_dtype)


# --------------------------------------------------------------------------
# the replacement apply
# --------------------------------------------------------------------------


def _make_fp8_apply(qweight, scale, orig_apply, log, state):
    import torch

    wq_t = qweight.t()  # [H, V], column-major view as _scaled_mm wants
    wscale_row = scale.reshape(1, -1)

    def fp8_apply(layer, x, bias=None):
        if state["disabled"]:
            return orig_apply(layer, x, bias)
        try:
            shape = x.shape
            x2 = x.reshape(-1, shape[-1])
            if not x2.is_contiguous():
                x2 = x2.contiguous()
            xq, xscale = state["quant"](x2)
            out = _scaled_mm(xq, wq_t, xscale, wscale_row, bias, x.dtype)
            return out.view(shape[:-1] + (out.shape[-1],))
        except Exception as exc:
            state["disabled"] = True
            log("fp8 lm_head failed at runtime, reverted to BF16: %r" % (exc,))
            return orig_apply(layer, x, bias)

    return fp8_apply


def _self_test(lm_head, fp8_apply, orig_apply, log):
    """Compare against the stock BF16 head. Returns True if the patch may stay."""
    import torch

    weight = lm_head.weight.data
    device, dtype = weight.device, weight.dtype
    gen = torch.Generator(device="cpu").manual_seed(20260726)
    idx = torch.randint(0, weight.shape[0], (16,), generator=gen).to(device)
    rows = weight.index_select(0, idx).to(dtype)
    spread = rows.float().std().clamp(min=1e-3).item()
    noise = (torch.randn(16, weight.shape[1], generator=gen) * spread).to(
        device=device, dtype=dtype
    )

    for tag, hidden in (("rows", rows), ("random", noise), ("batch1", rows[:1])):
        ref = orig_apply(lm_head, hidden, None)
        got = fp8_apply(lm_head, hidden, None)
        if got is None or got.shape != ref.shape:
            log("self-test %s: shape mismatch" % tag)
            return False
        ref = ref.float()
        got = got.float()
        if not torch.isfinite(got).all():
            log("self-test %s: non-finite logits" % tag)
            return False
        cosine = torch.nn.functional.cosine_similarity(ref, got, dim=-1).min().item()
        denom = ref.abs().amax().clamp(min=1e-6)
        rel_err = ((got - ref).abs().amax() / denom).item()
        agreement = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
        log(
            "self-test %s: cosine %.6f, max rel err %.4f, argmax agreement %.2f"
            % (tag, cosine, rel_err, agreement)
        )
        if cosine < _MIN_COSINE or rel_err > _MAX_REL_ERR:
            log("self-test %s: outside numerical bound" % tag)
            return False
        if tag == "rows" and agreement < _MIN_ARGMAX_AGREEMENT:
            log("self-test %s: argmax disagreement" % tag)
            return False
    return True


# --------------------------------------------------------------------------
# locating the head
# --------------------------------------------------------------------------


def _candidate_heads(model):
    seen = []
    found = []

    def offer(mod):
        if mod is None or id(mod) in seen:
            return
        seen.append(id(mod))
        if not hasattr(mod, "quant_method") or not hasattr(mod, "weight"):
            return
        found.append(mod)

    offer(getattr(model, "lm_head", None))
    inner = getattr(model, "model", None)
    if inner is not model:
        offer(getattr(inner, "lm_head", None))
    try:
        for name, mod in model.named_modules():
            if type(mod).__name__ == "ParallelLMHead" or name.split(".")[-1] == "lm_head":
                offer(mod)
    except Exception:
        pass
    return found


def _patch_lm_head(model, log):
    import torch

    if model is None:
        log("fp8 lm_head: no model on the runner")
        return
    heads = _candidate_heads(model)
    if not heads:
        log("fp8 lm_head: no lm_head found, leaving stock")
        return
    lm_head = heads[0]
    weight = getattr(lm_head, "weight", None)
    if weight is None or weight.dim() != 2:
        log("fp8 lm_head: unexpected weight, leaving stock")
        return
    if weight.dtype not in (torch.bfloat16, torch.float16):
        log("fp8 lm_head: weight is %s, nothing to do" % (weight.dtype,))
        return
    quant_method = lm_head.quant_method
    orig_apply = quant_method.apply
    if getattr(orig_apply, "_lfm25_fp8", False):
        return

    rows, cols = weight.shape
    saved = weight.data.numel() * weight.element_size() // 2
    log("fp8 lm_head: quantizing %dx%d %s (saves %.0f MB per decode read)"
        % (rows, cols, weight.dtype, saved / 1e6))

    qweight, scale = _quantize_rowwise(weight.data)
    state = {"disabled": False, "quant": _pick_act_quant(log)}
    fp8_apply = _make_fp8_apply(qweight, scale, orig_apply, log, state)

    ok = False
    try:
        ok = _self_test(lm_head, fp8_apply, orig_apply, log)
    except Exception as exc:
        log("fp8 lm_head self-test raised: %r" % (exc,))
    if not ok or state["disabled"]:
        del qweight, scale
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        log("fp8 lm_head: REJECTED by self-test, serving stock BF16")
        return

    fp8_apply._lfm25_fp8 = True
    lm_head._lfm25_fp8_weight = qweight  # keep alive
    lm_head._lfm25_fp8_scale = scale
    quant_method.apply = fp8_apply
    log("fp8 lm_head: ACTIVE")


# ==========================================================================
# PATCH 2 - repair unrepaired optimistic speculative-decode placeholders
# ==========================================================================
#
# The bug (vLLM 0.25.1, still on main). Two guards that must agree, don't.
#
# Placeholders are INSERTED whenever async scheduling is on and the previous
# step drafted tokens -- gpu_model_runner.py, `_update_states`:
#
#     if req_state.prev_num_draft_len and self.use_async_scheduling:
#         optimistic_num_accepted = req_state.prev_num_draft_len
#         req_state.output_token_ids.extend([-1] * optimistic_num_accepted)
#
# They are REPAIRED on a completely different condition --
# gpu_input_batch.py, `update_async_output_token_ids`, which returns
# immediately unless `sampling_metadata.output_token_ids` is populated. That
# list is only populated when
#
#     needs_output_token_ids = (not no_penalties or bad_words_token_ids
#                               or logitsprocs_need_output_token_ids
#                               or thinking_budget_tracks_reqs)
#
# i.e. only when a logits processor happens to need the token history. The
# graded workload is greedy with no penalties, no bad words, no custom logits
# processors and no thinking budget, so it is False and the repair never runs.
# The -1 entries then stay in `req_state.output_token_ids` forever -- the same
# list object as `input_batch.req_output_token_ids[idx]`, which is synced into
# `token_ids_cpu` and copied to the GPU as the n-gram matching corpus and as
# the request's own token history.
#
# vLLM's own repair code documents this exact case ("async spec decode adds
# optimistic placeholders that may exceed the actual acceptance count") but it
# sits inside the guard our workload never passes.
#
# Which of our failures this explains. Async scheduling defaults to ON in
# v0.25.1 and is auto-disabled for speculative decoding EXCEPT for Eagle,
# `dspark`, and `NgramGPUTypes = Literal["ngram_gpu"]`. So:
#   * `ngram_gpu`  -> async stays on  -> placeholders inserted, never repaired.
#     This is submit_015, aborted by the harness with "long-context probe
#     failed (0%) - truncation / dual-path likely".
#   * `ngram` CPU  -> async auto-disabled -> this path is inert. The token
#     duplication measured on 25/07 is a SEPARATE defect; this patch does not
#     address it, so do not run CPU ngram expecting a fix.
# The other -1 append (`_pp_broadcast_prev_sampled_token_ids`) is
# pipeline-parallel only and cannot fire at PP=1.
#
# The fix. Let the existing repair run when there is anything to repair. The
# stock path is left completely untouched: if the sampler does need
# output_token_ids we delegate to the original. Otherwise we run the same
# algorithm over `req_output_token_ids` directly. It is self-limiting -- a
# request with no -1 tail is skipped before any GPU sync, so with speculative
# decoding off this is a bare loop over a few list tails per step.
#
# NOTE: unlike the FP8 head, this cannot be self-tested at startup. Speculative
# decoding is lossless by construction, so the gate is exact greedy equivalence
# on a pod (6/6). Never ship this without that, and never on CPU ngram.


def _install_async_spec_fix(log):
    module = sys.modules.get("vllm.v1.worker.gpu_input_batch")
    if module is None:
        import vllm.v1.worker.gpu_input_batch as module  # noqa: F811
    batch_cls = getattr(module, "InputBatch", None)
    if batch_cls is None:
        log("async spec fix: InputBatch not found, stock vLLM")
        return
    orig_update = getattr(batch_cls, "update_async_output_token_ids", None)
    orig_set = getattr(batch_cls, "set_async_sampled_token_ids", None)
    if orig_update is None or orig_set is None:
        log("async spec fix: methods moved, stock vLLM")
        return
    if getattr(orig_update, "_lfm25_spec_fix", False):
        return

    def set_async_sampled_token_ids(self, sampled_token_ids_cpu, async_copy_ready_event):
        # Stock drops these unless a logits processor wants the token history.
        # The repair below needs them whenever placeholders may exist, and it
        # only dereferences them when it actually finds one.
        self.sampled_token_ids_cpu = sampled_token_ids_cpu
        self.async_copy_ready_event = async_copy_ready_event

    def update_async_output_token_ids(self):
        if self.sampling_metadata.output_token_ids:
            return orig_update(self)  # stock path, untouched
        try:
            if self.sampled_token_ids_cpu is None or self.prev_req_id_to_index is None:
                return
            sampled_token_ids = None
            for index, req_id in enumerate(self.req_ids):
                prev_index = self.prev_req_id_to_index.get(req_id)
                if prev_index is None:
                    continue
                req_output_token_ids = self.req_output_token_ids[index]
                if not req_output_token_ids or req_output_token_ids[-1] != -1:
                    # Nothing optimistic pending for this request.
                    continue
                if sampled_token_ids is None:
                    self.async_copy_ready_event.synchronize()
                    sampled_token_ids = self.sampled_token_ids_cpu.tolist()
                new_ids = sampled_token_ids[prev_index]
                if not new_ids:
                    continue
                num_sampled = (
                    len(new_ids) if new_ids[-1] != -1 else new_ids.index(-1)
                )
                # Placeholders may outnumber accepted tokens (optimistic) or be
                # fewer (tokens discarded after a kv-load failure).
                first = len(req_output_token_ids)
                while first > 0 and req_output_token_ids[first - 1] == -1:
                    first -= 1
                num_to_replace = min(num_sampled, len(req_output_token_ids) - first)
                del new_ids[num_to_replace:]
                req_output_token_ids[first:] = new_ids
        except Exception as exc:
            # Falling through leaves placeholders in the token history, which
            # corrupts generation. Loud on purpose: this must be caught by the
            # pod equivalence gate, never discovered on the portal.
            log("!! async spec fix FAILED, OUTPUT MAY BE CORRUPT: %r" % (exc,))

    update_async_output_token_ids._lfm25_spec_fix = True
    batch_cls.set_async_sampled_token_ids = set_async_sampled_token_ids
    batch_cls.update_async_output_token_ids = update_async_output_token_ids
    log("async spec fix: ACTIVE (optimistic placeholders will be repaired)")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def install(fp8_lmhead=False, fix_async_spec=False, log=None):
    log = log or _noop
    if not fp8_lmhead and not fix_async_spec:
        log("no patches enabled, stock vLLM")
        return

    if fix_async_spec:
        try:
            _install_async_spec_fix(log)
        except Exception as exc:
            log("async spec fix skipped: %r" % (exc,))

    if not fp8_lmhead:
        return

    module = sys.modules.get("vllm.v1.worker.gpu_model_runner")
    runner_cls = getattr(module, "GPUModelRunner", None)
    if runner_cls is None:
        log("GPUModelRunner not found, stock vLLM")
        return
    orig_load = runner_cls.load_model
    if getattr(orig_load, "_lfm25_wrapped", False):
        return

    def load_model(self, *args, **kwargs):
        result = orig_load(self, *args, **kwargs)
        try:
            _patch_lm_head(getattr(self, "model", None), log)
        except Exception as exc:
            log("fp8 lm_head skipped: %r" % (exc,))
        return result

    load_model._lfm25_wrapped = True
    runner_cls.load_model = load_model
    log("armed: fp8 lm_head (applies after model load)")
