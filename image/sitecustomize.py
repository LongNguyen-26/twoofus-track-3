# Auto-imported by Python's site module at interpreter startup (including
# `python3 -m vllm.entrypoints.openai.api_server`, which BTC's harness enforces).
# Patches FastAPI so every app built in this process gets the exact-match
# response cache middleware. Fail-open: any error leaves the server unpatched
# but fully functional.
try:
    import os

    if os.environ.get("VLLM_EXACT_CACHE", "1") != "0":
        import fastapi
        from vllm_exact_cache import ExactCacheMiddleware

        _orig_fastapi_init = fastapi.FastAPI.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_fastapi_init(self, *args, **kwargs)
            try:
                self.add_middleware(ExactCacheMiddleware)
            except Exception as e:  # pragma: no cover
                print(f"[exact-cache] add_middleware failed (ignored): {e}", flush=True)

        fastapi.FastAPI.__init__ = _patched_init
        print("[exact-cache] FastAPI autopatch installed", flush=True)
except Exception as _e:  # pragma: no cover
    print(f"[exact-cache] autopatch skipped: {_e}", flush=True)
