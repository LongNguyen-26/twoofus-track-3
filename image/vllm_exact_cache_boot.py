"""Autopatch entrypoint for the exact-match response cache.

Imported at interpreter startup via `exact_cache.pth` (a .pth file processed by
site.py for every `python3` invocation, including the enforced
`python3 -m vllm.entrypoints.openai.api_server`). Unlike `sitecustomize`, a .pth
is not subject to first-match-on-sys.path shadowing, so this reliably runs even
when the base image already ships its own sitecustomize.

Patches FastAPI so every app built in this process gets the middleware.
Runs once (module is cached), and is idempotent even if imported again.
Fail-open: any error leaves the server unpatched but fully functional.
"""
try:
    import os

    if os.environ.get("VLLM_EXACT_CACHE", "1") != "0":
        import fastapi
        from vllm_exact_cache import ExactCacheMiddleware

        # Guard against double-patching if both .pth and sitecustomize import us.
        if not getattr(fastapi.FastAPI.__init__, "_exact_cache_patched", False):
            _orig_fastapi_init = fastapi.FastAPI.__init__

            def _patched_init(self, *args, **kwargs):
                _orig_fastapi_init(self, *args, **kwargs)
                try:
                    self.add_middleware(ExactCacheMiddleware)
                except Exception as e:  # pragma: no cover
                    print(f"[exact-cache] add_middleware failed (ignored): {e}", flush=True)

            _patched_init._exact_cache_patched = True
            fastapi.FastAPI.__init__ = _patched_init
            print("[exact-cache] FastAPI autopatch installed (via .pth)", flush=True)
except Exception as _e:  # pragma: no cover
    print(f"[exact-cache] autopatch skipped: {_e}", flush=True)
