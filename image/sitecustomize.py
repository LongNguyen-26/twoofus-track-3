# Fallback loader for the exact-cache autopatch, for clean environments where
# no other sitecustomize shadows this one. The primary, shadow-proof mechanism
# is exact_cache.pth; this just converges on the same boot module (which is
# cached, so it runs at most once regardless of how many loaders import it).
try:
    import vllm_exact_cache_boot  # noqa: F401
except Exception as _e:  # pragma: no cover
    print(f"[exact-cache] sitecustomize fallback skipped: {_e}", flush=True)
