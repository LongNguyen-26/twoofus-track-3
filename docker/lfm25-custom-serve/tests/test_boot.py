"""Smoke-test the .pth shim without torch or vLLM installed.

Covers the paths that would break the server at startup: the shim importing
cleanly, the post-import hook firing deterministically on a real (stand-in)
vllm.v1.worker.gpu_model_runner import, load_model being wrapped and still
returning the original result, unrelated imports continuing to work while the
finder is installed, and every failure mode degrading to stock instead of
raising.
"""
import json
import os
import pathlib
import sys
import tempfile
import types

PAYLOAD = sys.argv[1]
sys.path.insert(0, PAYLOAD)

results = {}

# --- build a stand-in vllm package on disk --------------------------------
root = pathlib.Path(tempfile.mkdtemp())
pkg = root / "vllm" / "v1" / "worker"
pkg.mkdir(parents=True)
for level in (root / "vllm", root / "vllm" / "v1", pkg):
    (level / "__init__.py").write_text("")
(pkg / "gpu_model_runner.py").write_text(
    "class GPUModelRunner:\n"
    "    def load_model(self, *a, **kw):\n"
    "        self.model = None\n"
    "        return 'loaded'\n"
)
sys.path.insert(0, str(root))

# --- 1. bare import, no vLLM imported yet ---------------------------------
os.environ["LFM25_FP8_LMHEAD"] = "1"
import lfm25_boot  # noqa: E402

results["armed"] = lfm25_boot._state["trigger"] is not None
results["trigger_first"] = sys.meta_path[0] is lfm25_boot._state["trigger"]
results["not_installed_yet"] = lfm25_boot._state["installed"] is False

# unrelated imports (incl. a builtin and a fresh stdlib module) must still work
import time as _t  # noqa: E402,F401
import colorsys  # noqa: E402,F401

results["stdlib_import_ok"] = colorsys.rgb_to_hls(0, 0, 0) == (0, 0, 0)

# --- 2. the deterministic post-import hook --------------------------------
import vllm.v1.worker.gpu_model_runner as gmr  # noqa: E402

results["installed_on_target_import"] = lfm25_boot._state["installed"] is True
results["load_model_wrapped"] = getattr(gmr.GPUModelRunner.load_model, "_lfm25_wrapped", False)
results["finder_still_present"] = lfm25_boot._state["trigger"] in sys.meta_path

runner = gmr.GPUModelRunner()
results["load_model_returns"] = runner.load_model() == "loaded"  # no lm_head -> no-op

# imports keep working after installation
import wave  # noqa: E402,F401

results["import_after_install_ok"] = True

# --- 3. install() is idempotent and honours the off switch ----------------
import lfm25_patches  # noqa: E402

before = gmr.GPUModelRunner.load_model
lfm25_patches.install(fp8_lmhead=True, log=lambda m: None)
results["idempotent"] = gmr.GPUModelRunner.load_model is before

lfm25_patches.install(fp8_lmhead=False, log=lambda m: None)
results["off_switch_noop"] = True

# --- 4. head discovery on torch-free stand-ins ---------------------------
class Bare:
    pass


class FakeHead:
    def __init__(self):
        self.quant_method = Bare()
        self.weight = Bare()


class FakeModel:
    def __init__(self):
        self.lm_head = FakeHead()

    def named_modules(self):
        return [("", self), ("lm_head", self.lm_head)]


results["finds_lm_head"] = len(lfm25_patches._candidate_heads(FakeModel())) == 1
results["no_head_is_safe"] = lfm25_patches._candidate_heads(
    types.SimpleNamespace(named_modules=lambda: [])
) == []
results["none_model_is_safe"] = lfm25_patches._patch_lm_head(None, lambda m: None) is None

# --- 5. missing defaults module and the kill switch ----------------------
results["default_fallback"] = lfm25_boot._default("FP8_LMHEAD", False) is False
os.environ["LFM25_PATCH"] = "0"
results["kill_switch"] = lfm25_boot._enabled() is False

print(json.dumps(results, indent=2))
bad = [k for k, v in results.items() if not v]
print("ALL PASS" if not bad else "FAILURES: %s" % bad)
sys.exit(1 if bad else 0)
