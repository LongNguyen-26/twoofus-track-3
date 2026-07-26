"""Startup shim for the LFM2.5 custom serving image.

Loaded by ``lfm25_serve.pth`` at interpreter start, i.e. before the enforced
entrypoint ``python3 -m vllm.entrypoints.openai.api_server`` imports vLLM.
A ``.pth`` is used rather than ``sitecustomize.py`` because the vLLM base image
already ships a ``sitecustomize`` earlier on ``sys.path`` that would shadow ours
(learned the hard way in round 1).

Nothing here touches vLLM at import time. It installs a cheap ``sys.meta_path``
trigger that fires on subsequent imports and applies the patches once the target
modules exist. Every step is wrapped so that a failure degrades to stock vLLM
behaviour rather than breaking the server.

Patches are opt-in and controlled by environment variables, with per-image
defaults baked into ``lfm25_defaults.py``:

    LFM25_PATCH=0           disable everything
    LFM25_FP8_LMHEAD=1      quantize the (BF16, untied-at-runtime) lm_head to FP8
    LFM25_FP8_LINEARS=1     quantize Linear layers --quantization=fp8 skipped
                            (LFM2.5's 10 ShortConv in_proj/out_proj pairs, which
                            vLLM builds without a quant_config at all)
    LFM25_FIX_ASYNC_SPEC=1  repair vLLM's unrepaired optimistic speculative
                            decoding placeholders (only meaningful together
                            with --speculative-config method=ngram_gpu)
"""

import os
import sys

_PREFIX = "[lfm25]"
_TARGET = "vllm.v1.worker.gpu_model_runner"
_state = {"installed": False, "trigger": None}


def _log(msg):
    try:
        sys.stderr.write("%s %s\n" % (_PREFIX, msg))
        sys.stderr.flush()
    except Exception:
        pass


def _default(name, fallback):
    try:
        import lfm25_defaults

        return bool(getattr(lfm25_defaults, name, fallback))
    except Exception:
        return fallback


def _flag(env_name, default_name, fallback):
    raw = os.environ.get(env_name)
    if raw is None:
        return _default(default_name, fallback)
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _enabled():
    raw = os.environ.get("LFM25_PATCH", "1")
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _try_install():
    """Apply patches if their target modules are loaded. Cheap and idempotent."""
    if _state["installed"]:
        return
    if _TARGET not in sys.modules:
        return
    # Deliberately left on sys.meta_path: removing an entry while the import
    # system is iterating it makes the loop skip the next finder. The guard
    # above is one dict lookup per import, which is not worth that risk.
    _state["installed"] = True
    try:
        import lfm25_patches

        lfm25_patches.install(
            fp8_lmhead=_flag("LFM25_FP8_LMHEAD", "FP8_LMHEAD", False),
            fp8_linears=_flag("LFM25_FP8_LINEARS", "FP8_LINEARS", False),
            fix_async_spec=_flag("LFM25_FIX_ASYNC_SPEC", "FIX_ASYNC_SPEC", False),
            log=_log,
        )
    except Exception as exc:  # never break the server
        _log("patch install failed, running stock vLLM: %r" % (exc,))


class _WrappedLoader:
    """Delegating loader that runs a callback once the module has executed."""

    def __init__(self, inner, after):
        self._inner = inner
        self._after = after

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        try:
            self._after()
        except Exception as exc:
            _log("post-import hook failed, stock vLLM: %r" % (exc,))

    def __getattr__(self, item):  # load_module, get_source, is_package, ...
        return getattr(self._inner, item)


class _Trigger:
    """Meta-path finder that patches the instant the target module loads.

    For the target it borrows the real spec from ``PathFinder`` and wraps the
    loader, which makes the patch deterministic rather than dependent on some
    later import happening to tick us. For everything else it returns None,
    after a cheap idempotent check that also covers the target being imported
    by a mechanism ``PathFinder`` does not serve.
    """

    def find_spec(self, fullname, path=None, target=None):
        try:
            if fullname != _TARGET:
                _try_install()
                return None
            import importlib.machinery

            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.loader is None:
                return None
            spec.loader = _WrappedLoader(spec.loader, _try_install)
            return spec
        except Exception as exc:
            _log("finder error, stock vLLM: %r" % (exc,))
            return None


def _boot():
    if not _enabled():
        _log("LFM25_PATCH=0, stock vLLM")
        return
    trigger = _Trigger()
    _state["trigger"] = trigger
    sys.meta_path.insert(0, trigger)
    _log("autopatch armed (via .pth)")


try:
    _boot()
except Exception as _exc:  # pragma: no cover - must never raise at startup
    try:
        sys.stderr.write("%s boot failed, stock vLLM: %r\n" % (_PREFIX, _exc))
    except Exception:
        pass
