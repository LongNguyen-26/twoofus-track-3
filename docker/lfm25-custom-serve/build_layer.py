#!/usr/bin/env python3
"""Build the tar layers for nguyenlong26/lfm25-custom-serve.

The layer is a handful of small Python files, so the image is published with
``crane append`` on top of a digest-pinned vLLM base rather than with
``docker build``. That needs no Docker daemon and no local copy of the ~10 GB
base image, and the base layers are cross-repo mounted by the registry.

The payload is dropped into every plausible site-packages directory. Python
only processes ``.pth`` files inside real site directories, so the copies that
land elsewhere are inert, and this avoids having to know the base image's exact
Python version ahead of time.

Usage:
    python docker/lfm25-custom-serve/build_layer.py [outdir]
"""

import io
import pathlib
import sys
import tarfile
import time

HERE = pathlib.Path(__file__).resolve().parent
PAYLOAD = HERE / "payload"

SITE_DIRS = [
    "usr/local/lib/python3.12/dist-packages",
    "usr/local/lib/python3.12/site-packages",
    "usr/local/lib/python3.13/dist-packages",
    "usr/local/lib/python3.13/site-packages",
    "usr/lib/python3/dist-packages",
    "usr/lib/python3.12/dist-packages",
]

PTH = "import lfm25_boot\n"

# One image tag per default. The environment variable still wins at runtime, but
# baking the default means a submission does not depend on the grading harness
# passing `environment:` through from docker-compose.yml.
VARIANTS = {
    "stock": {"FP8_LMHEAD": False, "FIX_ASYNC_SPEC": False},
    "fp8head": {"FP8_LMHEAD": True, "FIX_ASYNC_SPEC": False},
    # The spec-decode fix only does anything when the compose also passes
    # --speculative-config with method=ngram_gpu; on its own it is inert.
    "specfix": {"FP8_LMHEAD": False, "FIX_ASYNC_SPEC": True},
    "fp8head-specfix": {"FP8_LMHEAD": True, "FIX_ASYNC_SPEC": True},
}

DEFAULTS_HEADER = '''"""Per-image defaults, baked at build time.

Overridden at runtime by the matching LFM25_* environment variable.
"""

'''


def build(variant, flags, outdir):
    files = {
        "lfm25_boot.py": (PAYLOAD / "lfm25_boot.py").read_bytes(),
        "lfm25_patches.py": (PAYLOAD / "lfm25_patches.py").read_bytes(),
        "lfm25_defaults.py": (
            DEFAULTS_HEADER + "".join("%s = %r\n" % kv for kv in sorted(flags.items()))
        ).encode(),
        "lfm25_serve.pth": PTH.encode(),
    }
    # Normalise to LF so the .pth is never broken by a CRLF checkout.
    files = {name: data.replace(b"\r\n", b"\n") for name, data in files.items()}

    out = outdir / ("layer_%s.tar" % variant)
    mtime = int(time.time())
    with tarfile.open(out, "w", format=tarfile.GNU_FORMAT) as tar:
        for site in SITE_DIRS:
            for name, data in sorted(files.items()):
                info = tarfile.TarInfo("%s/%s" % (site, name))
                info.size = len(data)
                info.mode = 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = "root"
                info.mtime = mtime
                tar.addfile(info, io.BytesIO(data))
    print("%-8s -> %s (%d bytes, %d entries)"
          % (variant, out.name, out.stat().st_size, len(SITE_DIRS) * len(files)))
    return out


def main():
    outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "build"
    outdir.mkdir(parents=True, exist_ok=True)
    for variant, flags in VARIANTS.items():
        build(variant, flags, outdir)


if __name__ == "__main__":
    main()
