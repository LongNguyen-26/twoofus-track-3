#!/usr/bin/env bash
# Install the optional BitsAndBytes runtime needed for vLLM in-flight W4
# quantization without changing the image's PyTorch or vLLM dependencies.
#
# The stock vLLM image may already include a working build. In that case this
# script only reports the version and exits. Override BNB_SPEC to pin a wheel:
#   BNB_SPEC='bitsandbytes==0.46.1' bash tools/runpod/pod_install_bitsandbytes.sh

set -euo pipefail

BNB_SPEC="${BNB_SPEC:-bitsandbytes>=0.46.1}"

if python3 -c "import bitsandbytes" >/dev/null 2>&1 \
  && [[ "${FORCE_REINSTALL:-0}" != "1" ]]; then
  python3 -c \
    "import bitsandbytes as bnb, torch; print('bitsandbytes', bnb.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
  echo "BitsAndBytes is already importable; no installation was performed."
  exit 0
fi

echo "Installing ${BNB_SPEC} without dependency replacement ..."
python3 -m pip install --no-cache-dir --no-deps "$BNB_SPEC"

python3 -c \
  "import bitsandbytes as bnb, torch; print('bitsandbytes', bnb.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "Live pod is ready for: bash tools/runpod/w4_battery_r2.sh"
