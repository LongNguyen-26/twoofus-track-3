# Tool layout

Run commands from the repository root unless a script says otherwise.

- `analysis/`: offline trace analysis and experiment-result summarizers.
- `replay/`: round-specific synthetic or organizer-trace replay clients.
- `evaluation/`: correctness, long-context, and GPQA checks.
- `runpod/`: RunPod setup, battery, and image-packaging scripts.
- `docs/`: experiment reports and operational runbooks.

Current round-2 workflow (Hopper pod with a verified SM cap, from 25/07):

```bash
bash tools/runpod/pod_setup_h100.sh        # model + CUDA MPS SM cap + verification
bash tools/runpod/nightly_source_probe.sh  # has spec decode been unblocked upstream?
CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=12 python3 tools/runpod/int4_microbench.py --marlin
bash tools/runpod/h_battery_r2.sh          # portal anchors + cheap untested levers
```

See `tools/docs/H100_RUNPOD.md` for deploy settings, GPU choice and gates. The
RTX 4090 batteries (`pod_battery_r2.sh`, `k_battery_r2.sh`, `draft_battery_r2.sh`,
`w4_battery_r2.sh`, `suffix_battery_r2.sh`) are kept for reference, but they ran
without an SM cap and their absolute numbers mispredict the MiG slice.

Archived after failing their gates: suffix decoding
(`tools/docs/SUFFIX_FINDINGS_20260724.md` — do not publish the suffix image) and
BitsAndBytes W4 (`tools/docs/W4_RUNPOD.md`).
