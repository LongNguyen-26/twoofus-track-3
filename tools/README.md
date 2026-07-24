# Tool layout

Run commands from the repository root unless a script says otherwise.

- `analysis/`: offline trace analysis and experiment-result summarizers.
- `replay/`: round-specific synthetic or organizer-trace replay clients.
- `evaluation/`: correctness, long-context, and GPQA checks.
- `runpod/`: RunPod setup, battery, and image-packaging scripts.
- `docs/`: experiment reports and operational runbooks.

Current round-2 candidate workflow:

```bash
bash tools/runpod/pod_install_bitsandbytes.sh
bash tools/runpod/w4_battery_r2.sh
```

See `tools/docs/W4_RUNPOD.md` for the bracket, candidate gates, and follow-up
GPQA command.

Suffix decoding is archived after failing correctness and latency gates. See
`tools/docs/SUFFIX_FINDINGS_20260724.md`; do not publish the suffix image.
