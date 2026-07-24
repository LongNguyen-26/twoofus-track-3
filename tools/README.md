# Tool layout

Run commands from the repository root unless a script says otherwise.

- `analysis/`: offline trace analysis and experiment-result summarizers.
- `replay/`: round-specific synthetic or organizer-trace replay clients.
- `evaluation/`: correctness, long-context, and GPQA checks.
- `runpod/`: RunPod setup, battery, and image-packaging scripts.
- `docs/`: experiment reports and operational runbooks.

Current round-2 suffix workflow:

```bash
bash tools/runpod/pod_install_suffix.sh
bash tools/runpod/suffix_battery_r2.sh
```

See `tools/docs/SUFFIX_RUNPOD.md` for deployment, image publication, and
portal-compose instructions.
