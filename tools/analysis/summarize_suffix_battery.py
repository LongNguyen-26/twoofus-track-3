"""Summarize a results_suffix_* directory produced by the suffix battery."""

import argparse
import re
from pathlib import Path


RUN_IDS = ("D0A", "S4", "S8", "S16", "D0B")
MODES = ("fresh", "shared")

ERS_RE = re.compile(
    r"\[ALL REQUESTS SCORED\] ERS = ([0-9.]+) "
    r"\(~([0-9.]+) points\)\s+errors (\d+)/(\d+)"
)
TTFT_RE = re.compile(
    r"ttft_ms: p50=([0-9.]+) p95=([0-9.]+) "
    r"mean=([0-9.]+) max=([0-9.]+)"
)
TPOT_RE = re.compile(
    r"tpot_ms: p50=([0-9.]+) p95=([0-9.]+) mean=([0-9.]+)"
)


def read_run(path):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    ers = ERS_RE.search(text)
    ttft = TTFT_RE.search(text)
    tpot = TPOT_RE.search(text)
    if not (ers and ttft and tpot):
        return None
    return {
        "ers": float(ers.group(1)),
        "points": float(ers.group(2)),
        "errors": int(ers.group(3)),
        "requests": int(ers.group(4)),
        "ttft_p50": float(ttft.group(1)),
        "ttft_p95": float(ttft.group(2)),
        "tpot_p50": float(tpot.group(1)),
        "tpot_p95": float(tpot.group(2)),
    }


def extract_fraction(path, label):
    if not path.exists():
        return "-"
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"{re.escape(label)} (\d+)/(\d+)", text)
    return f"{match.group(1)}/{match.group(2)}" if match else "-"


def metric_total(path, metric):
    if not path.exists():
        return None
    total = 0.0
    found = False
    line_re = re.compile(
        rf"^vllm:spec_decode_num_{re.escape(metric)}_total"
        r"(?:\{[^}]*\})?\s+([0-9.eE+-]+)$"
    )
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = line_re.match(line)
        if match:
            total += float(match.group(1))
            found = True
    return total if found else None


def baseline_for(data, mode, field):
    vals = [
        data[(run_id, mode)][field]
        for run_id in ("D0A", "D0B")
        if data.get((run_id, mode)) is not None
    ]
    return sum(vals) / len(vals) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    args = ap.parse_args()
    root = args.results_dir

    data = {}
    for run_id in RUN_IDS:
        for mode in MODES:
            data[(run_id, mode)] = read_run(root / f"{run_id}_{mode}.log")

    print("Suffix battery summary")
    print(f"results: {root}")
    print()
    print(
        "config mode    points   delta  ttft50 ttft95  "
        "tpot50  tpot_gain  errors"
    )
    print(
        "------ ------  -------  ------  ------ ------  "
        "------  ---------  ------"
    )
    for run_id in RUN_IDS:
        for mode in MODES:
            row = data[(run_id, mode)]
            if row is None:
                continue
            base_points = baseline_for(data, mode, "points")
            base_tpot = baseline_for(data, mode, "tpot_p50")
            delta = row["points"] - base_points if base_points is not None else 0.0
            gain = (
                (base_tpot - row["tpot_p50"]) / base_tpot * 100
                if base_tpot
                else 0.0
            )
            print(
                f"{run_id:<6} {mode:<6}  {row['points']:7.2f} "
                f"{delta:+7.2f}  {row['ttft_p50']:6.0f} "
                f"{row['ttft_p95']:6.0f}  {row['tpot_p50']:6.2f} "
                f"{gain:+8.1f}%  {row['errors']:2d}/{row['requests']}"
            )

    print()
    print("config equivalence needle acceptance")
    print("------ ----------- ------ ----------")
    for run_id in RUN_IDS:
        eq = extract_fraction(root / f"{run_id}_equivalence.log", "EQUIVALENCE")
        needle = extract_fraction(root / f"{run_id}_needle.log", "RETRIEVAL")
        drafted = metric_total(root / f"{run_id}_metrics.txt", "draft_tokens")
        accepted = metric_total(root / f"{run_id}_metrics.txt", "accepted_tokens")
        acceptance = (
            f"{accepted / drafted * 100:.1f}%"
            if drafted is not None and accepted is not None and drafted > 0
            else "-"
        )
        if any(
            (root / f"{run_id}_{suffix}").exists()
            for suffix in ("fresh.log", "shared.log", "server.log")
        ):
            print(f"{run_id:<6} {eq:^11} {needle:^6} {acceptance:>10}")

    for mode in MODES:
        d0a = data.get(("D0A", mode))
        d0b = data.get(("D0B", mode))
        if d0a and d0b:
            drift = d0b["points"] - d0a["points"]
            print(f"\n{mode} baseline drift D0B-D0A: {drift:+.2f} points")

    print(
        "\nPortal gate: require equivalence 6/6, needle not worse than D0, "
        "and a repeatable >=15% TPOT gain before spending a submission."
    )


if __name__ == "__main__":
    main()
