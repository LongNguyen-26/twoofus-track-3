"""Summarize a results_w4_* directory produced by w4_battery_r2.sh."""

import argparse
import re
from pathlib import Path


RUN_IDS = ("FP8A", "BNB4A", "BNB4B", "FP8B")
BASELINE_IDS = ("FP8A", "FP8B")
CANDIDATE_IDS = ("BNB4A", "BNB4B")
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


def read_run(path: Path):
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


def read_fraction(path: Path, label: str):
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"{re.escape(label)}\s+(\d+)/(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def format_fraction(value):
    return f"{value[0]}/{value[1]}" if value is not None else "-"


def read_startup(root: Path, run_id: str):
    path = root / f"{run_id}_startup_seconds.txt"
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def average(data, ids, mode, field):
    values = [
        data[(run_id, mode)][field]
        for run_id in ids
        if data.get((run_id, mode)) is not None
    ]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    root = args.results_dir

    data = {
        (run_id, mode): read_run(root / f"{run_id}_{mode}.log")
        for run_id in RUN_IDS
        for mode in MODES
    }

    print("W4 BitsAndBytes battery summary")
    print(f"results: {root}")
    print()
    print(
        "config mode    points   delta  ttft50 ttft95  "
        "tpot50  tpot_gain  errors  startup"
    )
    print(
        "------ ------  -------  ------  ------ ------  "
        "------  ---------  ------  -------"
    )
    for run_id in RUN_IDS:
        for mode in MODES:
            row = data[(run_id, mode)]
            if row is None:
                continue
            base_points = average(data, BASELINE_IDS, mode, "points")
            base_tpot = average(data, BASELINE_IDS, mode, "tpot_p50")
            delta = row["points"] - base_points if base_points is not None else 0.0
            gain = (
                (base_tpot - row["tpot_p50"]) / base_tpot * 100
                if base_tpot
                else 0.0
            )
            startup = read_startup(root, run_id)
            startup_text = f"{startup}s" if startup is not None else "-"
            print(
                f"{run_id:<6} {mode:<6}  {row['points']:7.2f} "
                f"{delta:+7.2f}  {row['ttft_p50']:6.0f} "
                f"{row['ttft_p95']:6.0f}  {row['tpot_p50']:6.2f} "
                f"{gain:+8.1f}%  {row['errors']:2d}/{row['requests']:<3d} "
                f"{startup_text:>7}"
            )

    print()
    print("config self_equiv cross_fp8 needle")
    print("------ ---------- ---------- ------")
    for run_id in RUN_IDS:
        self_eq = read_fraction(root / f"{run_id}_equivalence.log", "EQUIVALENCE")
        cross = read_fraction(root / f"{run_id}_cross_fp8.log", "EQUIVALENCE")
        needle = read_fraction(root / f"{run_id}_needle.log", "RETRIEVAL")
        if any(
            (root / f"{run_id}_{suffix}").exists()
            for suffix in ("fresh.log", "shared.log", "server.log")
        ):
            print(
                f"{run_id:<6} {format_fraction(self_eq):^10} "
                f"{format_fraction(cross):^10} {format_fraction(needle):^6}"
            )

    for mode in MODES:
        first = data.get(("FP8A", mode))
        last = data.get(("FP8B", mode))
        if first and last:
            print(
                f"\n{mode} FP8 bracket drift FP8B-FP8A: "
                f"{last['points'] - first['points']:+.2f} points, "
                f"{last['tpot_p50'] - first['tpot_p50']:+.2f} ms TPOT"
            )
        cand_a = data.get(("BNB4A", mode))
        cand_b = data.get(("BNB4B", mode))
        if cand_a and cand_b:
            print(
                f"{mode} BNB4 restart drift BNB4B-BNB4A: "
                f"{cand_b['points'] - cand_a['points']:+.2f} points, "
                f"{cand_b['tpot_p50'] - cand_a['tpot_p50']:+.2f} ms TPOT"
            )

    required_rows = [
        data.get((run_id, "fresh"))
        for run_id in (*BASELINE_IDS, *CANDIDATE_IDS)
    ]
    if any(row is None for row in required_rows):
        print(
            "\nVERDICT: INCOMPLETE. Run the full "
            "FP8A -> BNB4A -> BNB4B -> FP8B bracket."
        )
        return

    base_tpot = average(data, BASELINE_IDS, "fresh", "tpot_p50")
    base_points = average(data, BASELINE_IDS, "fresh", "points")
    base_needles = [
        read_fraction(root / f"{run_id}_needle.log", "RETRIEVAL")
        for run_id in BASELINE_IDS
    ]
    candidate_needles = [
        read_fraction(root / f"{run_id}_needle.log", "RETRIEVAL")
        for run_id in CANDIDATE_IDS
    ]
    bnb_restart_eq = read_fraction(
        root / "BNB4B_equivalence.log", "EQUIVALENCE"
    )

    speed_ok = all(
        (base_tpot - data[(run_id, "fresh")]["tpot_p50"]) / base_tpot
        >= 0.15
        for run_id in CANDIDATE_IDS
    )
    points_ok = all(
        data[(run_id, "fresh")]["points"] > base_points
        for run_id in CANDIDATE_IDS
    )
    errors_ok = all(
        data[(run_id, "fresh")]["errors"] == 0
        for run_id in CANDIDATE_IDS
    )
    startup_ok = all(
        read_startup(root, run_id) is not None
        and read_startup(root, run_id) < 600
        for run_id in CANDIDATE_IDS
    )
    repeatable_ok = (
        bnb_restart_eq is not None
        and bnb_restart_eq[1] > 0
        and bnb_restart_eq[0] == bnb_restart_eq[1]
    )
    needles_present = all(
        value is not None for value in base_needles + candidate_needles
    )
    needle_ok = needles_present and min(
        value[0] for value in candidate_needles
    ) >= min(value[0] for value in base_needles)

    checks = {
        "repeatable >=15% TPOT gain": speed_ok,
        "ERS above bracket baseline": points_ok,
        "zero replay errors": errors_ok,
        "startup below 600s": startup_ok,
        "BNB4 deterministic across restart": repeatable_ok,
        "needle not worse than FP8": needle_ok,
    }
    print("\nLocal candidate gates:")
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {label}")

    if all(checks.values()):
        print(
            "\nVERDICT: LOCAL PASS. Run BF16-vs-BNB4 GPQA next; "
            "do not submit before the accuracy gate passes."
        )
    else:
        print(
            "\nVERDICT: LOCAL REJECT. Do not spend a portal submission "
            "or package a custom image for this configuration."
        )


if __name__ == "__main__":
    main()
