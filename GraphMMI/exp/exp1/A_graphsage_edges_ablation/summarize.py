#!/usr/bin/env python3
"""Summarize exp1 results: find 4 ablation runs, print table, write CSV."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

RESULT_DIR = Path(__file__).resolve().parent / "result"
METRICS = ["auc", "aupr", "acc", "f1", "mcc"]
ORDER = {"neither": 0, "mirna_only": 1, "mrna_only": 2, "both": 3}


def read_metrics(run_dir: Path) -> dict[str, float]:
    import pandas as pd

    csv_path = run_dir / "transfer_metrics.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    row = df[(df.source == "human") & (df.target == "human")]
    if row.empty:
        row = df.head(1)
    return {m: float(row[m].iloc[0]) for m in METRICS}


def main() -> None:
    run_dirs = sorted(
        [d.parent for d in RESULT_DIR.glob("*/transfer_metrics.csv")],
        key=lambda d: d.name,
        reverse=True,
    )
    if len(run_dirs) < 4:
        print(f"Found {len(run_dirs)} run dirs, need 4. Run the shell scripts first.")
        sys.exit(1)

    results: list[dict] = []
    for d in run_dirs[:4]:
        cfg_path = d / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        mirna = cfg.get("mirna_sim_edges", False)
        mrna = cfg.get("mrna_sim_edges", False)
        if mirna and mrna:
            label = "both"
        elif mirna:
            label = "mirna_only"
        elif mrna:
            label = "mrna_only"
        else:
            label = "neither"
        m = read_metrics(d)
        m["config"] = label
        results.append(m)

    results.sort(key=lambda r: ORDER.get(r["config"], 99))

    # --- console table ---
    header = f"{'Config':<14}" + "".join(f"{m.upper():>8}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for r in results:
        line = f"{r['config']:<14}" + "".join(f"{r[m]:8.4f}" for m in METRICS)
        print(line)

    baseline = next((r for r in results if r["config"] == "neither"), None)
    if baseline:
        print(f"\n{'Delta vs baseline':<14}", end="")
        for m in METRICS:
            print(f"{'':>8}", end="")
        print()
        for r in results:
            if r["config"] == "neither":
                continue
            print(f"{r['config']:<14}", end="")
            for m in METRICS:
                d = r[m] - baseline[m]
                sign = "+" if d > 0 else ""
                print(f"{sign}{d:7.4f}", end=" ")
            print()

    # --- CSV output ---
    csv_out = RESULT_DIR / "ablation_summary.csv"
    delta_cols = []
    if baseline:
        delta_cols = [f"delta_{m}" for m in METRICS]
    fieldnames = ["config"] + METRICS + delta_cols
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {"config": r["config"]}
            for m in METRICS:
                row[m] = f"{r[m]:.4f}"
            if baseline and r["config"] != "neither":
                for m in METRICS:
                    row[f"delta_{m}"] = f"{r[m] - baseline[m]:+.4f}"
            writer.writerow(row)
    print(f"\nCSV saved → {csv_out}")


if __name__ == "__main__":
    main()
