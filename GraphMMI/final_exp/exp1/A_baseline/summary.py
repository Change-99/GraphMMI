#!/usr/bin/env python3
"""Compute summary statistics from baseline CSV result files."""
import sys
from pathlib import Path
import pandas as pd

RESULT_DIR = Path(__file__).resolve().parent / "result"

# Find all timestamp directories containing metrics_long.csv
dirs = sorted([d for d in RESULT_DIR.iterdir() if d.is_dir() and (d / "metrics_long.csv").exists()])
print(f"Found {len(dirs)} result directories: {[d.name for d in dirs]}", flush=True)

if not dirs:
    print("ERROR: no result directories found in", RESULT_DIR, flush=True)
    sys.exit(1)

all_metrics = ["aupr", "auc", "f1", "mcc"]

def summarize(csv_path: Path):
    df = pd.read_csv(csv_path)
    for model in ["ann", "xgb"]:
        for protocol in ["source_only", "transfer"]:
            sub = df[(df["model"] == model) & (df["protocol"] == protocol)]
            if len(sub) == 0:
                continue
            all_mean = {m: sub[m].mean() for m in all_metrics}
            diag = sub[sub["source"] == sub["target"]]
            diag_mean = {m: diag[m].mean() for m in all_metrics}
            off = sub[sub["source"] != sub["target"]]
            off_mean = {m: off[m].mean() for m in all_metrics}

            print(f"  [{model:>4s}] {protocol:>12s}  "
                  f"AUPR all={all_mean['aupr']:.4f}  diag={diag_mean['aupr']:.4f}  off={off_mean['aupr']:.4f}  "
                  f"AUC all={all_mean['auc']:.4f}  "
                  f"F1 all={all_mean['f1']:.4f}  "
                  f"MCC all={all_mean['mcc']:.4f}", flush=True)

for d in dirs:
    csv = d / "metrics_long.csv"
    print(f"\n=== {d.name} ===", flush=True)
    summarize(csv)
