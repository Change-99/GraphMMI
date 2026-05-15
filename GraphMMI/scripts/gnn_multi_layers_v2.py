#!/usr/bin/env python3
"""GNN encoder layer-depth ablation — human→human, single-species.

Takes a list of layer counts and runs each configuration sequentially,
producing a side-by-side metrics table.

Usage:
  python scripts/gnn_multi_layers.py \\
      --encoder-layer 1 2 3 \\
      --encoders graphsage \\
      --epochs 20 --patience 5 \\
      --run-root exp/exp2/result

Optional similarity edges:
  --mirna-sim-edges --mrna-sim-edges
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECIES = ["human"]
METRICS = ["auc", "aupr", "acc", "f1", "mcc"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GNN layer-depth ablation")
    p.add_argument("--encoder-layer", type=int, nargs="+", default=[1, 2, 3],
                   help="List of GNN layer counts to test (default: 1 2 3)")
    p.add_argument("--encoders", nargs="+", default=["graphsage", "gatv2"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--mirna-sim-edges", action="store_true")
    p.add_argument("--mrna-sim-edges", action="store_true")
    p.add_argument("--residual", action="store_true")
    p.add_argument("--layer-norm", action="store_true")
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--no-heatmaps", action="store_true", default=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.run_root
    result_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for encoder in args.encoders:
        for n_layers in args.encoder_layer:
            tag = f"{encoder}_L{n_layers}"
            print(f"\n{'='*50}")
            print(f"  {tag}  (human->human)")
            print(f"{'='*50}")

            cmd = [
                sys.executable, "-u", str(ROOT / "scripts/train_gnn_transfer_v2.py"),
                "--species", "human",
                "--encoders", encoder,
                "--settings", "zero_shot",
                "--epochs", str(args.epochs),
                "--patience", str(args.patience),
                "--num-layers", str(n_layers),
                "--run-root", str(result_dir),
                "--no-heatmaps",
            ]
            if args.mirna_sim_edges:
                cmd.append("--mirna-sim-edges")
            if args.mrna_sim_edges:
                cmd.append("--mrna-sim-edges")
            if args.residual:
                cmd.append("--residual")
            if args.layer_norm:
                cmd.append("--layer-norm")
            if args.dropout is not None:
                cmd.extend(["--dropout", str(args.dropout)])

            started = time.time()
            subprocess.run(cmd, cwd=ROOT, check=True)
            elapsed = time.time() - started

            # find the just-created run dir
            run_dirs = sorted(result_dir.glob("*/transfer_metrics.csv"),
                              key=lambda p: p.stat().st_mtime, reverse=True)
            csv_path = run_dirs[0] if run_dirs else None
            if csv_path is None:
                print(f"  WARNING: no transfer_metrics.csv found")
                continue

            import pandas as pd
            df = pd.read_csv(csv_path)
            row = df[(df.source == "human") & (df.target == "human")]
            if row.empty:
                row = df.head(1)
            r = row.iloc[0]

            rows.append({
                "encoder": encoder,
                "layers": n_layers,
                "tag": tag,
                "auc": float(r["auc"]),
                "aupr": float(r["aupr"]),
                "acc": float(r["acc"]),
                "f1": float(r["f1"]),
                "mcc": float(r["mcc"]),
                "loss": float(r.get("loss", float("nan"))),
                "thr": float(r.get("selected_threshold", float("nan"))),
                "epoch": int(r.get("source_best_epoch", 0)),
                "elapsed_s": int(elapsed),
            })
            print(f"  {tag}: AUC={r['auc']:.4f} AUPR={r['aupr']:.4f} "
                  f"F1={r['f1']:.4f} ({elapsed:.0f}s)")

    # --- summary ---
    if not rows:
        print("No results collected.")
        return

    print(f"\n{'='*70}")
    print("  LAYER-DEPTH ABLATION SUMMARY")
    print(f"{'='*70}")

    header = f"{'Config':<20}" + "".join(f"{m.upper():>8}" for m in METRICS) + f"  {'Epoch':>5}  {'Time':>6}"
    print(header)
    print("-" * len(header))
    for r in rows:
        line = f"{r['tag']:<20}" + "".join(f"{r[m]:8.4f}" for m in METRICS)
        line += f"  {r['epoch']:>5}  {r['elapsed_s']:>5}s"
        print(line)

    # CSV output
    csv_out = result_dir / "layer_ablation_summary.csv"
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["tag", "encoder", "layers"] + METRICS + ["loss", "thr", "epoch", "elapsed_s"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved -> {csv_out}")

    # best by AUPR
    rows.sort(key=lambda r: r["aupr"], reverse=True)
    best = rows[0]
    print(f"Best by AUPR: {best['tag']}  AUPR={best['aupr']:.4f}  AUC={best['auc']:.4f}")


if __name__ == "__main__":
    main()
