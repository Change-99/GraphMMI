#!/usr/bin/env bash
# Exp1-B-A: GATv2 + miRNA-miRNA similarity edges only
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/exp/exp1/B_gatv2_edges_ablation/result"
mkdir -p "$RESULT_DIR"

echo "=== Exp1-B-A: GATv2 + miRNA-only ==="
python -u scripts/train_gnn_transfer.py \
  --species human \
  --encoders gatv2 \
  --settings zero_shot \
  --epochs 20 \
  --patience 5 \
  --run-root "$RESULT_DIR" \
  --mirna-sim-edges \
  --no-heatmaps \
  2>&1 | tee "$RESULT_DIR/run_mirna_only.log"

echo "Done."
