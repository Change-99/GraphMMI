#!/usr/bin/env bash
# Exp1-B-B: GATv2 + mRNA-mRNA similarity edges only
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/exp/exp1/B_gatv2_edges_ablation/result"
mkdir -p "$RESULT_DIR"

echo "=== Exp1-B-B: GATv2 + mRNA-only ==="
python -u scripts/train_gnn_transfer.py \
  --species human \
  --encoders gatv2 \
  --settings zero_shot \
  --epochs 40 \
  --patience 5 \
  --run-root "$RESULT_DIR" \
  --mrna-sim-edges \
  --no-heatmaps \
  2>&1 | tee "$RESULT_DIR/run_mrna_only.log"

echo "Done."
