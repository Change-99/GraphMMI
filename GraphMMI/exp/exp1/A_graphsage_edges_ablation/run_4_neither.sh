#!/usr/bin/env bash
# Exp1-D: GraphSAGE baseline (no similarity edges)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/exp/exp1/A_graphsage_edges_ablation/result"
mkdir -p "$RESULT_DIR"

echo "=== Exp1-D: GraphSAGE baseline (no sim) ==="
python -u scripts/train_gnn_transfer.py \
  --species human \
  --encoders graphsage \
  --settings zero_shot \
  --epochs 20 \
  --patience 5 \
  --run-root "$RESULT_DIR" \
  --no-heatmaps \
  2>&1 | tee "$RESULT_DIR/run_neither.log"

echo "Done."
