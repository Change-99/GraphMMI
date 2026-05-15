#!/usr/bin/env bash
# Exp3: GraphSAGE L4 + both-sim + pair-features v1 (baseline)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/exp/exp3/A_graphsage_v1/result"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Exp3: GraphSAGE L4 + both-sim + pair-features v1"
echo "============================================"

python -u scripts/gnn_multi_layers.py \
  --encoder-layer 4 \
  --encoders graphsage \
  --epochs 40 \
  --patience 10 \
  --run-root "$RESULT_DIR" \
  --mirna-sim-edges \
  --mrna-sim-edges

echo ""
echo "Done. Results -> $RESULT_DIR"
