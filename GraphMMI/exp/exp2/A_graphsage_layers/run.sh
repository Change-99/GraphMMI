#!/usr/bin/env bash
# Exp2-A: GraphSAGE layer-depth ablation (1 / 2 / 3) with both sim edges
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/exp/exp2/A_graphsage_layers/result"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Exp2-A: GraphSAGE Layer Depth"
echo " Layers: 1 2 3"
echo " Species: human -> human"
echo "============================================"

python -u scripts/gnn_multi_layers.py \
  --encoder-layer 1 2 3 4 5 6 \
  --encoders graphsage \
  --epochs 40 \
  --patience 8 \
  --run-root "$RESULT_DIR" \
  --mirna-sim-edges \
  --mrna-sim-edges

echo ""
echo "Done. Results -> $RESULT_DIR"
