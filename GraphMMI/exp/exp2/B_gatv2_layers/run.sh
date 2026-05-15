#!/usr/bin/env bash
# Exp2-B: GATv2 layer-depth ablation (1 / 2 / 3) with both sim edges
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/exp/exp2/B_gatv2_layers/result"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Exp2-B: GATv2 Layer Depth"
echo " Layers: 1 2 3"
echo " Species: human -> human"
echo "============================================"

python -u scripts/gnn_multi_layers.py \
  --encoder-layer 1 2 3 \
  --encoders gatv2 \
  --epochs 30 \
  --patience 8 \
  --run-root "$RESULT_DIR" \
  --mirna-sim-edges 

echo ""
echo "Done. Results -> $RESULT_DIR"
