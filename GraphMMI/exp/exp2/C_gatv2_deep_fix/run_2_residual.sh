#!/usr/bin/env bash
# C2: GATv2 + residual — layers 1 2 3
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/exp/exp2/C_gatv2_deep_fix/result"
mkdir -p "$RESULT_DIR"

echo "=== C2: GATv2 + residual ==="
python -u scripts/gnn_multi_layers.py \
  --encoder-layer 1 \
  --encoders gatv2 \
  --epochs 100 \
  --patience 8 \
  --run-root "$RESULT_DIR" \
  --mirna-sim-edges \
  --residual

echo "Done."
