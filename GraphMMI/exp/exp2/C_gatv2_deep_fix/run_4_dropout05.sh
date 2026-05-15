#!/usr/bin/env bash
# C4: GATv2 + residual + layernorm + dropout 0.5 — layers 1 2 3
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/exp/exp2/C_gatv2_deep_fix/result"
mkdir -p "$RESULT_DIR"

echo "=== C4: GATv2 + residual + layernorm + dropout=0.5 ==="
python -u scripts/gnn_multi_layers.py \
  --encoder-layer 4 5 6 \
  --encoders gatv2 \
  --epochs 200 \
  --patience 8 \
  --run-root "$RESULT_DIR" \
  --mirna-sim-edges \
  --residual \
  --layer-norm \
  --dropout 0.5

echo "Done."
