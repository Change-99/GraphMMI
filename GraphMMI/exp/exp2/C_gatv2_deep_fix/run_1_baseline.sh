#!/usr/bin/env bash
# C1: GATv2 baseline (原始对照) — layers 1 2 3
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/exp/exp2/C_gatv2_deep_fix/result"
mkdir -p "$RESULT_DIR"

echo "=== C1: GATv2 baseline (no residual, no layernorm) ==="
python -u scripts/gnn_multi_layers.py \
  --encoder-layer 1 2 3 \
  --encoders gatv2 \
  --epochs 30 \
  --patience 8 \
  --run-root "$RESULT_DIR" \
  --mirna-sim-edges

echo "Done."
