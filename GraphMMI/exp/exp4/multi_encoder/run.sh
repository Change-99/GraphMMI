#!/usr/bin/env bash
# Exp4: Layer-wise Relation-Aware GNN
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/exp/exp4/multi_encoder/result"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Exp4: Layer-wise Relation-Aware GNN"
echo "============================================"

python -u scripts/multi_encoders.py \
  --species human \
  --epochs 60 \
  --patience 12 \
  --lr 1e-3 \
  --weight-decay 1e-4 \
  --dropout 0.3 \
  --mirna-sim-edges \
  --mrna-sim-edges \
  --run-root "$RESULT_DIR" \
  2>&1 | tee "$RESULT_DIR/run.log"

echo ""
echo "Done. Results -> $RESULT_DIR"
