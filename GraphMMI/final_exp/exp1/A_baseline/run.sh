#!/usr/bin/env bash
# ============================================================
# Exp1: ANN + XGBoost baseline reproduction (4-species transfer)
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/final_exp/exp1/result"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Exp1: ANN + XGBoost Baseline (4-species)"
echo "============================================"
echo "Result dir: $RESULT_DIR"
echo ""

# ---- ANN baseline ----
echo "[1/2] Running ANN baseline..."
python -u scripts/baseline_ann_xgb_transfer.py \
  --models ann \
  --epochs 100 \
  --transfer-size 500 \
  --seed 42 \
  --skip-preprocess \
  --run-root "$RESULT_DIR" \
  2>&1 | tee "$RESULT_DIR/ann_run.log"

echo ""

# ---- XGBoost baseline ----
echo "[2/2] Running XGBoost baseline..."
python -u scripts/baseline_ann_xgb_transfer.py \
  --models xgb \
  --xgb-n-estimators 100 \
  --transfer-size 500 \
  --seed 42 \
  --skip-preprocess \
  --run-root "$RESULT_DIR" \
  2>&1 | tee "$RESULT_DIR/xgb_run.log"

echo ""
echo "============================================"
echo " Exp1 done."
echo " Results -> $RESULT_DIR"
echo "============================================"
