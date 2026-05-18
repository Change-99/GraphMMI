#!/usr/bin/env bash
# ============================================================
# Exp2: target_site-level GNN — GraphSAGE + GATv2
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/final_exp/exp2/result"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Exp2: target_site-level GNN (fine graph)"
echo "============================================"
echo "Result dir: $RESULT_DIR"
echo ""

# ---- Step 1: Build target_site graph data ----
echo "[1/3] Building target_site graph data..."
python -u scripts/final_embedding.py \
  --node-mode target_site \
  --sim-mode topk \
  --mirna-sim-topk 5 \
  --mrna-sim-topk 5 \
  --output-dir "$ROOT/data/processed/graph/final_target_site"

echo ""

# ---- Step 2: GraphSAGE L4 ----
echo "[2/3] Running GraphSAGE L4 (target_site)..."
python -u scripts/train_gnn_transfer.py \
  --species human  cow mouse worm\
  --encoders graphsage \
  --settings strict_zero_shot finetune \
  --epochs 40 --patience 8 \
  --num-layers 4 \
  --graphsage-hidden-dim 128 \
  --processed-dir "$ROOT/data/processed/graph/final_target_site" \
  --mirna-sim-edges --mrna-sim-edges \
  --skip-preprocess --refresh-fixed-negatives \
  --run-root "$RESULT_DIR" \
  --no-heatmaps \
  2>&1 | tee "$RESULT_DIR/graphsage_run.log"

echo ""

# ---- Step 3: GATv2 L1 ----
echo "[3/3] Running GATv2 L1 (target_site)..."
python -u scripts/train_gnn_transfer.py \
  --species human  cow mouse worm\
  --encoders gatv2 \
  --settings strict_zero_shot finetune \
  --epochs 40 --patience 8 \
  --num-layers 1 \
  --gatv2-hidden-dim 64 \
  --processed-dir "$ROOT/data/processed/graph/final_target_site" \
  --mirna-sim-edges --mrna-sim-edges \
  --skip-preprocess --refresh-fixed-negatives \
  --run-root "$RESULT_DIR" \
  --no-heatmaps \
  2>&1 | tee "$RESULT_DIR/gatv2_run.log"

echo ""
echo "============================================"
echo " Exp2 done."
echo " Results -> $RESULT_DIR"
echo "============================================"
