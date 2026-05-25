#!/usr/bin/env bash
# ============================================================
# Exp1-B: mRNA-level (coarse) GNN — GraphSAGE + GATv2
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

RESULT_DIR="$ROOT/final_exp/exp1/B_mrna_gnn/result"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Exp1-B: mRNA-level GNN (coarse graph)"
echo "============================================"
echo "Result dir: $RESULT_DIR"
echo ""

# ---- Step 1: Build mRNA graph data ----
echo "[1/3] Building mRNA graph data..."
python -u scripts/final_embedding.py \
  --node-mode mrna \
  --sim-mode topk \
  --output-dir "$ROOT/data/processed/graph/final_mrna"

echo ""

# ---- Step 2: GraphSAGE L4 ----
echo "[2/3] Running GraphSAGE L4..."
python -u scripts/train_gnn_transfer.py \
  --species human cow mouse worm \
  --encoders graphsage \
  --settings strict_zero_shot calibrated_zero_shot \
  --epochs 40 --patience 8 \
  --num-layers 4 \
  --graphsage-hidden-dim 128 \
  --processed-dir "$ROOT/data/processed/graph/random" \
  --mirna-sim-edges --mrna-sim-edges \
  --skip-preprocess --refresh-fixed-negatives \
  --run-root "$RESULT_DIR" \
  2>&1 | tee "$RESULT_DIR/graphsage_run.log"

echo ""

# ---- Step 3: GATv2 L1 ----
echo "[3/3] Running GATv2 L1..."
python -u scripts/train_gnn_transfer.py \
  --species human cow mouse worm \
  --encoders gatv2 \
  --settings strict_zero_shot calibrated_zero_shot \
  --epochs 40 --patience 8 \
  --num-layers 1 \
  --gatv2-hidden-dim 64 \
  --processed-dir "$ROOT/data/processed/graph/random" \
  --mirna-sim-edges --mrna-sim-edges \
  --skip-preprocess --refresh-fixed-negatives \
  --run-root "$RESULT_DIR" \
  2>&1 | tee "$RESULT_DIR/gatv2_run.log"

echo ""
echo "============================================"
echo " Exp1-B done."
echo " Results -> $RESULT_DIR"
echo "============================================"
