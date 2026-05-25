#!/usr/bin/env bash
# ============================================================
# Exp3-A: Similarity-edge ablation
#
# Best backbone: 4-layer GraphSAGE on target_site graph.
# Variants:
#   1. no-sim      : interaction edges only
#   2. miRNA-only  : interaction + miRNA-miRNA similarity edges
#   3. target-only : interaction + target-site similarity edges
#   4. both-sim    : interaction + miRNA + target-site similarity edges
#
# Species: human, cow only.
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

DATA_DIR="$ROOT/data/processed/graph/final_target_site"
RESULT_DIR="$ROOT/final_exp/exp3/A_edges/result"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Exp3-A: edge ablation, GraphSAGE L4"
echo " Data:    $DATA_DIR"
echo " Results: $RESULT_DIR"
echo "============================================"

if [[ ! -f "$DATA_DIR/human/graph_inputs.pt" || ! -f "$DATA_DIR/cow/graph_inputs.pt" ]]; then
  echo "[setup] target_site graph data not found; building..."
  python -u scripts/final_embedding.py \
    --node-mode target_site \
    --sim-mode topk \
    --mirna-sim-topk 5 \
    --mrna-sim-topk 5 \
    --output-dir "$DATA_DIR"
fi

COMMON_ARGS=(
  --species human cow
  --encoders graphsage
  --settings finetune
  --epochs 40
  --patience 8
  --finetune-epochs 15
  --finetune-patience 5
  --num-layers 4
  --graphsage-hidden-dim 128
  --processed-dir "$DATA_DIR"
  --skip-preprocess
  --run-root "$RESULT_DIR"
  --no-heatmaps
)

echo ""
echo "[1/4] no-sim"
python -u scripts/train_gnn_transfer.py \
  "${COMMON_ARGS[@]}" \
  --refresh-fixed-negatives \
  2>&1 | tee "$RESULT_DIR/no_sim.log"

echo ""
echo "[2/4] miRNA-only"
python -u scripts/train_gnn_transfer.py \
  "${COMMON_ARGS[@]}" \
  --mirna-sim-edges \
  2>&1 | tee "$RESULT_DIR/mirna_only.log"

echo ""
echo "[3/4] target-only"
python -u scripts/train_gnn_transfer.py \
  "${COMMON_ARGS[@]}" \
  --mrna-sim-edges \
  2>&1 | tee "$RESULT_DIR/target_only.log"

echo ""
echo "[4/4] both-sim"
python -u scripts/train_gnn_transfer.py \
  "${COMMON_ARGS[@]}" \
  --mirna-sim-edges --mrna-sim-edges \
  2>&1 | tee "$RESULT_DIR/both_sim.log"

echo ""
echo "Exp3-A done. Results -> $RESULT_DIR"
