#!/usr/bin/env bash
# ============================================================
# Exp3-D: Negative-sampling ablation
#
# Backbone: 4-layer GraphSAGE, target_site graph, both similarity edges.
# Strategies:
#   endpoint_corrupt
#   degree_aware
#   sequence_aware
#
# Validation/test negatives are fixed to endpoint_corrupt so this ablation
# compares training negative-sampling strategies under the same test set.
#
# Species: human only.
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

DATA_DIR="$ROOT/data/processed/graph/final_target_site"
RESULT_DIR="$ROOT/final_exp/exp3/D_neg_sample/result"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Exp3-D: negative sampling ablation"
echo " Data:    $DATA_DIR"
echo " Results: $RESULT_DIR"
echo "============================================"

if [[ ! -f "$DATA_DIR/human/graph_inputs.pt" ]]; then
  echo "[setup] target_site graph data not found; building..."
  python -u scripts/final_embedding.py \
    --node-mode target_site \
    --sim-mode topk \
    --mirna-sim-topk 5 \
    --mrna-sim-topk 5 \
    --output-dir "$DATA_DIR"
fi

COMMON_ARGS=(
  --species human
  --encoders graphsage
  --settings strict_zero_shot finetune
  --epochs 40
  --patience 8
  --finetune-epochs 15
  --finetune-patience 5
  --num-layers 4
  --graphsage-hidden-dim 128
  --processed-dir "$DATA_DIR"
  --mirna-sim-edges --mrna-sim-edges
  --skip-preprocess
  --run-root "$RESULT_DIR"
  --no-heatmaps
)

for strategy in endpoint_corrupt degree_aware sequence_aware; do
  echo ""
  echo "[negative strategy=$strategy]"
  python -u scripts/train_gnn_transfer.py \
    "${COMMON_ARGS[@]}" \
    --neg-strategy "$strategy" \
    --eval-neg-strategy endpoint_corrupt \
    --refresh-fixed-negatives \
    2>&1 | tee "$RESULT_DIR/${strategy}.log"
done

echo ""
echo "Exp3-D done. Results -> $RESULT_DIR"
