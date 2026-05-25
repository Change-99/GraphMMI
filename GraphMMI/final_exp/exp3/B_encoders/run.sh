#!/usr/bin/env bash
# ============================================================
# Exp3-B: Encoder depth ablation
#
# Test GraphSAGE and GATv2 with 1..6 GNN layers.
# Uses target_site graph with both similarity-edge families.
#
# Species: human only.
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

DATA_DIR="$ROOT/data/processed/graph/final_target_site"
RESULT_DIR="$ROOT/final_exp/exp3/B_encoders/result"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Exp3-B: encoder/layer ablation"
echo " Data:    $DATA_DIR"
echo " Results: $RESULT_DIR"
echo "============================================"

SPECIES=(human)
echo "Species: human"

missing_data=0
for sp in "${SPECIES[@]}"; do
  if [[ ! -f "$DATA_DIR/$sp/graph_inputs.pt" ]]; then
    missing_data=1
  fi
done

if [[ "$missing_data" -eq 1 ]]; then
  echo "[setup] target_site graph data not found; building..."
  python -u scripts/final_embedding.py \
    --node-mode target_site \
    --sim-mode topk \
    --mirna-sim-topk 5 \
    --mrna-sim-topk 5 \
    --output-dir "$DATA_DIR"
fi

COMMON_ARGS=(
  --species "${SPECIES[@]}"
  --settings strict_zero_shot
  --epochs 40
  --patience 8
  --finetune-epochs 15
  --finetune-patience 5
  --processed-dir "$DATA_DIR"
  --mirna-sim-edges --mrna-sim-edges
  --skip-preprocess
  --run-root "$RESULT_DIR"
  --no-heatmaps
)

for encoder in graphsage gatv2; do
  for layers in 1 2 3 4 5 6; do
    echo ""
    echo "[encoder=$encoder layers=$layers]"
    extra_args=()
    if [[ "$encoder" == "graphsage" ]]; then
      extra_args=(--graphsage-hidden-dim 128)
    else
      extra_args=(--gatv2-hidden-dim 64)
    fi
    python -u scripts/train_gnn_transfer.py \
      "${COMMON_ARGS[@]}" \
      --encoders "$encoder" \
      --num-layers "$layers" \
      "${extra_args[@]}" \
      2>&1 | tee "$RESULT_DIR/${encoder}_L${layers}.log"
  done
done

echo ""
echo "Exp3-B done. Results -> $RESULT_DIR"
