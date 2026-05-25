#!/usr/bin/env bash
# ============================================================
# Exp3-C: Decoder architecture ablation
#
# Backbone: 4-layer GraphSAGE, target_site graph, both similarity edges.
# Decoders:
#   baseline  : concat + MLP
#   residual  : residual MLP
#   gated     : gated pair-feature fusion
#   bilinear  : bilinear matching term
#   separated : node and pair features encoded separately, then fused
#
# Species: human only.
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

DATA_DIR="$ROOT/data/processed/graph/final_target_site"
RESULT_DIR="$ROOT/final_exp/exp3/C_decoder/result"
mkdir -p "$RESULT_DIR"

echo "============================================"
echo " Exp3-C: decoder ablation"
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

python -u scripts/decoder_optimed.py \
  --species human \
  --decoders baseline residual gated bilinear separated \
  --epochs 40 \
  --patience 8 \
  --num-layers 4 \
  --hidden-dim 128 \
  --processed-dir "$DATA_DIR" \
  --mirna-sim-edges --mrna-sim-edges \
  --run-root "$RESULT_DIR" \
  2>&1 | tee "$RESULT_DIR/decoder_ablation.log"

echo ""
echo "Exp3-C done. Results -> $RESULT_DIR"
