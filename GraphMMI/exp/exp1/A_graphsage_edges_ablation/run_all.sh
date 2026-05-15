#!/usr/bin/env bash
# Exp1: Run all 4 ablation configs sequentially
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " Exp1: GraphSAGE Similarity Edge Ablation"
echo " Species: human → human"
echo "============================================"

bash "$SCRIPT_DIR/run_4_neither.sh"
bash "$SCRIPT_DIR/run_1_mirna_only.sh"
bash "$SCRIPT_DIR/run_2_mrna_only.sh"
bash "$SCRIPT_DIR/run_3_both.sh"

echo ""
echo "============================================"
echo " All 4 runs complete."
echo "============================================"

python "$SCRIPT_DIR/summarize.py"
