#!/usr/bin/env bash
# C: GATv2 deep-layer fix — run all 4 configs sequentially
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo " Exp2-C: GATv2 Deep-Layer Fix"
echo " Layers: 1 2 3"
echo "============================================"

bash "$SCRIPT_DIR/run_1_baseline.sh"
bash "$SCRIPT_DIR/run_2_residual.sh"
bash "$SCRIPT_DIR/run_3_residual_ln.sh"
bash "$SCRIPT_DIR/run_4_dropout05.sh"

echo ""
echo "============================================"
echo " All 4 runs complete."
echo " Compare result/layer_ablation_summary.csv"
echo "============================================"
