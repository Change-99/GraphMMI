#!/usr/bin/env bash
# Run all Exp3 ablations sequentially.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "============================================"
echo " Exp3: all ablations"
echo "============================================"

bash final_exp/exp3/A_edges/run.sh
bash final_exp/exp3/B_encoders/run.sh
bash final_exp/exp3/C_decoder/run.sh
bash final_exp/exp3/D_neg_sample/run.sh

echo ""
echo "Exp3 all done. Results -> final_exp/exp3/*/result"
