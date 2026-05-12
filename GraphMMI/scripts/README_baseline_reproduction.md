# Baseline ANN/XGBoost Reproduction

This folder keeps the tabular baseline separate from the future GraphMMI GNN
pipeline.

Main script:

```bash
python scripts/baseline_ann_xgb_transfer.py --models ann xgb --transfer-size 500
```

Data-only check, useful before installing TensorFlow/XGBoost:

```bash
python scripts/baseline_ann_xgb_transfer.py --check-data
```

Outputs are written to:

```text
runs/baseline_transfer/<timestamp>/
  metrics_long.csv
  run_summary.json
  matrices/
  heatmaps/
  models/
```

The script reproduces the baseline idea as an isolated tabular task:

- preprocess `data/external/*_pos.csv` and `*_neg.csv`
- merge the 8 datasets into 4 species: human, cow, mouse, worm
- train source-species ANN/XGBoost models
- evaluate source-only cross-species prediction
- fine-tune on a small target-species subset for transfer learning
- write 4x4 source-target matrices and heatmaps

It does not import `TransferLearningMTI`, so baseline experiments stay
independent from the later GraphSAGE/GNN implementation.
