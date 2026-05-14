# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

miRNA-mRNA interaction prediction using GNNs (GraphSAGE, GATv2) and tabular baselines (ANN, XGBoost), with cross-species transfer learning. Species: human, cow, mouse, worm.

## Repo Structure

- [GraphMMI/](GraphMMI/) — **Active pipeline.** Source at `src/graphmmi/`, scripts in `scripts/`. The GNN model uses `GraphMMILinkPredictor` which composes `NodeInputEncoder` → encoder (GraphSAGE or GATv2) → `LinkDecoder`. Negative edges are dynamically sampled during training (not precomputed).
- [gnn/](gnn/) — Older/parallel GraphSAGE experiments with a different codebase (`mti_graphsage` package in `src/`). The UI server imports from here via `from mti_graphsage import GraphSAGENodePairPredictor`.
- [TransferLearningMTI/](TransferLearningMTI/) — Original tabular transfer learning codebase (Jupyter notebooks, separate conda env). Not imported by GraphMMI.
- [ui/](ui/) — FastAPI dashboard for exploring predictions, transfer matrices, and graphs. Serves static files from the UI root.

## Key Commands

### GraphMMI GNN Pipeline (main project)

```bash
# Preprocess: build graph inputs from external CSV data
python scripts/preprocess_graph_data.py

# Smoke test (fast validation)
python scripts/train_gnn_transfer.py --species worm cow --encoders graphsage --settings zero_shot --epochs 1 --neg-ratio 0.2 --eval-neg-ratio 0.2

# Full transfer experiment (GraphSAGE + GATv2, zero-shot + finetune)
python scripts/train_gnn_transfer.py
```

### Baselines

```bash
# Data check only (no TF/XGBoost needed)
python scripts/baseline_ann_xgb_transfer.py --check-data

# Full baseline reproduction
python scripts/baseline_ann_xgb_transfer.py --models ann xgb --transfer-size 500
```

### gnn/ experiments

```bash
# Train single-species GraphSAGE
python scripts/train_graphsage.py --species human --epochs 200 --device auto

# Same with cold_miRNA split
python scripts/train_graphsage.py --species human --data-root data/processed/graphsage_mrna/cold_mirna --epochs 200 --device auto

# Dynamic negative sampling variant
python scripts/train_graphsage_dynamic_neg.py --species human --epochs 200 --device auto

# Cross-species transfer matrix
python scripts/run_species_transfer_matrix.py

# Edge-MLP baseline (test if edge features alone solve the task)
python exp/edge_mlp_baseline.py --species human --epochs 200 --device auto

# GraphSAGE-only baseline (no edge features)
python exp/graphsage_only.py --species human --epochs 200 --device auto
```

### UI

```bash
cd ui && uvicorn server:app --reload
```

### Environment

GraphMMI GNN: `pip install -r GraphMMI/requirements.txt` (PyTorch, numpy, pandas, matplotlib).

Baselines additionally need TensorFlow and XGBoost, or use the original conda env from `TransferLearningMTI/environment.yml`.

## Architecture Notes

### GraphMMI Model (`src/graphmmi/models.py`)

`GraphMMILinkPredictor` is the shared predictor. Only the GNN encoder changes between GraphSAGE and GATv2:

```
NodeInputEncoder (numeric features + trainable ID/type/species embeddings)
  → GraphSAGEEncoder or GATv2Encoder
  → LinkDecoder ([z_src, z_dst, z_src*z_dst, |z_src-z_dst|] → MLP → logit)
```

No dependency on PyTorch Geometric — all layers (MeanSAGE, GATv2) are implemented from scratch with raw scatter ops.

### Data pipeline (`src/graphmmi/data.py`)

- `GraphBundle` holds x (node features), edge_index (train-positive only), node_type, species_id, split-positive edges
- Negative edges are sampled dynamically via `sample_negative_edges()` using endpoint corruption, excluding all known positives across all splits
- Pair features (sequence-derived: log lengths, GC, seed matches) are computed on-the-fly for both positive and negative edges — this avoids label leakage

### Transfer experiment flow ([train_gnn_transfer.py](GraphMMI/scripts/train_gnn_transfer.py))

1. Train source model on each species
2. For each (source, target) pair: evaluate zero-shot (load weights, freeze embeddings) or finetune on target with reduced epochs
3. Select classification threshold on target validation split before reporting test metrics
4. Output CSV, JSON, and heatmaps

### Leakage safety

Training graph edges are train-positive only. Validation/test positives are never used for message passing. Negative edges are sampled so they never overlap with any known positive pair (including val/test).

### gnn/ vs GraphMMI/

These are separate codebases. `gnn/src/mti_graphsage/` has its own `GraphSAGENodePairPredictor` and `GraphSAGEEncoder`. The UI server imports from `gnn/`. GraphMMI is the newer, unified pipeline that supports both GraphSAGE and GATv2 with a shared architecture.

## Data Layout

- External raw data: `data/external/*_pos.csv`, `*_neg.csv` (not in repo)
- GraphMMI processed: `data/processed/graph/random/<species>/graph_inputs.pt`
- gnn processed: `data/processed/graphsage_mrna/<random|cold_mirna>/<species>/`
- Runs output: `runs/gnn_transfer/<timestamp>/`, `runs/baseline_transfer/<timestamp>/`
