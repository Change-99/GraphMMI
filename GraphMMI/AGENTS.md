# Repository Guidelines

## Project Structure & Module Organization

GraphMMI is a Python research codebase for miRNA-mRNA interaction prediction. Core reusable code lives in `src/graphmmi/`: `data.py` loads graph bundles and sampling utilities, and `models.py` defines PyTorch encoders and decoders. Runnable workflows live in `scripts/`, including graph preprocessing, GNN transfer training, baseline ANN/XGBoost reproduction, and summaries. Experiment launchers and result summaries are grouped under `exp/exp*/`. Raw and derived datasets are under `data/`; generated training outputs go to `runs/`. Avoid committing new large run artifacts unless they are intentional experiment records.

## Build, Test, and Development Commands

Create an environment and install dependencies:

```bash
conda create -n graphmmi python=3.10
conda activate graphmmi
pip install -r requirements.txt
```

Build graph inputs from `data/external/*_pos.csv`:

```bash
python scripts/preprocess_graph_data.py
```

Run a fast GNN smoke test:

```bash
python scripts/train_gnn_transfer.py --species worm cow --encoders graphsage --epochs 1 --finetune-epochs 1 --neg-ratio 0.2 --eval-neg-ratio 0.2 --run-root runs/gnn_transfer_smoke
```

Run the default GNN experiment:

```bash
python scripts/train_gnn_transfer.py
```

Check baseline data without heavy optional dependencies:

```bash
python scripts/baseline_ann_xgb_transfer.py --check-data
```

## Coding Style & Naming Conventions

Use Python 3.10+ and follow the existing style: 4-space indentation, type hints for public helpers, `dataclass` containers for graph data, and `pathlib.Path` for filesystem paths. Keep tensor variables concise (`x`, `edge_index`, `edge_attr`) where they match PyTorch conventions, and use descriptive names for experiment flags and caches. Prefer small helper functions in `src/graphmmi/` and keep script orchestration inside `scripts/`.

## Testing Guidelines

There is no dedicated pytest suite. Validate changes with the smallest command that covers the touched path: run preprocessing after data-pipeline edits, the one-epoch GNN smoke test after model/training edits, and `--check-data` after baseline data changes. For experiment scripts, confirm that `config.json`, `transfer_metrics.csv`, and `transfer_metrics.json` are written under the expected `runs/` or `exp/.../result/` directory.

## Commit & Pull Request Guidelines

Recent commits use short imperative summaries such as `Add pair feature v2...` and `Optimize GNN transfer training memory usage`. Keep commit subjects concise, start with a verb, and mention the affected pipeline when useful. Pull requests should include the motivation, commands run, key metric changes or generated output paths, and any data assumptions. Add screenshots only for heatmap or plot changes.

## Security & Configuration Tips

Do not hard-code machine-specific absolute paths, credentials, or private data locations. Keep generated caches and run outputs reproducible from scripts, and document any new external data requirement in the relevant README under `scripts/`.
