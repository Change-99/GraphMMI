# GraphMMI Graph Data Pipeline

This pipeline builds leakage-safe graph inputs for GraphSAGE and GATv2.

## Data Rule

Only `*_pos.csv` files are used.

```text
positive interactions -> real graph edges
negative interactions -> dynamically sampled during training
```

The training graph contains only train positive edges. Validation/test positive
edges do not enter message passing.

## Preprocess

```bash
python scripts/preprocess_graph_data.py
```

Outputs:

```text
data/processed/graph/random/
  preprocess_summary.json
  human/
    nodes.csv
    positive_edges.csv
    train_pos_edges.csv
    val_pos_edges.csv
    test_pos_edges.csv
    graph_inputs.npz
    graph_inputs.pt
    metadata.json
  cow/
  mouse/
  worm/
```

## Node Input

Each node starts from:

```text
numeric sequence features
+ trainable node ID embedding
+ node type embedding
+ species embedding
```

`nodes.csv` keeps biological metadata, while `graph_inputs.pt/npz` stores the
numeric tensors needed by the model.

`seq_length` is transformed to `seq_log_length = log1p(seq_length)`, then all
numeric node features are z-score standardized with statistics fitted only on
nodes that appear in train positive edges.

## Train And Plot

Smoke test:

```bash
conda run -n mti39 python scripts/train_gnn_transfer.py \
  --species worm cow \
  --encoders graphsage \
  --settings strict_zero_shot calibrated_zero_shot finetune \
  --epochs 1 \
  --finetune-epochs 1 \
  --neg-ratio 0.2 \
  --eval-neg-ratio 0.2 \
  --run-root runs/gnn_transfer_smoke
```

Full default experiment:

```bash
conda run -n mti39 python scripts/train_gnn_transfer.py
```

The default training mode uses dynamic training negatives and fixed validation /
test negatives. Fixed negatives are cached per target species under
`data/processed/graph/random/<species>/fixed_negatives/`, so all source models,
encoders, and transfer settings evaluate against the same target negative set.

Useful switches:

```text
--neg-strategy random|degree_aware|sequence_aware
--eval-neg-strategy same|random|degree_aware|sequence_aware
--fixed-eval-negatives / --no-fixed-eval-negatives
--refresh-fixed-negatives
--finetune-strategy full|last_layer|decoder
--residual --layer-norm --decoder-layer-norm
```

`strict_zero_shot` uses the fixed threshold, `calibrated_zero_shot` selects the
threshold on the target validation split without updating model parameters, and
`finetune` updates target parameters before selecting the threshold.

The default `--edge-attr-mode pair` computes the same lightweight pair features
for positive and sampled negative edges before the decoder.

Outputs:

```text
runs/gnn_transfer/<timestamp>/
  config.json
  transfer_metrics.csv
  transfer_metrics.json
  heatmaps/
    graphsage_zero_shot_auc.png
    graphsage_zero_shot_aupr.png
    ...
    gatv2_finetune_all_metrics.png
```

## Model Fairness

GraphSAGE and GATv2 share:

- the same node input encoder
- the same training graph
- the same dynamic negative sampler
- the same link decoder

Only the GNN encoder is changed.

## Transfer Heatmaps

For the later transfer-learning figures, keep the baseline-style source-target
matrix over `human/cow/mouse/worm`, then repeat it for:

- encoder: `GraphSAGE` / `GATv2`
- transfer setting: `strict zero-shot` / `calibrated zero-shot` / `target fine-tune`
- metric: `AUC`, `AUPR`, `ACC`, `F1`, `MCC`

For strict zero-shot cross-species transfer, local node IDs are disabled because
they are not biologically aligned across species. During cross-species fine-tune,
ID/species embeddings are not loaded from the source state even if their tensor
shapes happen to match.

## Edge Attributes

The preprocessing step still exports original positive-only edge attributes for
inspection, but the clean GNN default does not use them directly. Training uses
label-safe pair features computed from endpoint sequences for both positive and
dynamic negative edges:

```text
log lengths, length ratio/difference
GC values, GC difference/product
seed reverse-complement exact hits
seed hit counts normalized by mRNA length
seed GC values
```

This avoids the leakage caused by giving positives real RNAduplex features while
filling negative edge attributes with zeros.
