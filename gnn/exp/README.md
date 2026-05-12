# Experiments

This directory keeps small diagnostic experiments separate from the main
GraphSAGE pipeline.

## Experiment 1: Edge-MLP Baseline

Goal: test whether the 631-D edge features alone can solve the task.

Model:

```text
edge_attr [num_edges, 631]
  -> MLP
  -> logit
  -> BCEWithLogitsLoss(logit, label)
```

It does not use node features, GraphSAGE, or `edge_index`.

Interpretation:

- If Edge-MLP test AUC is greater than 0.99, the edge features or negative
  sampling strategy are likely strong enough to solve the label task without
  graph representation learning.
- If GraphSAGE improves clearly over Edge-MLP, the graph structure and node
  embeddings are contributing additional signal.

Example:

```bash
python exp/edge_mlp_baseline.py --species human --epochs 200 --hidden-dim 256 --num-layers 3 --dropout 0.3 --lr 1e-3 --weight-decay 1e-4 --patience 30 --device auto
```

## Experiment 2: GraphSAGE-Only Baseline

Goal: test how much signal comes from node k-mer features and the train-positive
graph structure after removing all 631-D edge features.

Model:

```text
x [num_nodes, 22] + edge_index_train_pos
  -> GraphSAGE
  -> node embeddings z
  -> [z_src, z_dst, z_src * z_dst, |z_src - z_dst|]
  -> MLP
  -> logit
  -> BCEWithLogitsLoss(logit, label)
```

It does not read or concatenate `edge_attr`.

Use the leakage-safe miRNA-mRNA graph outputs from
`data/processed/graphsage_mrna/random` or
`data/processed/graphsage_mrna/cold_mirna`, not the old target-site graph.

Interpretation:

- If GraphSAGE-only AUC drops sharply compared with Edge-MLP or the full model,
  the graph/node representation is not the main source of predictive power.
- If GraphSAGE-only remains strong, the train-positive graph and sequence-derived
  node features are carrying useful signal.

Example:

```bash
python exp/graphsage_only.py --species human --epochs 200 --hidden-dim 128 --num-layers 2 --dropout 0.3 --lr 1e-3 --weight-decay 1e-4 --patience 30 --device auto
```

Cold-miRNA version:

```bash
python exp/graphsage_only.py --species human --data-root data/processed/graphsage_mrna/cold_mirna --epochs 200 --hidden-dim 128 --num-layers 2 --dropout 0.3 --lr 1e-3 --weight-decay 1e-4 --patience 30 --device auto
```

## Experiment 3: GraphSAGE + Edge Attributes

Goal: combine node/graph structural information with pairwise biological
features.

Model:

```text
x + edge_index_train_pos
  -> GraphSAGE
  -> node embeddings z
  -> [z_src, z_dst, edge_attr]
  -> MLP
  -> logit
  -> BCEWithLogitsLoss(logit, label)
```

This experiment requires non-empty precomputed `edge_attr`, so it currently uses
the older `data/processed/graphsage` target-site feature dataset. The corrected
`graphsage_mrna` graph has `edge_attr_dim = 0` because its negative edges are
freshly sampled miRNA-mRNA pairs without target-site-level features.

Example:

```bash
python exp/graphsage_edgeattr.py --species human --epochs 200 --hidden-dim 128 --num-layers 2 --dropout 0.3 --lr 1e-3 --weight-decay 1e-4 --patience 30 --device auto
```
