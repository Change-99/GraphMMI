# Exp3 Ablation Experiments

All scripts use the target-site graph:

```text
data/processed/graph/final_target_site
```

If the graph data is missing, each script builds it with:

```bash
python scripts/final_embedding.py --node-mode target_site --sim-mode topk --mirna-sim-topk 5 --mrna-sim-topk 5
```

## A_edges

Similarity-edge ablation with the best backbone, GraphSAGE L4.

Variants:

- `no-sim`
- `miRNA-only`
- `target-only`
- `both-sim`

Species: `human cow`.

Run:

```bash
bash final_exp/exp3/A_edges/run.sh
```

## B_encoders

Encoder/layer-depth ablation.

Models:

- GraphSAGE, layers 1..6
- GATv2, layers 1..6

Species: `human`.

Run:

```bash
bash final_exp/exp3/B_encoders/run.sh
```

## C_decoder

Decoder architecture ablation on GraphSAGE L4.

Decoders:

- `baseline`
- `residual`
- `gated`
- `bilinear`
- `separated`

Species: `human`.

Run:

```bash
bash final_exp/exp3/C_decoder/run.sh
```

## D_neg_sample

Negative-sampling ablation on GraphSAGE L4.

Strategies:

- `endpoint_corrupt`
- `degree_aware`
- `sequence_aware`

Species: `human`.

Run:

```bash
bash final_exp/exp3/D_neg_sample/run.sh
```

## Run Everything

```bash
bash final_exp/exp3/run_all.sh
```
