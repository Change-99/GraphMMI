#!/usr/bin/env python3
"""Minimal reproduction: load data, build model, run 1 epoch, print val_aupr."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from graphmmi import GraphMMILinkPredictor, load_graph_bundle, pair_feature_dim, pair_feature_matrix, sample_negative_edges
from graphmmi.data import GraphBundle, positive_pair_set

def set_seed(s):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def stable_seed(b, *p):
    v = b
    for part in p:
        for i, ch in enumerate(part): v += (i+1)*ord(ch)
    return int(v % (2**31-1))

set_seed(stable_seed(42, "graphsage", "zero_shot", "human", "source"))

device = torch.device("cpu")
graph = load_graph_bundle(ROOT / "data/processed/graph/embedding_optimed_v2_topk/human/graph_inputs.pt", device=device, load_edge_attr=False)
graph.positive_pair_cache = positive_pair_set(graph.all_positive_edge_index)

# Build model EXACTLY as in train_gnn_transfer.py
edge_attr_dim = pair_feature_dim()  # = 17
model = GraphMMILinkPredictor(
    encoder_name="graphsage",
    num_numeric_features=int(graph.x.size(1)),
    num_nodes=int(graph.x.size(0)),
    hidden_dim=128, num_layers=4, edge_attr_dim=edge_attr_dim,
    dropout=0.3, gat_heads=2, gat_concat=False,
    id_embedding_dim=0, type_embedding_dim=8, species_embedding_dim=0,
    residual=False, layer_norm=False, decoder_layer_norm=False,
    use_edge_weight=False,
).to(device)

print(f"edge_attr_dim={edge_attr_dim}")
print(f"model params: {sum(p.numel() for p in model.parameters())}")

# Build MP edges (same as resolve_edge_index_for_training with both sim on)
parts = [graph.edge_index]
if graph.similarity_edge_index_mirna is not None:
    parts.append(graph.similarity_edge_index_mirna)
if graph.similarity_edge_index_mrna is not None:
    parts.append(graph.similarity_edge_index_mrna)
mp_ei = torch.cat(parts, dim=1)
mp_ew = torch.ones(mp_ei.size(1), device=device)
print(f"mp edges: {mp_ei.size(1)} int={graph.edge_index.size(1)} mirna_sim={graph.similarity_edge_index_mirna.size(1)} mrna_sim={graph.similarity_edge_index_mrna.size(1)}")

# Train batch (epoch 1)
gen = torch.Generator(device=device).manual_seed(stable_seed(stable_seed(42, "graphsage", "zero_shot", "human"), "source", "human", "1"))
pos = graph.split_pos_edge_index["train"]
neg = sample_negative_edges(pos, graph.node_type, graph.all_positive_edge_index,
                             neg_ratio=1.0, strategy="endpoint_corrupt",
                             generator=gen, node_sequences=graph.node_sequences,
                             blocked_pairs=graph.positive_pair_cache)
lei = torch.cat([pos, neg], dim=1)
lab = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))])
ea = pair_feature_matrix(lei, graph)  # v1 default
print(f"train batch: pos={pos.size(1)} neg={neg.size(1)} edge_attr_dim={ea.size(1)}")

# Forward
model.train()
logits = model(graph.x, graph.node_type, graph.species_id, mp_ei, lei, ea, edge_weight=mp_ew)
loss = F.binary_cross_entropy_with_logits(logits, lab)
print(f"train loss: {loss.item():.4f}")

# Val batch (epoch 1, with fixed negatives cache cleared)
model.eval()
with torch.no_grad():
    val_seed = stable_seed(stable_seed(42, "graphsage", "zero_shot", "human"), "source", "human", "val", "1")
    vgen = torch.Generator(device=device).manual_seed(val_seed)
    vpos = graph.split_pos_edge_index["val"]
    vneg = sample_negative_edges(vpos, graph.node_type, graph.all_positive_edge_index,
                                  neg_ratio=1.0, strategy="endpoint_corrupt",
                                  generator=vgen, node_sequences=graph.node_sequences,
                                  blocked_pairs=graph.positive_pair_cache)
    vlei = torch.cat([vpos, vneg], dim=1)
    vlab = torch.cat([torch.ones(vpos.size(1)), torch.zeros(vneg.size(1))])
    vea = pair_feature_matrix(vlei, graph)
    vlogits = model(graph.x, graph.node_type, graph.species_id, mp_ei, vlei, vea, edge_weight=mp_ew)

    vprobs = torch.sigmoid(vlogits).cpu().numpy().astype(np.float64)
    vlab_np = vlab.cpu().numpy().astype(np.int64)

    # inline metrics (same as train_gnn_transfer.py)
    def _avg_ranks(v):
        o = np.argsort(v, kind="mergesort"); r = np.empty(len(v), dtype=np.float64)
        sv = v[o]; start = 0
        while start < len(v):
            end = start + 1
            while end < len(v) and sv[end] == sv[start]: end += 1
            r[o[start:end]] = (start + 1 + end) / 2.0; start = end
        return r
    def _auc(l, s):
        l = l.astype(np.int64); n_pos = int(l.sum()); n_neg = len(l) - n_pos
        if n_pos == 0 or n_neg == 0: return float("nan")
        ranks = _avg_ranks(s)
        return (float(ranks[l == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    def _aupr(l, s):
        l = l.astype(np.int64); n_pos = int(l.sum())
        if n_pos == 0: return float("nan")
        o = np.argsort(-s, kind="mergesort"); sl = l[o]; tp = np.cumsum(sl)
        return float((tp / (np.arange(len(sl)) + 1) * sl).sum() / n_pos)

    val_aupr = float(_aupr(vlab_np, vprobs))
    val_auc = float(_auc(vlab_np, vprobs))
    print(f"val_aupr={val_aupr:.4f} val_auc={val_auc:.4f}")
    print(f"pos={vpos.size(1)} neg={vneg.size(1)}")
    print(f"vneg samples: {vneg[:, :5].T.tolist()}")
