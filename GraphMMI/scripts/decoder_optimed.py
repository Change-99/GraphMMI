#!/usr/bin/env python3
"""Decoder architecture comparison for link prediction.

Tests multiple decoder designs on top of the same GraphSAGE encoder.
Uses the existing training pipeline with swapped decoder architectures.

Decoder variants:
  baseline  — concat [z_src, z_dst, z_src*z_dst, |z_src-z_dst|] + pair → MLP
  bilinear  — baseline + bilinear term: z_src^T W z_dst
  residual  — baseline MLP with residual skip connections
  separated — embedding features and pair features processed separately, then fused
  gated     — learnable scalar gate per feature group

Usage:
  python scripts/decoder_optimed.py --species human --decoders baseline bilinear \\
      --epochs 40 --patience 8 --num-layers 4 \\
      --processed-dir data/processed/graph/final_v2 \\
      --mirna-sim-edges --mrna-sim-edges \\
      --skip-preprocess --refresh-fixed-negatives \\
      --run-root runs/decoder_ablation --no-heatmaps
"""

from __future__ import annotations

import argparse, csv, gc, json, math, random, sys, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphmmi import load_graph_bundle, pair_feature_dim_v3, pair_feature_matrix, sample_negative_edges
from graphmmi.data import GraphBundle, positive_pair_set
from graphmmi.models import NodeInputEncoder, GraphSAGEEncoder

SPECIES_ORDER = ["human", "cow", "mouse", "worm"]
METRICS = ["auc", "aupr", "acc", "f1", "mcc"]


# ======================================================================
# decoder variants
# ======================================================================

class BaselineDecoder(nn.Module):
    """Original decoder: concat embedding features + pair features → MLP."""

    def __init__(self, hidden_dim: int = 128, edge_attr_dim: int = 0,
                 dropout: float = 0.3, layer_norm: bool = False):
        super().__init__()
        self.edge_attr_dim = edge_attr_dim
        input_dim = hidden_dim * 4 + edge_attr_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, z: Tensor, edge_label_index: Tensor,
                edge_attr: Tensor | None = None) -> Tensor:
        src, dst = edge_label_index
        parts = [z[src], z[dst], z[src] * z[dst], torch.abs(z[src] - z[dst])]
        if self.edge_attr_dim:
            parts.append(edge_attr)
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)


class BilinearDecoder(nn.Module):
    """Baseline + bilinear term: z_src^T W z_dst, captures pairwise interaction better."""

    def __init__(self, hidden_dim: int = 128, edge_attr_dim: int = 0,
                 dropout: float = 0.3, layer_norm: bool = False):
        super().__init__()
        self.edge_attr_dim = edge_attr_dim
        self.bilinear = nn.Bilinear(hidden_dim, hidden_dim, 32)
        input_dim = hidden_dim * 4 + 32 + edge_attr_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, z: Tensor, edge_label_index: Tensor,
                edge_attr: Tensor | None = None) -> Tensor:
        src, dst = edge_label_index
        zs, zd = z[src], z[dst]
        bilinear = self.bilinear(zs, zd)
        parts = [zs, zd, zs * zd, torch.abs(zs - zd), bilinear]
        if self.edge_attr_dim:
            parts.append(edge_attr)
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)


class ResidualDecoder(nn.Module):
    """MLP with residual connections between blocks."""

    def __init__(self, hidden_dim: int = 128, edge_attr_dim: int = 0,
                 dropout: float = 0.3, layer_norm: bool = False):
        super().__init__()
        self.edge_attr_dim = edge_attr_dim
        input_dim = hidden_dim * 4 + edge_attr_dim
        self.block1 = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(dropout))
        self.block2 = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout))
        self.block3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout))
        self.down1 = nn.Linear(input_dim, 256)  # for residual
        self.down2 = nn.Linear(256, 128)  # for residual
        self.head = nn.Linear(128, 1)

    def forward(self, z: Tensor, edge_label_index: Tensor,
                edge_attr: Tensor | None = None) -> Tensor:
        src, dst = edge_label_index
        parts = [z[src], z[dst], z[src] * z[dst], torch.abs(z[src] - z[dst])]
        if self.edge_attr_dim:
            parts.append(edge_attr)
        x = torch.cat(parts, dim=-1)
        h = self.block1(x) + self.down1(x)  # residual 1
        h = self.block2(h) + h              # residual 2 (identity since dim=256)
        h = self.block3(h) + self.down2(h)  # residual 3 (down to 128)
        return self.head(h).squeeze(-1)


class SeparatedDecoder(nn.Module):
    """Embedding features and pair features processed in separate streams, then fused."""

    def __init__(self, hidden_dim: int = 128, edge_attr_dim: int = 0,
                 dropout: float = 0.3, layer_norm: bool = False):
        super().__init__()
        self.edge_attr_dim = edge_attr_dim
        # embedding stream: z_src, z_dst, z_src*z_dst, |z_src-z_dst|
        self.emb_net = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # pair feature stream
        if edge_attr_dim:
            self.pair_net = nn.Sequential(
                nn.Linear(edge_attr_dim, 64),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
            )
        # fusion
        fusion_in = 128 + (32 if edge_attr_dim else 0)
        self.head = nn.Sequential(
            nn.Linear(fusion_in, 128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, z: Tensor, edge_label_index: Tensor,
                edge_attr: Tensor | None = None) -> Tensor:
        src, dst = edge_label_index
        emb_feat = torch.cat([z[src], z[dst], z[src] * z[dst], torch.abs(z[src] - z[dst])], dim=-1)
        h_emb = self.emb_net(emb_feat)
        if self.edge_attr_dim and edge_attr is not None:
            h_pair = self.pair_net(edge_attr)
            h = torch.cat([h_emb, h_pair], dim=-1)
        else:
            h = h_emb
        return self.head(h).squeeze(-1)


class GatedDecoder(nn.Module):
    """Learnable scalar gates per feature group: alpha * emb_feat + beta * pair_feat."""

    def __init__(self, hidden_dim: int = 128, edge_attr_dim: int = 0,
                 dropout: float = 0.3, layer_norm: bool = False):
        super().__init__()
        self.edge_attr_dim = edge_attr_dim
        input_dim = hidden_dim * 4 + edge_attr_dim
        self.emb_gate = nn.Parameter(torch.tensor(1.0))
        self.pair_gate = nn.Parameter(torch.tensor(1.0)) if edge_attr_dim else None
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, z: Tensor, edge_label_index: Tensor,
                edge_attr: Tensor | None = None) -> Tensor:
        src, dst = edge_label_index
        parts = [z[src] * self.emb_gate, z[dst] * self.emb_gate,
                 (z[src] * z[dst]) * self.emb_gate,
                 torch.abs(z[src] - z[dst]) * self.emb_gate]
        if self.edge_attr_dim and edge_attr is not None:
            parts.append(edge_attr * self.pair_gate)
        return self.mlp(torch.cat(parts, dim=-1)).squeeze(-1)


DECODER_REGISTRY = {
    "baseline": BaselineDecoder,
    "bilinear": BilinearDecoder,
    "residual": ResidualDecoder,
    "separated": SeparatedDecoder,
    "gated": GatedDecoder,
}


# ======================================================================
# model wrapper (same encoder, swappable decoder)
# ======================================================================

class DecoderTestPredictor(nn.Module):
    """GraphMMILinkPredictor with swappable decoder."""

    def __init__(self, encoder_name: str, num_numeric_features: int, num_nodes: int,
                 hidden_dim: int = 128, num_layers: int = 4,
                 edge_attr_dim: int = 0, dropout: float = 0.3,
                 decoder_type: str = "baseline",
                 id_embedding_dim: int = 0, type_embedding_dim: int = 8,
                 species_embedding_dim: int = 0,
                 residual: bool = False, layer_norm: bool = False,
                 use_edge_weight: bool = False):
        super().__init__()
        self.input_encoder = NodeInputEncoder(
            num_numeric_features=num_numeric_features, num_nodes=num_nodes,
            id_embedding_dim=id_embedding_dim, type_embedding_dim=type_embedding_dim,
            species_embedding_dim=species_embedding_dim, out_dim=hidden_dim,
            dropout=dropout)
        if encoder_name.lower() == "graphsage":
            self.gnn = GraphSAGEEncoder(hidden_dim=hidden_dim, num_layers=num_layers,
                                        dropout=dropout, residual=residual,
                                        layer_norm=layer_norm)
        else:
            raise ValueError(f"Only graphsage supported in decoder test: {encoder_name}")
        decoder_cls = DECODER_REGISTRY[decoder_type]
        self.decoder = decoder_cls(hidden_dim=hidden_dim, edge_attr_dim=edge_attr_dim,
                                   dropout=dropout)

    def encode(self, x, node_type, species_id, edge_index, edge_weight=None):
        h = self.input_encoder(x, node_type, species_id)
        return self.gnn(h, edge_index, edge_weight=edge_weight)

    def decode(self, z, edge_label_index, edge_attr=None):
        return self.decoder(z, edge_label_index, edge_attr)

    def forward(self, x, node_type, species_id, edge_index, edge_label_index,
                edge_attr=None, edge_weight=None):
        z = self.encode(x, node_type, species_id, edge_index, edge_weight=edge_weight)
        return self.decode(z, edge_label_index, edge_attr)


# ======================================================================
# training (same logic as train_gnn_transfer.py, simplified)
# ======================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Decoder architecture ablation")
    p.add_argument("--processed-dir", type=Path,
                   default=ROOT / "data/processed/graph/final_v2")
    p.add_argument("--run-root", type=Path, default=ROOT / "runs/decoder_ablation")
    p.add_argument("--species", nargs="+", default=["human"])
    p.add_argument("--decoders", nargs="+", default=["baseline", "bilinear", "separated"],
                   choices=list(DECODER_REGISTRY.keys()))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--neg-ratio", type=float, default=1.0)
    p.add_argument("--eval-neg-ratio", type=float, default=1.0)
    p.add_argument("--neg-strategy", default="endpoint_corrupt")
    p.add_argument("--mirna-sim-edges", action="store_true")
    p.add_argument("--mrna-sim-edges", action="store_true")
    p.add_argument("--no-heatmaps", action="store_true")
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def stable_seed(b, *p):
    v = b
    for part in p:
        for i, ch in enumerate(part): v += (i + 1) * ord(ch)
    return int(v % (2 ** 31 - 1))


def clone_sd(m):
    return {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}


# metrics
def _avg_ranks(v):
    o = np.argsort(v, kind="mergesort"); r = np.empty(len(v), dtype=np.float64)
    sv = v[o]; start = 0
    while start < len(v):
        end = start + 1
        while end < len(v) and sv[end] == sv[start]: end += 1
        r[o[start:end]] = (start + 1 + end) / 2.0; start = end
    return r

def roc_auc_np(l, s):
    l = l.astype(np.int64); n_pos = int(l.sum()); n_neg = len(l) - n_pos
    if n_pos == 0 or n_neg == 0: return float("nan")
    ranks = _avg_ranks(s)
    return (float(ranks[l == 1].sum()) - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)

def avg_prec_np(l, s):
    l = l.astype(np.int64); n_pos = int(l.sum())
    if n_pos == 0: return float("nan")
    o = np.argsort(-s, kind="mergesort"); sl = l[o]; tp = np.cumsum(sl)
    return float((tp / (np.arange(len(sl)) + 1) * sl).sum() / n_pos)


def train_one_decoder(decoder_type, graph, args, device, mp_edge_index, mp_edge_weight):
    """Train a single decoder variant, return results."""
    set_seed(stable_seed(args.seed, decoder_type, "human"))

    edge_attr_dim = pair_feature_dim_v3()
    model = DecoderTestPredictor(
        encoder_name="graphsage",
        num_numeric_features=int(graph.x.size(1)),
        num_nodes=int(graph.x.size(0)),
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        edge_attr_dim=edge_attr_dim, dropout=args.dropout,
        decoder_type=decoder_type, id_embedding_dim=0,
        type_embedding_dim=8, species_embedding_dim=0,
        use_edge_weight=(args.mirna_sim_edges or args.mrna_sim_edges),
    ).to(device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    best_sd = clone_sd(model); best_aupr = -float("inf"); stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_seed = stable_seed(args.seed, decoder_type, "train", str(epoch))
        gen = torch.Generator(device=device).manual_seed(train_seed)
        pos = graph.split_pos_edge_index["train"]
        neg = sample_negative_edges(pos, graph.node_type, graph.all_positive_edge_index,
                                     neg_ratio=args.neg_ratio, strategy=args.neg_strategy,
                                     generator=gen, node_sequences=graph.node_sequences,
                                     blocked_pairs=graph.positive_pair_cache)
        lei = torch.cat([pos, neg], dim=1)
        lab = torch.cat([torch.ones(pos.size(1), device=device),
                         torch.zeros(neg.size(1), device=device)])
        ea = pair_feature_matrix(lei, graph, version="v3")
        opt.zero_grad(set_to_none=True)
        logits = model(graph.x, graph.node_type, graph.species_id, mp_edge_index,
                       lei, ea, edge_weight=mp_edge_weight)
        loss = F.binary_cross_entropy_with_logits(logits, lab)
        loss.backward(); opt.step()

        # val
        model.eval()
        with torch.no_grad():
            val_seed = stable_seed(args.seed, decoder_type, "val", str(epoch))
            vgen = torch.Generator(device=device).manual_seed(val_seed)
            vpos = graph.split_pos_edge_index["val"]
            vneg = sample_negative_edges(vpos, graph.node_type, graph.all_positive_edge_index,
                                          neg_ratio=args.eval_neg_ratio, strategy=args.neg_strategy,
                                          generator=vgen, node_sequences=graph.node_sequences,
                                          blocked_pairs=graph.positive_pair_cache)
            vlei = torch.cat([vpos, vneg], dim=1)
            vlab = torch.cat([torch.ones(vpos.size(1), device=device),
                              torch.zeros(vneg.size(1), device=device)])
            vea = pair_feature_matrix(vlei, graph, version="v3")
            vlogits = model(graph.x, graph.node_type, graph.species_id, mp_edge_index,
                            vlei, vea, edge_weight=mp_edge_weight)
            vprobs = torch.sigmoid(vlogits).cpu().numpy().astype(np.float64)
            vlab_np = vlab.cpu().numpy().astype(np.int64)
            val_aupr = float(avg_prec_np(vlab_np, vprobs))
            val_auc = float(roc_auc_np(vlab_np, vprobs))

        improved = val_aupr > best_aupr
        if improved:
            best_aupr, best_sd = val_aupr, clone_sd(model); stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or improved:
            tag = " *" if improved else ""
            print(f"  [{decoder_type}] e={epoch:03d} loss={loss.item():.4f} "
                  f"val_aupr={val_aupr:.4f} val_auc={val_auc:.4f}{tag}")
        if stale >= args.patience:
            break

    model.load_state_dict(best_sd)

    # test
    model.eval()
    with torch.no_grad():
        test_seed = stable_seed(args.seed, decoder_type, "test_fixed")
        tgen = torch.Generator(device=device).manual_seed(test_seed)
        tpos = graph.split_pos_edge_index["test"]
        tneg = sample_negative_edges(tpos, graph.node_type, graph.all_positive_edge_index,
                                      neg_ratio=args.eval_neg_ratio, strategy=args.neg_strategy,
                                      generator=tgen, node_sequences=graph.node_sequences,
                                      blocked_pairs=graph.positive_pair_cache)
        tlei = torch.cat([tpos, tneg], dim=1)
        tlab = torch.cat([torch.ones(tpos.size(1), device=device),
                          torch.zeros(tneg.size(1), device=device)])
        tea = pair_feature_matrix(tlei, graph, version="v3")
        tlogits = model(graph.x, graph.node_type, graph.species_id, mp_edge_index,
                        tlei, tea, edge_weight=mp_edge_weight)
        tprobs = torch.sigmoid(tlogits).cpu().numpy().astype(np.float64)
        tlab_np = tlab.cpu().numpy().astype(np.int64)
        test_auc = float(roc_auc_np(tlab_np, tprobs))
        test_aupr = float(avg_prec_np(tlab_np, tprobs))

    return {"decoder": decoder_type, "best_epoch": epoch - stale,
            "best_val_aupr": best_aupr, "test_auc": test_auc, "test_aupr": test_aupr}


def main():
    args = parse_args()
    device = torch.device(args.device)
    set_seed(args.seed)

    # load graph
    graph = load_graph_bundle(
        args.processed_dir / args.species[0] / "graph_inputs.pt",
        device=device, load_edge_attr=False)
    graph.positive_pair_cache = positive_pair_set(graph.all_positive_edge_index)

    # build MP edges
    parts = [graph.edge_index]
    ew_parts = [torch.ones(graph.edge_index.size(1), dtype=torch.float32, device=device)]
    if args.mirna_sim_edges and graph.similarity_edge_index_mirna is not None:
        parts.append(graph.similarity_edge_index_mirna)
        w = graph.similarity_edge_weight_mirna
        ew_parts.append(w.to(dtype=torch.float32, device=device) if w is not None
                        else torch.ones(graph.similarity_edge_index_mirna.size(1), device=device))
    if args.mrna_sim_edges and graph.similarity_edge_index_mrna is not None:
        parts.append(graph.similarity_edge_index_mrna)
        w = graph.similarity_edge_weight_mrna
        ew_parts.append(w.to(dtype=torch.float32, device=device) if w is not None
                        else torch.ones(graph.similarity_edge_index_mrna.size(1), device=device))
    mp_ei = torch.cat(parts, dim=1)
    mp_ew = torch.cat(ew_parts, dim=0)

    print(f"[LOAD] nodes={graph.x.size(0)} feats={graph.x.size(1)} "
          f"mp_edges={mp_ei.size(1)} edge_attr_dim={pair_feature_dim_v3()}")

    rows = []
    for dec in args.decoders:
        print(f"\n=== decoder: {dec} ===")
        result = train_one_decoder(dec, graph, args, device, mp_ei, mp_ew)
        rows.append(result)
        print(f"[{dec}] test AUC={result['test_auc']:.4f} AUPR={result['test_aupr']:.4f}")

    # save
    args.run_root.mkdir(parents=True, exist_ok=True)
    out = args.run_root / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "decoder_ablation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["decoder", "best_epoch", "best_val_aupr", "test_auc", "test_aupr"])
        w.writeheader(); w.writerows(rows)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
