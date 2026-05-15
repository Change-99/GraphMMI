#!/usr/bin/env python3
"""GraphMAE pretraining: mask node features, encode, reconstruct.

Trains a GraphSAGE encoder to reconstruct masked node features from
neighbourhood context.  The pretrained encoder weights can then be
loaded into train_gnn_transfer.py for link-prediction finetuning.

Usage:
  # single-species pretraining
  python scripts/pretrain_graphmae.py --species human

  # multi-species (separate pretraining per species)
  python scripts/pretrain_graphmae.py --species human cow mouse worm
"""

from __future__ import annotations

import argparse, sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphmmi import GraphMMILinkPredictor, load_graph_bundle

SPECIES_ORDER = ["human", "cow", "mouse", "worm"]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GraphMAE pretraining")
    p.add_argument("--processed-dir", type=Path,
                   default=ROOT / "data/processed/graph/embedding_optimed_v2_topk")
    p.add_argument("--species", nargs="+", default=["human"])
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "runs/pretrain_graphmae")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    # architecture
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.3)
    # masking
    p.add_argument("--mask-rate", type=float, default=0.30,
                   help="Fraction of nodes whose features are masked.")
    # training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    # edges
    p.add_argument("--mirna-sim-edges", action="store_true", default=True)
    p.add_argument("--mrna-sim-edges", action="store_true", default=True)
    return p.parse_args()


# ------------------------------------------------------------------
# model
# ------------------------------------------------------------------

class GraphMAEPretrainer(nn.Module):
    """Wraps a GNN encoder with an MLP decoder for feature reconstruction."""

    def __init__(self, encoder: GraphMMILinkPredictor,
                 hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, node_type: torch.Tensor,
                species_id: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encoder.encode(x, node_type, species_id, edge_index,
                                edge_weight=edge_weight)
        return self.decoder(z)


# ------------------------------------------------------------------
# masking
# ------------------------------------------------------------------

def mask_node_features(x: torch.Tensor, mask_rate: float,
                       generator: torch.Generator | None = None,
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero out a random subset of node features.

    Returns (x_masked, mask_bool) where mask_bool is True for masked nodes.
    Guarantees at least one node is masked when mask_rate > 0.
    """
    num_nodes = x.size(0)
    mask_nodes = (torch.rand(num_nodes, generator=generator, device=x.device)
                  < mask_rate)
    if mask_rate > 0 and mask_nodes.sum() == 0:
        idx = torch.randint(0, num_nodes, (1,), generator=generator, device=x.device)
        mask_nodes[idx] = True
    x_masked = x.clone()
    x_masked[mask_nodes] = 0.0
    return x_masked, mask_nodes


# ------------------------------------------------------------------
# training
# ------------------------------------------------------------------

def build_message_passing_graph(graph, args) -> torch.Tensor:
    """Concatenate interaction + similarity edges for message passing."""
    parts = [graph.edge_index]
    if args.mirna_sim_edges and graph.similarity_edge_index_mirna is not None:
        parts.append(graph.similarity_edge_index_mirna)
    if args.mrna_sim_edges and graph.similarity_edge_index_mrna is not None:
        parts.append(graph.similarity_edge_index_mrna)
    return torch.cat(parts, dim=1)


def pretrain_one_species(species: str, args: argparse.Namespace,
                         device: torch.device) -> Path:
    path = args.processed_dir / species / "graph_inputs.pt"
    print(f"\n[GraphMAE] loading {path}")
    graph = load_graph_bundle(path, device=device, load_edge_attr=False)
    mp_edge_index = build_message_passing_graph(graph, args)
    n_int = graph.edge_index.size(1)
    n_mir = 0 if graph.similarity_edge_index_mirna is None else graph.similarity_edge_index_mirna.size(1)
    n_mrn = 0 if graph.similarity_edge_index_mrna is None else graph.similarity_edge_index_mrna.size(1)
    print(f"  nodes={graph.x.size(0)} feats={graph.x.size(1)} "
          f"mp_edges={mp_edge_index.size(1)}")
    print(f"  edges: int={n_int} mirna_sim={n_mir} mrna_sim={n_mrn}")
    print(f"  x mean={graph.x.mean().item():.4f} std={graph.x.std().item():.4f}")

    encoder = GraphMMILinkPredictor(
        encoder_name="graphsage",
        num_numeric_features=int(graph.x.size(1)),
        num_nodes=int(graph.x.size(0)),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        edge_attr_dim=0,
        dropout=args.dropout,
        id_embedding_dim=0,
        type_embedding_dim=8,
        species_embedding_dim=0,
    ).to(device)

    model = GraphMAEPretrainer(
        encoder=encoder,
        hidden_dim=args.hidden_dim,
        out_dim=int(graph.x.size(1)),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        x_masked, mask_nodes = mask_node_features(
            graph.x, args.mask_rate, generator=gen)

        x_recon = model(x_masked, graph.node_type, graph.species_id,
                        mp_edge_index)

        loss = F.mse_loss(x_recon[mask_nodes], graph.x[mask_nodes])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"  epoch={epoch:03d} loss={loss.item():.6f} "
                  f"masked={mask_nodes.sum().item()}")

    # Save only encoder-relevant weights (input_encoder + gnn).
    # The link-prediction decoder is random and should not be loaded.
    encoder_sd = {
        k: v.detach().cpu()
        for k, v in encoder.state_dict().items()
        if k.startswith("input_encoder.") or k.startswith("gnn.")
    }
    out_path = (args.output_dir
                / f"{species}_graphsage_l{args.num_layers}_mae.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "encoder_state_dict": encoder_sd,
        "species": species,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "mask_rate": args.mask_rate,
        "num_node_features": int(graph.x.size(1)),
    }, out_path)
    print(f"  saved -> {out_path}  (encoder keys: {len(encoder_sd)})")
    return out_path


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    for species in args.species:
        pretrain_one_species(species, args, device)

    print(f"\nDone. Models saved to {args.output_dir}")


if __name__ == "__main__":
    main()
