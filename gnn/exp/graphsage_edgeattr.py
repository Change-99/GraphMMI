#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import torch
    from torch import Tensor, nn
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is not installed. Install it first, for example: "
        "python -m pip install torch --index-url https://download.pytorch.org/whl/cpu"
    ) from exc

from mti_graphsage.model import GraphSAGEEncoder


class EdgeAttrDecoder(nn.Module):
    """Decoder over concat(z_src, z_dst, edge_attr)."""

    def __init__(
        self,
        hidden_dim: int = 128,
        edge_attr_dim: int = 631,
        mlp_hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        decoder_in = hidden_dim * 2 + edge_attr_dim
        self.mlp = nn.Sequential(
            nn.Linear(decoder_in, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim // 2, 1),
        )

    def forward(self, z: Tensor, edge_label_index: Tensor, edge_attr: Tensor) -> Tensor:
        src, dst = edge_label_index
        z_src = z[src]
        z_dst = z[dst]
        pair_features = torch.cat([z_src, z_dst, edge_attr], dim=-1)
        return self.mlp(pair_features).squeeze(-1)


class GraphSAGEEdgeAttrPredictor(nn.Module):
    """GraphSAGE encoder plus edge-attribute-aware decoder."""

    def __init__(
        self,
        node_feature_dim: int = 22,
        hidden_dim: int = 128,
        edge_attr_dim: int = 631,
        num_layers: int = 2,
        dropout: float = 0.3,
        decoder_hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.encoder = GraphSAGEEncoder(
            in_channels=node_feature_dim,
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = EdgeAttrDecoder(
            hidden_dim=hidden_dim,
            edge_attr_dim=edge_attr_dim,
            mlp_hidden_dim=decoder_hidden_dim,
            dropout=dropout,
        )

    def encode(self, x: Tensor, edge_index: Tensor) -> Tensor:
        return self.encoder(x, edge_index)

    def decode(self, z: Tensor, edge_label_index: Tensor, edge_attr: Tensor) -> Tensor:
        return self.decoder(z, edge_label_index, edge_attr)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_label_index: Tensor,
        edge_attr: Tensor,
    ) -> Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index, edge_attr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment 3: GraphSAGE + edge_attr using concat(z_src, z_dst, edge_attr)."
    )
    parser.add_argument("--species", required=True, choices=["human", "mouse", "worm", "cow"])
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed/graphsage")
    parser.add_argument("--run-root", type=Path, default=ROOT / "exp/runs/graphsage_edgeattr")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--directed-message-passing",
        action="store_true",
        help="Use directed train positive edges instead of the default undirected train positive graph.",
    )
    parser.add_argument(
        "--use-pos-weight",
        action="store_true",
        help="Use train negative/positive ratio as BCEWithLogitsLoss pos_weight.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")
    return torch.device(device_arg)


def load_graph_data(path: Path, device: torch.device, directed: bool) -> dict[str, Tensor | np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing GraphSAGE input file: {path}")

    data = np.load(path, allow_pickle=True)
    edge_index_key = "edge_index" if directed else "edge_index_undirected"
    graph: dict[str, Tensor | np.ndarray] = {
        "x": torch.from_numpy(data["x"]).float().to(device),
        "edge_index": torch.from_numpy(data[edge_index_key]).long().to(device),
        "node_ids": data["node_ids"],
        "edge_feature_names": data["edge_feature_names"],
    }
    for split in ["train", "val", "test"]:
        graph[f"{split}_edge_label_index"] = torch.from_numpy(
            data[f"{split}_edge_label_index"]
        ).long().to(device)
        graph[f"{split}_label"] = torch.from_numpy(data[f"{split}_edge_label"]).float().to(device)
        graph[f"{split}_edge_attr"] = torch.from_numpy(data[f"{split}_edge_attr"]).float().to(device)
        graph[f"{split}_sample_id"] = data[f"{split}_sample_id"]
    return graph


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    sum_pos_ranks = ranks[pos].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")

    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    ranks = np.arange(1, len(labels) + 1)
    precision = tp / ranks
    return float(precision[sorted_labels == 1].sum() / n_pos)


def classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    pred = probs >= threshold
    truth = labels.astype(bool)
    tp = int(np.logical_and(pred, truth).sum())
    tn = int(np.logical_and(~pred, ~truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


@torch.no_grad()
def evaluate(
    model: GraphSAGEEdgeAttrPredictor,
    graph: dict[str, Tensor | np.ndarray],
    split: str,
    loss_fn: nn.Module,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    z = model.encode(graph["x"], graph["edge_index"])  # type: ignore[arg-type]
    logits = model.decode(
        z,
        graph[f"{split}_edge_label_index"],  # type: ignore[arg-type]
        graph[f"{split}_edge_attr"],  # type: ignore[arg-type]
    )
    labels = graph[f"{split}_label"]  # type: ignore[assignment]
    loss = loss_fn(logits, labels).item()  # type: ignore[arg-type]
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()  # type: ignore[union-attr]

    metrics = {
        "loss": float(loss),
        "auc": binary_auc(labels_np, probs),
        "ap": average_precision(labels_np, probs),
    }
    metrics.update(classification_metrics(labels_np, probs, threshold))
    return metrics


def train_one_epoch(
    model: GraphSAGEEdgeAttrPredictor,
    graph: dict[str, Tensor | np.ndarray],
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(
        graph["x"],  # type: ignore[arg-type]
        graph["edge_index"],  # type: ignore[arg-type]
        graph["train_edge_label_index"],  # type: ignore[arg-type]
        graph["train_edge_attr"],  # type: ignore[arg-type]
    )
    loss = loss_fn(logits, graph["train_label"])  # type: ignore[arg-type]
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def save_predictions(
    model: GraphSAGEEdgeAttrPredictor,
    graph: dict[str, Tensor | np.ndarray],
    split: str,
    path: Path,
) -> None:
    model.eval()
    z = model.encode(graph["x"], graph["edge_index"])  # type: ignore[arg-type]
    logits = model.decode(
        z,
        graph[f"{split}_edge_label_index"],  # type: ignore[arg-type]
        graph[f"{split}_edge_attr"],  # type: ignore[arg-type]
    )
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels = graph[f"{split}_label"].detach().cpu().numpy().astype(int)  # type: ignore[union-attr]
    sample_ids = graph[f"{split}_sample_id"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "label", "probability", "logit"])
        for sample_id, label, prob, logit in zip(sample_ids, labels, probs, logits.cpu().numpy()):
            writer.writerow([sample_id, int(label), float(prob), float(logit)])


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data_path = args.data_root / args.species / "graphsage_inputs.npz"
    graph = load_graph_data(data_path, device=device, directed=args.directed_message_passing)
    node_feature_dim = int(graph["x"].size(1))  # type: ignore[union-attr]
    edge_attr_dim = int(graph["train_edge_attr"].size(1))  # type: ignore[union-attr]
    if edge_attr_dim <= 0:
        raise ValueError(
            f"{data_path} has edge_attr_dim={edge_attr_dim}. "
            "GraphSAGE+edge_attr requires precomputed pairwise edge features."
        )

    model = GraphSAGEEdgeAttrPredictor(
        node_feature_dim=node_feature_dim,
        hidden_dim=args.hidden_dim,
        edge_attr_dim=edge_attr_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        decoder_hidden_dim=args.decoder_hidden_dim,
    ).to(device)

    train_labels = graph["train_label"]  # type: ignore[assignment]
    if args.use_pos_weight:
        n_pos = train_labels.sum().clamp_min(1.0)  # type: ignore[union-attr]
        n_neg = (train_labels.numel() - train_labels.sum()).clamp_min(1.0)  # type: ignore[union-attr]
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=(n_neg / n_pos).to(device))
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_root / f"{args.species}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best_model.pt"

    config = {key: json_safe(value) for key, value in vars(args).items()}
    config.update(
        {
            "data_path": str(data_path),
            "device": str(device),
            "node_feature_dim": node_feature_dim,
            "edge_attr_dim": edge_attr_dim,
            "decoder_input_dim": args.hidden_dim * 2 + edge_attr_dim,
            "decoder_formula": "concat(z_src, z_dst, edge_attr)",
            "message_passing_graph": "directed_train_pos"
            if args.directed_message_passing
            else "undirected_train_pos",
            "model": "GraphSAGE+edge_attr",
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val_auc = -1.0
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []

    print(
        f"Experiment 3 GraphSAGE+edge_attr {args.species}: "
        f"x={tuple(graph['x'].shape)} "  # type: ignore[union-attr]
        f"edge_index={tuple(graph['edge_index'].shape)} "  # type: ignore[union-attr]
        f"train={graph['train_label'].numel()} "  # type: ignore[union-attr]
        f"val={graph['val_label'].numel()} "  # type: ignore[union-attr]
        f"test={graph['test_label'].numel()} "  # type: ignore[union-attr]
        f"edge_attr_dim={edge_attr_dim} "
        f"decoder_in={args.hidden_dim * 2 + edge_attr_dim} device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, graph, optimizer, loss_fn)
        train_metrics = evaluate(model, graph, "train", loss_fn, args.threshold)
        val_metrics = evaluate(model, graph, "val", loss_fn, args.threshold)

        row = {
            "epoch": epoch,
            "train_loss_backward": train_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        val_auc = val_metrics["auc"]
        improved = val_auc > best_val_auc
        if improved:
            best_val_auc = val_auc
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_val_auc": best_val_auc,
                },
                best_path,
            )
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or improved:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} train_auc={train_metrics['auc']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} val_auc={val_auc:.4f} "
                f"val_ap={val_metrics['ap']:.4f} best_epoch={best_epoch}"
            )

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}, best_val_auc={best_val_auc:.4f}")
            break

    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "train": evaluate(model, graph, "train", loss_fn, args.threshold),
        "val": evaluate(model, graph, "val", loss_fn, args.threshold),
        "test": evaluate(model, graph, "test", loss_fn, args.threshold),
        "interpretation": "Compares combined graph/node embeddings plus pairwise biological edge features.",
    }

    (run_dir / "history.json").write_text(json.dumps(json_safe(history), indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(json_safe(final_metrics), indent=2), encoding="utf-8")
    save_predictions(model, graph, "test", run_dir / "test_predictions.csv")

    print("Final metrics:")
    print(json.dumps(json_safe(final_metrics), indent=2))
    print(f"Saved run: {run_dir}")


if __name__ == "__main__":
    main()
