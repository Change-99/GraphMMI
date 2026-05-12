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

from mti_graphsage import GraphSAGENodePairPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train GraphSAGE on positive miRNA-mRNA edges with dynamic non-edge "
            "negative sampling. Precomputed neg.csv-derived negatives are not used."
        )
    )
    parser.add_argument("--species", required=True, choices=["human", "mouse", "worm", "cow"])
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed/graphsage_mrna/random")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/graphsage_dynamic_neg")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--decoder-hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument(
        "--negative-strategy",
        choices=["uniform", "endpoint_corrupt"],
        default="uniform",
        help="How to sample synthetic non-edge negatives for each positive edge.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--directed-message-passing",
        action="store_true",
        help="Use directed train positive edges instead of the default undirected train positive graph.",
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


def load_positive_graph(path: Path, device: torch.device, directed: bool) -> dict[str, Tensor | np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing GraphSAGE input file: {path}")

    data = np.load(path, allow_pickle=True)
    edge_index_key = "edge_index" if directed else "edge_index_undirected"
    graph: dict[str, Tensor | np.ndarray] = {
        "x": torch.from_numpy(data["x"]).float().to(device),
        "node_type": data["node_type"].astype(np.int64),
        "edge_index": torch.from_numpy(data[edge_index_key]).long().to(device),
        "node_ids": data["node_ids"],
    }
    positive_pairs: set[tuple[int, int]] = set()
    for split in ["train", "val", "test"]:
        label = data[f"{split}_edge_label"].astype(np.int64)
        edge_label_index = data[f"{split}_edge_label_index"].astype(np.int64)
        pos_index = edge_label_index[:, label == 1]
        graph[f"{split}_pos_edge_index_np"] = pos_index
        graph[f"{split}_pos_edge_index"] = torch.from_numpy(pos_index).long().to(device)
        for src, dst in pos_index.T:
            positive_pairs.add((int(src), int(dst)))
    graph["positive_pairs"] = positive_pairs
    return graph


def sample_negative_edges(
    pos_edge_index: np.ndarray,
    mirna_nodes: np.ndarray,
    mrna_nodes: np.ndarray,
    positive_pairs: set[tuple[int, int]],
    neg_ratio: float,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n_neg = int(round(pos_edge_index.shape[1] * neg_ratio))
    if n_neg <= 0:
        return np.zeros((2, 0), dtype=np.int64)
    if len(mirna_nodes) == 0 or len(mrna_nodes) == 0:
        raise ValueError("Both miRNA and mRNA node pools are required for negative sampling.")

    sampled: list[tuple[int, int]] = []
    local_seen: set[tuple[int, int]] = set()
    max_attempts = max(10000, n_neg * 200)
    attempts = 0
    while len(sampled) < n_neg and attempts < max_attempts:
        attempts += 1
        if strategy == "endpoint_corrupt":
            src, dst = pos_edge_index[:, int(rng.integers(0, pos_edge_index.shape[1]))]
            if bool(rng.integers(0, 2)):
                pair = (int(src), int(mrna_nodes[int(rng.integers(0, len(mrna_nodes)))]))
            else:
                pair = (int(mirna_nodes[int(rng.integers(0, len(mirna_nodes)))]), int(dst))
        else:
            pair = (
                int(mirna_nodes[int(rng.integers(0, len(mirna_nodes)))]),
                int(mrna_nodes[int(rng.integers(0, len(mrna_nodes)))]),
            )
        if pair in positive_pairs or pair in local_seen:
            continue
        sampled.append(pair)
        local_seen.add(pair)

    if len(sampled) != n_neg:
        raise RuntimeError(f"Failed to sample {n_neg} non-edge negatives; sampled {len(sampled)}")
    return np.asarray(sampled, dtype=np.int64).T


def make_labeled_edges(
    pos_edge_index: np.ndarray,
    neg_edge_index: np.ndarray,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    edge_index = np.concatenate([pos_edge_index, neg_edge_index], axis=1)
    labels = np.concatenate(
        [
            np.ones(pos_edge_index.shape[1], dtype=np.float32),
            np.zeros(neg_edge_index.shape[1], dtype=np.float32),
        ]
    )
    order = np.random.permutation(labels.shape[0])
    return (
        torch.from_numpy(edge_index[:, order]).long().to(device),
        torch.from_numpy(labels[order]).float().to(device),
    )


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
    model: GraphSAGENodePairPredictor,
    graph: dict[str, Tensor | np.ndarray],
    edge_label_index: Tensor,
    labels: Tensor,
    loss_fn: nn.Module,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    z = model.encode(graph["x"], graph["edge_index"])  # type: ignore[arg-type]
    logits = model.decode(z, edge_label_index)
    loss = loss_fn(logits, labels).item()
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    metrics = {
        "loss": float(loss),
        "auc": binary_auc(labels_np, probs),
        "ap": average_precision(labels_np, probs),
    }
    metrics.update(classification_metrics(labels_np, probs, threshold))
    return metrics


def train_one_epoch(
    model: GraphSAGENodePairPredictor,
    graph: dict[str, Tensor | np.ndarray],
    edge_label_index: Tensor,
    labels: Tensor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(
        graph["x"],  # type: ignore[arg-type]
        graph["edge_index"],  # type: ignore[arg-type]
        edge_label_index,
    )
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def save_predictions(
    model: GraphSAGENodePairPredictor,
    graph: dict[str, Tensor | np.ndarray],
    edge_label_index: Tensor,
    labels: Tensor,
    path: Path,
) -> None:
    model.eval()
    z = model.encode(graph["x"], graph["edge_index"])  # type: ignore[arg-type]
    logits = model.decode(z, edge_label_index)
    probs = torch.sigmoid(logits).cpu().numpy()
    labels_np = labels.cpu().numpy().astype(int)
    edges_np = edge_label_index.cpu().numpy()

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["src_idx", "dst_idx", "label", "probability", "logit"])
        for src, dst, label, prob, logit in zip(
            edges_np[0], edges_np[1], labels_np, probs, logits.cpu().numpy()
        ):
            writer.writerow([int(src), int(dst), int(label), float(prob), float(logit)])


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
    rng = np.random.default_rng(args.seed)

    data_path = args.data_root / args.species / "graphsage_inputs.npz"
    graph = load_positive_graph(data_path, device=device, directed=args.directed_message_passing)
    node_type = graph["node_type"]  # type: ignore[assignment]
    mirna_nodes = np.flatnonzero(node_type == 0)
    mrna_nodes = np.flatnonzero(node_type == 1)
    positive_pairs = graph["positive_pairs"]  # type: ignore[assignment]

    fixed_eval_edges: dict[str, tuple[Tensor, Tensor]] = {}
    for split in ["val", "test"]:
        split_rng = np.random.default_rng(args.seed + (17 if split == "val" else 29))
        pos_edge_index = graph[f"{split}_pos_edge_index_np"]  # type: ignore[assignment]
        neg_edge_index = sample_negative_edges(
            pos_edge_index=pos_edge_index,
            mirna_nodes=mirna_nodes,
            mrna_nodes=mrna_nodes,
            positive_pairs=positive_pairs,
            neg_ratio=args.neg_ratio,
            strategy=args.negative_strategy,
            rng=split_rng,
        )
        fixed_eval_edges[split] = make_labeled_edges(pos_edge_index, neg_edge_index, device)

    node_feature_dim = int(graph["x"].size(1))  # type: ignore[union-attr]
    model = GraphSAGENodePairPredictor(
        node_feature_dim=node_feature_dim,
        hidden_channels=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        decoder_hidden_channels=args.decoder_hidden_dim,
    ).to(device)
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
            "num_mirna_nodes": int(len(mirna_nodes)),
            "num_mrna_nodes": int(len(mrna_nodes)),
            "known_positive_pairs_excluded_from_negatives": int(len(positive_pairs)),
            "message_passing_graph": "directed_train_pos"
            if args.directed_message_passing
            else "undirected_train_pos",
            "model": "GraphSAGE positive graph + dynamic non-edge negatives",
        }
    )
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val_auc = -1.0
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    print(
        f"Training {args.species}: x={tuple(graph['x'].shape)} "
        f"edge_index={tuple(graph['edge_index'].shape)} "
        f"train_pos={graph['train_pos_edge_index_np'].shape[1]} "
        f"neg_ratio={args.neg_ratio} strategy={args.negative_strategy} device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        train_pos = graph["train_pos_edge_index_np"]  # type: ignore[assignment]
        train_neg = sample_negative_edges(
            pos_edge_index=train_pos,
            mirna_nodes=mirna_nodes,
            mrna_nodes=mrna_nodes,
            positive_pairs=positive_pairs,
            neg_ratio=args.neg_ratio,
            strategy=args.negative_strategy,
            rng=rng,
        )
        train_edge_label_index, train_labels = make_labeled_edges(train_pos, train_neg, device)
        train_loss = train_one_epoch(
            model, graph, train_edge_label_index, train_labels, optimizer, loss_fn
        )
        train_metrics = evaluate(
            model, graph, train_edge_label_index, train_labels, loss_fn, args.threshold
        )
        val_edge_label_index, val_labels = fixed_eval_edges["val"]
        val_metrics = evaluate(model, graph, val_edge_label_index, val_labels, loss_fn, args.threshold)

        row = {
            "epoch": epoch,
            "train_loss_backward": train_loss,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(row)

        val_auc = val_metrics["auc"]
        if val_auc > best_val_auc:
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

        if epoch == 1 or epoch % 10 == 0 or bad_epochs == 0:
            print(
                f"epoch={epoch:03d} train_loss={train_loss:.4f} "
                f"train_auc={train_metrics['auc']:.4f} val_auc={val_auc:.4f} "
                f"val_ap={val_metrics['ap']:.4f} best_epoch={best_epoch}"
            )

        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch={epoch}; best_epoch={best_epoch}, best_val_auc={best_val_auc:.4f}")
            break

    checkpoint = load_checkpoint(best_path, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_edge_label_index, val_labels = fixed_eval_edges["val"]
    test_edge_label_index, test_labels = fixed_eval_edges["test"]
    final_metrics = {
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "val": evaluate(model, graph, val_edge_label_index, val_labels, loss_fn, args.threshold),
        "test": evaluate(model, graph, test_edge_label_index, test_labels, loss_fn, args.threshold),
        "interpretation": (
            "Positive edges build the graph and provide positive supervision; "
            "negative labels are sampled dynamically from known non-edges."
        ),
    }

    (run_dir / "history.json").write_text(json.dumps(json_safe(history), indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(json_safe(final_metrics), indent=2), encoding="utf-8")
    save_predictions(model, graph, test_edge_label_index, test_labels, run_dir / "test_predictions.csv")

    print("Final metrics:")
    print(json.dumps(json_safe(final_metrics), indent=2))
    print(f"Saved run: {run_dir}")


if __name__ == "__main__":
    main()
