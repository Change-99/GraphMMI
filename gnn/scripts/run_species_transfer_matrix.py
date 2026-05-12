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


SPECIES = ["human", "cow", "mouse", "worm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one positive-graph GraphSAGE model per source species and evaluate "
            "all source-target combinations as a 4x4 transfer matrix."
        )
    )
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed/graphsage_mrna/random")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/species_transfer_matrix")
    parser.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
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
        default="endpoint_corrupt",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--metric", default="auc", choices=["auc", "ap", "accuracy", "f1"])
    parser.add_argument("--no-heatmap", action="store_true")
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
        for src, dst in pos_index.T:
            positive_pairs.add((int(src), int(dst)))
    graph["positive_pairs"] = positive_pairs
    return graph


def node_pools(graph: dict[str, Tensor | np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    node_type = graph["node_type"]  # type: ignore[assignment]
    return np.flatnonzero(node_type == 0), np.flatnonzero(node_type == 1)


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
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor]:
    edge_index = np.concatenate([pos_edge_index, neg_edge_index], axis=1)
    labels = np.concatenate(
        [
            np.ones(pos_edge_index.shape[1], dtype=np.float32),
            np.zeros(neg_edge_index.shape[1], dtype=np.float32),
        ]
    )
    order = rng.permutation(labels.shape[0])
    return (
        torch.from_numpy(edge_index[:, order]).long().to(device),
        torch.from_numpy(labels[order]).float().to(device),
    )


def sample_labeled_split(
    graph: dict[str, Tensor | np.ndarray],
    split: str,
    neg_ratio: float,
    strategy: str,
    device: torch.device,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor]:
    mirna_nodes, mrna_nodes = node_pools(graph)
    pos_edge_index = graph[f"{split}_pos_edge_index_np"]  # type: ignore[assignment]
    neg_edge_index = sample_negative_edges(
        pos_edge_index=pos_edge_index,
        mirna_nodes=mirna_nodes,
        mrna_nodes=mrna_nodes,
        positive_pairs=graph["positive_pairs"],  # type: ignore[arg-type]
        neg_ratio=neg_ratio,
        strategy=strategy,
        rng=rng,
    )
    return make_labeled_edges(pos_edge_index, neg_edge_index, device, rng)


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


def best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    if len(labels) == 0:
        return 0.5, {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    candidates = np.unique(probs.astype(np.float64))
    if candidates.size == 0:
        candidates = np.asarray([0.5], dtype=np.float64)
    candidates = np.concatenate(([0.0], candidates, [1.0]))

    best_threshold = 0.5
    best_metrics = classification_metrics(labels, probs, best_threshold)
    best_score = best_metrics["f1"]
    for threshold in candidates:
        metrics = classification_metrics(labels, probs, float(threshold))
        score = metrics["f1"]
        if score > best_score or (np.isclose(score, best_score) and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)):
            best_threshold = float(threshold)
            best_metrics = metrics
            best_score = score
    return best_threshold, best_metrics


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


def train_source_model(
    source: str,
    graph: dict[str, Tensor | np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
) -> tuple[GraphSAGENodePairPredictor, dict[str, Any]]:
    node_feature_dim = int(graph["x"].size(1))  # type: ignore[union-attr]
    model = GraphSAGENodePairPredictor(
        node_feature_dim=node_feature_dim,
        hidden_channels=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        decoder_hidden_channels=args.decoder_hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    train_rng = np.random.default_rng(args.seed + 1009 * (SPECIES.index(source) + 1))
    val_edges = sample_labeled_split(
        graph,
        split="val",
        neg_ratio=args.neg_ratio,
        strategy=args.negative_strategy,
        device=device,
        rng=np.random.default_rng(args.seed + 2003 * (SPECIES.index(source) + 1)),
    )

    best_val_auc = -1.0
    best_epoch = 0
    bad_epochs = 0
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, Any]] = []
    print(f"[train {source}] train_pos={graph['train_pos_edge_index_np'].shape[1]}")

    for epoch in range(1, args.epochs + 1):
        train_edges = sample_labeled_split(
            graph,
            split="train",
            neg_ratio=args.neg_ratio,
            strategy=args.negative_strategy,
            device=device,
            rng=train_rng,
        )
        train_loss = train_one_epoch(model, graph, train_edges[0], train_edges[1], optimizer, loss_fn)
        train_metrics = evaluate(model, graph, train_edges[0], train_edges[1], loss_fn, args.threshold)
        val_metrics = evaluate(model, graph, val_edges[0], val_edges[1], loss_fn, args.threshold)
        history.append(
            {
                "epoch": epoch,
                "train_loss_backward": train_loss,
                "train": train_metrics,
                "val": val_metrics,
            }
        )

        val_auc = val_metrics["auc"]
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            bad_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            bad_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or bad_epochs == 0:
            print(
                f"[train {source}] epoch={epoch:03d} train_auc={train_metrics['auc']:.4f} "
                f"val_auc={val_auc:.4f} best_epoch={best_epoch}"
            )
        if bad_epochs >= args.patience:
            print(f"[train {source}] early_stop epoch={epoch} best_epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError(f"No checkpoint state captured for source species {source}.")
    model.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    source_dir = run_dir / "models" / source
    source_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_epoch": best_epoch,
            "best_val_auc": best_val_auc,
        },
        source_dir / "best_model.pt",
    )
    (source_dir / "history.json").write_text(json.dumps(json_safe(history), indent=2), encoding="utf-8")
    return model, {"best_epoch": best_epoch, "best_val_auc": best_val_auc}


def write_metric_matrix(
    rows: list[dict[str, Any]],
    species: list[str],
    metric: str,
    path: Path,
) -> None:
    values = {
        (row["source"], row["target"]): row[f"test_{metric}"]
        for row in rows
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source\\target", *species])
        for source in species:
            writer.writerow([source, *[values[(source, target)] for target in species]])


def write_threshold_matrix(rows: list[dict[str, Any]], species: list[str], path: Path) -> None:
    values = {
        (row["source"], row["target"]): row["best_threshold"]
        for row in rows
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source\\target", *species])
        for source in species:
            writer.writerow([source, *[values[(source, target)] for target in species]])


def plot_heatmap(csv_path: Path, metric: str, output_path: Path) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        targets = header[1:]
        sources: list[str] = []
        values: list[list[float]] = []
        for row in reader:
            sources.append(row[0])
            values.append([float(value) for value in row[1:]])

    matrix = np.asarray(values, dtype=np.float32)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        write_svg_heatmap(matrix, sources, targets, metric, output_path.with_suffix(".svg"))
        return

    fig, ax = plt.subplots(figsize=(6, 5), dpi=160)
    im = ax.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(targets)), labels=targets)
    ax.set_yticks(np.arange(len(sources)), labels=sources)
    ax.set_xlabel("Target species")
    ax.set_ylabel("Source species")
    ax.set_title(f"Species transfer {metric.upper()}")
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[row_idx, col_idx] < 0.55 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def write_svg_heatmap(
    matrix: np.ndarray,
    sources: list[str],
    targets: list[str],
    metric: str,
    output_path: Path,
) -> None:
    cell = 74
    left = 92
    top = 72
    width = left + cell * len(targets) + 24
    height = top + cell * len(sources) + 48

    def color(value: float) -> str:
        if not np.isfinite(value):
            return "#d8dee9"
        value = min(max(float(value), 0.0), 1.0)
        red = int(68 + value * 185)
        green = int(1 + value * 190)
        blue = int(84 - value * 40)
        return f"#{red:02x}{green:02x}{blue:02x}"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" font-family="Arial" font-size="18">'
        f"Species transfer {metric.upper()}</text>",
        f'<text x="{width / 2:.1f}" y="{height - 10}" text-anchor="middle" font-family="Arial" font-size="13">'
        "Target species</text>",
        f'<text x="18" y="{top + cell * len(sources) / 2:.1f}" text-anchor="middle" '
        f'font-family="Arial" font-size="13" transform="rotate(-90 18 {top + cell * len(sources) / 2:.1f})">'
        "Source species</text>",
    ]
    for col_idx, target in enumerate(targets):
        x = left + col_idx * cell + cell / 2
        lines.append(
            f'<text x="{x:.1f}" y="{top - 16}" text-anchor="middle" font-family="Arial" font-size="13">{target}</text>'
        )
    for row_idx, source in enumerate(sources):
        y = top + row_idx * cell + cell / 2
        lines.append(
            f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="13">{source}</text>'
        )
        for col_idx in range(len(targets)):
            x = left + col_idx * cell
            y0 = top + row_idx * cell
            value = float(matrix[row_idx, col_idx])
            label = "nan" if not np.isfinite(value) else f"{value:.3f}"
            text_color = "white" if np.isfinite(value) and value < 0.55 else "black"
            lines.append(
                f'<rect x="{x}" y="{y0}" width="{cell}" height="{cell}" fill="{color(value)}" stroke="white"/>'
            )
            lines.append(
                f'<text x="{x + cell / 2:.1f}" y="{y0 + cell / 2 + 5:.1f}" text-anchor="middle" '
                f'font-family="Arial" font-size="13" fill="{text_color}">{label}</text>'
            )
    lines.append("</svg>")
    output_path.write_text("\n".join(lines), encoding="utf-8")


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    species = args.species
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    graphs = {
        item: load_positive_graph(
            args.data_root / item / "graphsage_inputs.npz",
            device=device,
            directed=args.directed_message_passing,
        )
        for item in species
    }
    config = {key: json_safe(value) for key, value in vars(args).items()}
    config.update({"run_dir": run_dir, "device": str(device)})
    (run_dir / "config.json").write_text(json.dumps(json_safe(config), indent=2), encoding="utf-8")

    loss_fn = nn.BCEWithLogitsLoss()
    rows: list[dict[str, Any]] = []
    source_summaries: dict[str, Any] = {}
    for source in species:
        model, source_summary = train_source_model(source, graphs[source], args, device, run_dir)
        source_summaries[source] = source_summary
        for target in species:
            eval_rng = np.random.default_rng(
                args.seed + 3001 * (SPECIES.index(source) + 1) + 101 * (SPECIES.index(target) + 1)
            )
            edge_label_index, labels = sample_labeled_split(
                graphs[target],
                split="test",
                neg_ratio=args.neg_ratio,
                strategy=args.negative_strategy,
                device=device,
                rng=eval_rng,
            )
            val_rng = np.random.default_rng(
                args.seed + 5003 * (SPECIES.index(source) + 1) + 173 * (SPECIES.index(target) + 1)
            )
            val_edge_label_index, val_labels = sample_labeled_split(
                graphs[target],
                split="val",
                neg_ratio=args.neg_ratio,
                strategy=args.negative_strategy,
                device=device,
                rng=val_rng,
            )
            model.eval()
            with torch.no_grad():
                z = model.encode(graphs[target]["x"], graphs[target]["edge_index"])  # type: ignore[arg-type]
                val_logits = model.decode(z, val_edge_label_index)
                test_logits = model.decode(z, edge_label_index)
            val_probs = torch.sigmoid(val_logits).detach().cpu().numpy()
            val_labels_np = val_labels.detach().cpu().numpy()
            best_threshold, val_best_metrics = best_f1_threshold(val_labels_np, val_probs)
            test_probs = torch.sigmoid(test_logits).detach().cpu().numpy()
            test_labels_np = labels.detach().cpu().numpy()
            metrics = {
                "loss": float(loss_fn(test_logits, labels).item()),
                "auc": binary_auc(test_labels_np, test_probs),
                "ap": average_precision(test_labels_np, test_probs),
            }
            metrics.update(classification_metrics(test_labels_np, test_probs, best_threshold))
            row = {
                "source": source,
                "target": target,
                "source_best_epoch": source_summary["best_epoch"],
                "source_best_val_auc": source_summary["best_val_auc"],
                "best_threshold": best_threshold,
                "val_best_f1": val_best_metrics["f1"],
                "val_best_accuracy": val_best_metrics["accuracy"],
                **{f"test_{key}": value for key, value in metrics.items()},
            }
            rows.append(row)
            print(
                f"[eval {source}->{target}] thr={best_threshold:.3f} "
                f"auc={metrics['auc']:.4f} ap={metrics['ap']:.4f} "
                f"acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}"
            )

    metrics_path = run_dir / "transfer_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config": config,
        "source_summaries": source_summaries,
        "rows": rows,
    }
    (run_dir / "transfer_metrics.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")

    for metric in ["auc", "ap", "accuracy", "f1"]:
        matrix_csv = run_dir / f"transfer_{metric}_matrix.csv"
        write_metric_matrix(rows, species, metric, matrix_csv)
        if not args.no_heatmap:
            plot_heatmap(matrix_csv, metric, run_dir / f"transfer_{metric}_heatmap.png")
    write_threshold_matrix(rows, species, run_dir / "transfer_best_threshold_matrix.csv")

    print(f"Saved transfer run: {run_dir}")
    print(f"Primary heatmap: {run_dir / f'transfer_{args.metric}_heatmap.png'}")


if __name__ == "__main__":
    main()
