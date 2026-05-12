#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
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
        "PyTorch is not installed in this environment. Run this script in the same "
        "environment used for training, or install torch first."
    ) from exc

from mti_graphsage import GraphSAGENodePairPredictor


SPECIES = ["human", "cow", "mouse", "worm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose transfer score distributions, PR curves, and probability calibration."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--species", nargs="+", default=SPECIES, choices=SPECIES)
    parser.add_argument("--source", choices=SPECIES, default=None)
    parser.add_argument("--target", choices=SPECIES, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--neg-ratio", type=float, default=None)
    parser.add_argument("--negative-strategy", choices=["uniform", "endpoint_corrupt"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    parser.add_argument("--directed-message-passing", action="store_true")
    parser.add_argument("--hist-bins", type=int, default=80)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def load_source_model(
    run_dir: Path,
    source: str,
    node_feature_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    decoder_hidden_dim: int,
    device: torch.device,
) -> GraphSAGENodePairPredictor:
    model = GraphSAGENodePairPredictor(
        node_feature_dim=node_feature_dim,
        hidden_channels=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        decoder_hidden_channels=decoder_hidden_dim,
    ).to(device)
    checkpoint_path = run_dir / "models" / source / "best_model.pt"
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_scores(
    model: GraphSAGENodePairPredictor,
    graph: dict[str, Tensor | np.ndarray],
    edge_label_index: Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    z = model.encode(graph["x"], graph["edge_index"])  # type: ignore[arg-type]
    logits = model.decode(z, edge_label_index)
    probs = torch.sigmoid(logits)
    return logits.detach().cpu().numpy(), probs.detach().cpu().numpy()


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


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


def precision_recall_curve(labels: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    sorted_labels = labels[order].astype(np.int64)
    sorted_scores = scores[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int((labels == 1).sum()), 1)
    keep = np.r_[True, sorted_scores[1:] != sorted_scores[:-1]]
    return precision[keep], recall[keep], sorted_scores[keep]


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, _ = precision_recall_curve(labels, scores)
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


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
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def best_f1_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = np.unique(probs.astype(np.float64))
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


def brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - labels) ** 2))


def log_loss(labels: np.ndarray, probs: np.ndarray) -> float:
    clipped = np.clip(probs, 1e-15, 1.0 - 1e-15)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


def expected_calibration_error(labels: np.ndarray, probs: np.ndarray, n_bins: int) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (probs >= start) & (probs < end)
        if end == 1.0:
            mask = (probs >= start) & (probs <= end)
        if not np.any(mask):
            continue
        conf = float(probs[mask].mean())
        acc = float(labels[mask].mean())
        ece += float(mask.mean()) * abs(acc - conf)
    return float(ece)


def fit_platt_temperature(labels: np.ndarray, logits: np.ndarray) -> tuple[float, float]:
    best_scale = 1.0
    best_bias = 0.0
    best_loss = float("inf")
    scales = np.logspace(-2, 2, 121)
    biases = np.linspace(-10.0, 10.0, 201)
    for scale in scales:
        scaled_logits = logits / scale
        for bias in biases:
            probs = sigmoid(scaled_logits + bias)
            loss = log_loss(labels, probs)
            if loss < best_loss:
                best_loss = loss
                best_scale = float(scale)
                best_bias = float(bias)
    return best_scale, best_bias


def apply_platt(logits: np.ndarray, scale: float, bias: float) -> np.ndarray:
    return sigmoid(logits / scale + bias)


def fit_histogram_calibrator(labels: np.ndarray, probs: np.ndarray, n_bins: int) -> dict[str, list[float]]:
    order = np.argsort(probs)
    sorted_probs = probs[order]
    sorted_labels = labels[order]
    chunks = np.array_split(np.arange(len(sorted_probs)), n_bins)
    thresholds: list[float] = []
    values: list[float] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        thresholds.append(float(sorted_probs[chunk[-1]]))
        value = float(sorted_labels[chunk].mean())
        values.append(min(max(value, 1e-6), 1.0 - 1e-6))
    if thresholds:
        thresholds[-1] = 1.0
    return {"thresholds": thresholds, "values": values}


def apply_histogram_calibrator(probs: np.ndarray, calibrator: dict[str, list[float]]) -> np.ndarray:
    thresholds = np.asarray(calibrator["thresholds"], dtype=np.float64)
    values = np.asarray(calibrator["values"], dtype=np.float64)
    indices = np.searchsorted(thresholds, probs, side="left")
    indices = np.clip(indices, 0, len(values) - 1)
    return values[indices]


def score_summary(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label_name, label_value in [("positive", 1), ("negative", 0)]:
        subset = probs[labels == label_value]
        result[label_name] = {
            "n": int(len(subset)),
            "min": float(subset.min()) if len(subset) else None,
            "p01": float(np.quantile(subset, 0.01)) if len(subset) else None,
            "p25": float(np.quantile(subset, 0.25)) if len(subset) else None,
            "median": float(np.quantile(subset, 0.50)) if len(subset) else None,
            "p75": float(np.quantile(subset, 0.75)) if len(subset) else None,
            "p99": float(np.quantile(subset, 0.99)) if len(subset) else None,
            "max": float(subset.max()) if len(subset) else None,
            "mean": float(subset.mean()) if len(subset) else None,
        }
    return result


def save_score_csv(path: Path, labels: np.ndarray, logits: np.ndarray, probs: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "logit", "sigmoid_score"])
        for label, logit, prob in zip(labels.astype(int), logits, probs):
            writer.writerow([int(label), float(logit), float(prob)])


def plot_distribution(
    path: Path,
    labels: np.ndarray,
    probs: np.ndarray,
    title: str,
    bins: int,
) -> None:
    import matplotlib.pyplot as plt

    pos = probs[labels == 1]
    neg = probs[labels == 0]
    positive_floor = min(float(probs[probs > 0].min()) if np.any(probs > 0) else 1e-300, 1e-6)
    log_pos = np.log10(np.clip(pos, positive_floor, 1.0))
    log_neg = np.log10(np.clip(neg, positive_floor, 1.0))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    axes[0].hist(neg, bins=bins, alpha=0.55, density=True, label="negative")
    axes[0].hist(pos, bins=bins, alpha=0.55, density=True, label="positive")
    axes[0].set_xlabel("sigmoid score")
    axes[0].set_ylabel("density")
    axes[0].legend()
    axes[0].set_title("Raw score scale")
    axes[1].hist(log_neg, bins=bins, alpha=0.55, density=True, label="negative")
    axes[1].hist(log_pos, bins=bins, alpha=0.55, density=True, label="positive")
    axes[1].set_xlabel("log10(sigmoid score)")
    axes[1].set_ylabel("density")
    axes[1].legend()
    axes[1].set_title("Log score scale")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_pr_curve(path: Path, labels: np.ndarray, probs: np.ndarray, title: str) -> dict[str, float]:
    import matplotlib.pyplot as plt

    precision, recall, thresholds = precision_recall_curve(labels, probs)
    ap = average_precision(labels, probs)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=160)
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(f"{title}\nAP={ap:.3f}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return {
        "ap": ap,
        "max_precision": float(precision.max()) if len(precision) else 0.0,
        "precision_at_recall_50": precision_at_recall(precision, recall, 0.50),
        "precision_at_recall_80": precision_at_recall(precision, recall, 0.80),
        "precision_at_recall_95": precision_at_recall(precision, recall, 0.95),
        "min_threshold": float(thresholds.min()) if len(thresholds) else None,
        "max_threshold": float(thresholds.max()) if len(thresholds) else None,
    }


def precision_at_recall(precision: np.ndarray, recall: np.ndarray, target_recall: float) -> float:
    mask = recall >= target_recall
    if not np.any(mask):
        return 0.0
    return float(precision[mask].max())


def plot_calibration(
    path: Path,
    labels: np.ndarray,
    raw_probs: np.ndarray,
    platt_probs: np.ndarray,
    hist_probs: np.ndarray,
    n_bins: int,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5), dpi=160)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="ideal")
    for name, probs in [
        ("raw", raw_probs),
        ("platt", platt_probs),
        ("histogram", hist_probs),
    ]:
        xs, ys = calibration_points(labels, probs, n_bins)
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=name)
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed positive rate")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def calibration_points(labels: np.ndarray, probs: np.ndarray, n_bins: int) -> tuple[list[float], list[float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    xs: list[float] = []
    ys: list[float] = []
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (probs >= start) & (probs < end)
        if end == 1.0:
            mask = (probs >= start) & (probs <= end)
        if not np.any(mask):
            continue
        xs.append(float(probs[mask].mean()))
        ys.append(float(labels[mask].mean()))
    return xs, ys


def diagnostic_for_pair(
    source: str,
    target: str,
    graph: dict[str, Tensor | np.ndarray],
    run_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    node_feature_dim = int(graph["x"].size(1))  # type: ignore[union-attr]
    model = load_source_model(
        run_dir=run_dir,
        source=source,
        node_feature_dim=node_feature_dim,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        decoder_hidden_dim=int(config["decoder_hidden_dim"]),
        device=device,
    )
    seed = int(config["seed"]) if args.seed is None else args.seed
    source_idx = SPECIES.index(source) + 1
    target_idx = SPECIES.index(target) + 1
    val_rng = np.random.default_rng(seed + 5003 * source_idx + 173 * target_idx)
    test_rng = np.random.default_rng(seed + 3001 * source_idx + 101 * target_idx)
    neg_ratio = float(config["neg_ratio"]) if args.neg_ratio is None else args.neg_ratio
    strategy = str(config["negative_strategy"]) if args.negative_strategy is None else args.negative_strategy

    val_edges, val_labels_t = sample_labeled_split(graph, "val", neg_ratio, strategy, device, val_rng)
    test_edges, test_labels_t = sample_labeled_split(graph, "test", neg_ratio, strategy, device, test_rng)
    val_logits, val_probs = predict_scores(model, graph, val_edges)
    test_logits, test_probs = predict_scores(model, graph, test_edges)
    val_labels = val_labels_t.detach().cpu().numpy()
    test_labels = test_labels_t.detach().cpu().numpy()

    threshold, val_best = best_f1_threshold(val_labels, val_probs)
    raw_test_metrics = classification_metrics(test_labels, test_probs, threshold)
    raw_auc = binary_auc(test_labels, test_probs)
    raw_ap = average_precision(test_labels, test_probs)

    scale, bias = fit_platt_temperature(val_labels, val_logits)
    platt_val = apply_platt(val_logits, scale, bias)
    platt_test = apply_platt(test_logits, scale, bias)
    platt_threshold, platt_val_best = best_f1_threshold(val_labels, platt_val)
    platt_test_metrics = classification_metrics(test_labels, platt_test, platt_threshold)

    hist_calibrator = fit_histogram_calibrator(val_labels, val_probs, args.calibration_bins)
    hist_val = apply_histogram_calibrator(val_probs, hist_calibrator)
    hist_test = apply_histogram_calibrator(test_probs, hist_calibrator)
    hist_threshold, hist_val_best = best_f1_threshold(val_labels, hist_val)
    hist_test_metrics = classification_metrics(test_labels, hist_test, hist_threshold)

    pair_dir = output_dir / f"{source}_to_{target}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    save_score_csv(pair_dir / "validation_scores.csv", val_labels, val_logits, val_probs)
    save_score_csv(pair_dir / "test_scores.csv", test_labels, test_logits, test_probs)
    plot_distribution(
        pair_dir / "validation_score_distribution.png",
        val_labels,
        val_probs,
        f"{source}->{target} validation score distribution",
        args.hist_bins,
    )
    pr_summary = plot_pr_curve(
        pair_dir / "validation_pr_curve.png",
        val_labels,
        val_probs,
        f"{source}->{target} validation PR curve",
    )
    plot_calibration(
        pair_dir / "calibration_curve.png",
        test_labels,
        test_probs,
        platt_test,
        hist_test,
        args.calibration_bins,
        f"{source}->{target} test calibration",
    )

    summary = {
        "source": source,
        "target": target,
        "split_for_distribution": "val",
        "raw": {
            "best_threshold_from_val": threshold,
            "val_best_metrics": val_best,
            "test_metrics_at_val_threshold": raw_test_metrics,
            "test_auc": raw_auc,
            "test_ap": raw_ap,
            "val_score_summary": score_summary(val_labels, val_probs),
            "test_score_summary": score_summary(test_labels, test_probs),
            "test_brier": brier_score(test_labels, test_probs),
            "test_log_loss": log_loss(test_labels, test_probs),
            "test_ece": expected_calibration_error(test_labels, test_probs, args.calibration_bins),
        },
        "pr_curve": pr_summary,
        "platt_scaling": {
            "temperature": scale,
            "bias": bias,
            "best_threshold_from_val": platt_threshold,
            "val_best_metrics": platt_val_best,
            "test_metrics_at_val_threshold": platt_test_metrics,
            "test_brier": brier_score(test_labels, platt_test),
            "test_log_loss": log_loss(test_labels, platt_test),
            "test_ece": expected_calibration_error(test_labels, platt_test, args.calibration_bins),
        },
        "histogram_calibration": {
            "calibrator": hist_calibrator,
            "best_threshold_from_val": hist_threshold,
            "val_best_metrics": hist_val_best,
            "test_metrics_at_val_threshold": hist_test_metrics,
            "test_brier": brier_score(test_labels, hist_test),
            "test_log_loss": log_loss(test_labels, hist_test),
            "test_ece": expected_calibration_error(test_labels, hist_test, args.calibration_bins),
        },
    }
    (pair_dir / "diagnostics.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    return summary


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
    config = load_json(args.run_dir / "config.json")
    data_root = args.data_root or Path(config["data_root"])
    output_dir = args.output_dir or (args.run_dir / "diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(config["seed"]) if args.seed is None else args.seed
    set_seed(seed)
    device_arg = str(config["device"]) if args.device is None else args.device
    device = resolve_device("cpu" if device_arg == "cpu" else device_arg)
    directed = bool(config.get("directed_message_passing", False)) or args.directed_message_passing

    sources = [args.source] if args.source else args.species
    targets = [args.target] if args.target else args.species
    graphs = {
        target: load_positive_graph(data_root / target / "graphsage_inputs.npz", device, directed)
        for target in targets
    }

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for source in sources:
        for target in targets:
            print(f"[diagnose {source}->{target}]")
            summary = diagnostic_for_pair(source, target, graphs[target], args.run_dir, output_dir, config, args, device)
            summaries.append(summary)
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "raw_threshold": summary["raw"]["best_threshold_from_val"],
                    "raw_test_auc": summary["raw"]["test_auc"],
                    "raw_test_ap": summary["raw"]["test_ap"],
                    "raw_test_f1": summary["raw"]["test_metrics_at_val_threshold"]["f1"],
                    "raw_test_precision": summary["raw"]["test_metrics_at_val_threshold"]["precision"],
                    "raw_test_recall": summary["raw"]["test_metrics_at_val_threshold"]["recall"],
                    "raw_test_brier": summary["raw"]["test_brier"],
                    "raw_test_ece": summary["raw"]["test_ece"],
                    "platt_temperature": summary["platt_scaling"]["temperature"],
                    "platt_bias": summary["platt_scaling"]["bias"],
                    "platt_test_f1": summary["platt_scaling"]["test_metrics_at_val_threshold"]["f1"],
                    "platt_test_brier": summary["platt_scaling"]["test_brier"],
                    "platt_test_ece": summary["platt_scaling"]["test_ece"],
                    "hist_test_f1": summary["histogram_calibration"]["test_metrics_at_val_threshold"]["f1"],
                    "hist_test_brier": summary["histogram_calibration"]["test_brier"],
                    "hist_test_ece": summary["histogram_calibration"]["test_ece"],
                }
            )

    with (output_dir / "diagnostic_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "diagnostic_summary.json").write_text(json.dumps(json_safe(summaries), indent=2), encoding="utf-8")
    print(f"Saved diagnostics: {output_dir}")


if __name__ == "__main__":
    main()
