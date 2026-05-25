#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG_PATH = ROOT / ".mplconfig"
MPLCONFIG_PATH.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_PATH))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F

sys.path.insert(0, str(ROOT / "src"))
from graphmmi import GraphMMILinkPredictor, load_graph_bundle, \
    pair_feature_dim, pair_feature_dim_v2, pair_feature_dim_v3, \
    pair_feature_matrix, sample_negative_edges
from graphmmi.data import GraphBundle, positive_pair_set


SPECIES_ORDER = ["human", "cow", "mouse", "worm"]
METRICS = ["auc", "aupr", "acc", "f1", "mcc"]
NEGATIVE_STRATEGIES = ["endpoint_corrupt", "random", "uniform", "degree_aware", "sequence_aware"]
ZERO_SHOT_SETTINGS = {"zero_shot", "strict_zero_shot", "calibrated_zero_shot"}


@dataclass
class TrainResult:
    model: GraphMMILinkPredictor
    best_epoch: int
    best_val_aupr: float
    best_val_auc: float


@dataclass
class TrainSummary:
    best_epoch: int
    best_val_aupr: float
    best_val_auc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GraphSAGE/GATv2 and generate 4x4 transfer-learning heatmaps."
    )
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed/graph/random")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/gnn_transfer")
    parser.add_argument("--species", nargs="+", default=SPECIES_ORDER, choices=SPECIES_ORDER)
    parser.add_argument("--encoders", nargs="+", default=["graphsage", "gatv2"], choices=["graphsage", "gatv2"])
    parser.add_argument(
        "--settings",
        nargs="+",
        default=["strict_zero_shot", "calibrated_zero_shot", "finetune"],
        choices=["zero_shot", "strict_zero_shot", "calibrated_zero_shot", "finetune"],
        help=(
            "zero_shot is kept as a backward-compatible alias for calibrated_zero_shot. "
            "strict_zero_shot uses a fixed threshold; calibrated_zero_shot selects the "
            "threshold on target validation data; finetune updates target parameters."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--finetune-patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--neg-strategy", choices=NEGATIVE_STRATEGIES, default="endpoint_corrupt")
    parser.add_argument(
        "--eval-neg-strategy",
        choices=["same", *NEGATIVE_STRATEGIES],
        default="same",
        help="Negative sampling strategy for fixed val/test negatives. 'same' reuses --neg-strategy.",
    )
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument("--eval-neg-ratio", type=float, default=1.0)
    parser.add_argument(
        "--fixed-eval-negatives",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fix and cache validation/test negatives per target species for comparable metrics.",
    )
    parser.add_argument(
        "--refresh-fixed-negatives",
        action="store_true",
        help="Regenerate cached fixed validation/test negatives even if cache files exist.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threshold-metric", choices=["mcc", "f1", "acc"], default="mcc")
    parser.add_argument("--finetune-strategy", choices=["full", "last_layer", "decoder"], default="full")
    parser.add_argument("--graphsage-hidden-dim", type=int, default=128)
    parser.add_argument("--gatv2-hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--residual", action="store_true", help="Use residual connections in GNN layers.")
    parser.add_argument("--layer-norm", action="store_true", help="Use LayerNorm in GNN layers.")
    parser.add_argument("--decoder-layer-norm", action="store_true", help="Use LayerNorm in the link decoder MLP.")
    parser.add_argument("--gat-heads", type=int, default=2)
    parser.add_argument("--gat-concat", action="store_true")
    parser.add_argument("--id-embedding-dim", type=int, default=32)
    parser.add_argument("--type-embedding-dim", type=int, default=8)
    parser.add_argument("--species-embedding-dim", type=int, default=8)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument(
        "--keep-edge-attr-in-memory",
        action="store_true",
        help="Keep original positive-only edge attributes in GraphBundle. Off by default to reduce RAM.",
    )
    parser.add_argument("--edge-attr-mode", choices=["none", "pair"], default="pair")
    parser.add_argument("--pair-feature-version", choices=["v1", "v2", "v3"], default="v3")
    parser.add_argument("--pretrained-encoder", type=Path, default=None)
    parser.add_argument(
        "--use-edge-attr",
        action="store_true",
        help=(
            "Deprecated alias for --edge-attr-mode pair. Pair features are computed for both "
            "positive and dynamic negative edges."
        ),
    )
    parser.add_argument("--mirna-sim-edges", action="store_true",
                        help="Use miRNA-miRNA similarity edges in message passing.")
    parser.add_argument("--mrna-sim-edges", action="store_true",
                        help="Use mRNA-mRNA similarity edges in message passing.")
    parser.add_argument("--mirna-sim-topk", type=int, default=5,
                        help="DEPRECATED in training. Top-k is set during preprocessing (final_embedding.py).")
    parser.add_argument("--mrna-sim-topk", type=int, default=5,
                        help="DEPRECATED in training. Top-k is set during preprocessing (final_embedding.py).")
    parser.add_argument("--no-heatmaps", action="store_true", help="Skip heatmap generation.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_seed(base_seed: int, *parts: str) -> int:
    value = base_seed
    for part in parts:
        for idx, ch in enumerate(part):
            value += (idx + 1) * ord(ch)
    return int(value % (2**31 - 1))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def ensure_preprocessed(args: argparse.Namespace) -> None:
    missing: list[str] = []
    stale: list[str] = []
    for species in args.species:
        path = args.processed_dir / species / "graph_inputs.pt"
        if not path.exists():
            missing.append(species)
            continue
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            feature_names = set(map(str, data.get("node_feature_names", [])))
            if "node_sequences" not in data or "seq_log_length" not in feature_names:
                stale.append(species)
        except Exception:
            stale.append(species)
    needs_refresh = missing + stale
    if not needs_refresh or args.skip_preprocess:
        if needs_refresh:
            raise FileNotFoundError(
                f"Missing or stale graph_inputs.pt for {needs_refresh} under {args.processed_dir}. "
                "Run scripts/preprocess_graph_data.py first."
            )
        return
    print(f"Refreshing graph inputs for {needs_refresh}; running preprocessing first.")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/preprocess_graph_data.py"),
            "--output-dir",
            str(args.processed_dir),
        ],
        cwd=ROOT,
        check=True,
    )


def hidden_dim_for_encoder(args: argparse.Namespace, encoder: str) -> int:
    if encoder == "graphsage":
        return args.graphsage_hidden_dim
    return args.gatv2_hidden_dim


def embedding_dims_for_setting(args: argparse.Namespace, setting: str) -> tuple[int, int]:
    if setting in ZERO_SHOT_SETTINGS:
        return 0, 0
    return args.id_embedding_dim, args.species_embedding_dim


def _pair_feature_dim_for_version(version: str) -> int:
    if version == "v3": return pair_feature_dim_v3()
    if version == "v2": return pair_feature_dim_v2()
    return pair_feature_dim()


def build_model(
    args: argparse.Namespace,
    encoder: str,
    setting: str,
    graph: GraphBundle,
    device: torch.device,
) -> GraphMMILinkPredictor:
    id_dim, species_dim = embedding_dims_for_setting(args, setting)
    edge_attr_dim = _pair_feature_dim_for_version(args.pair_feature_version) if args.edge_attr_mode == "pair" else 0
    use_edge_weight = args.mirna_sim_edges or args.mrna_sim_edges
    return GraphMMILinkPredictor(
        encoder_name=encoder,
        num_numeric_features=int(graph.x.size(1)),
        num_nodes=int(graph.x.size(0)),
        hidden_dim=hidden_dim_for_encoder(args, encoder),
        num_layers=args.num_layers,
        edge_attr_dim=edge_attr_dim,
        dropout=args.dropout,
        gat_heads=args.gat_heads,
        gat_concat=args.gat_concat,
        id_embedding_dim=id_dim,
        type_embedding_dim=args.type_embedding_dim,
        species_embedding_dim=species_dim,
        residual=args.residual,
        layer_norm=args.layer_norm,
        decoder_layer_norm=args.decoder_layer_norm,
        use_edge_weight=use_edge_weight,
    ).to(device)


def compatible_state_dict(
    source_state: dict[str, torch.Tensor],
    target_model: torch.nn.Module,
    skip_prefixes: tuple[str, ...] = (),
) -> dict[str, torch.Tensor]:
    target_state = target_model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    for key, value in source_state.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            continue
        if key in target_state and tuple(target_state[key].shape) == tuple(value.shape):
            compatible[key] = value
    return compatible


def clone_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _load_pretrained_encoder(model, ckpt_path, species, device):
    if ckpt_path.is_dir():
        candidates = sorted(ckpt_path.glob(f"{species}_*.pt"))
        if not candidates:
            print(f"[pretrained] no checkpoint for {species} in {ckpt_path}, skipping")
            return
        ckpt_path = candidates[0]
    if not ckpt_path.exists():
        print(f"[pretrained] {ckpt_path} not found, skipping")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    source_state = ckpt.get("encoder_state_dict") or ckpt.get("model_state_dict") or ckpt
    # Only load encoder weights (input_encoder + gnn), skip decoder
    source_state = {k: v for k, v in source_state.items()
                    if k.startswith("input_encoder.") or k.startswith("gnn.")}
    state = compatible_state_dict(source_state, model)
    missing, unexpected = model.load_state_dict(state, strict=False)
    encoder_missing = [k for k in missing if not k.startswith("decoder.")]
    print(f"[pretrained] {ckpt_path.name}: loaded {len(state)}/{len(source_state)} keys, "
          f"missing={len(missing)} (encoder: {len(encoder_missing)}), unexpected={len(unexpected)}")


def train_summary(train_result: TrainResult) -> TrainSummary:
    return TrainSummary(
        best_epoch=train_result.best_epoch,
        best_val_aupr=train_result.best_val_aupr,
        best_val_auc=train_result.best_val_auc,
    )


def collect_memory(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def resolve_edge_index_for_training(graph: GraphBundle, args: argparse.Namespace) -> tuple[Tensor, Tensor | None]:
    """Return (edge_index, edge_weight) for message passing.

    Dynamically builds the augmented graph by selectively concatenating
    per-type similarity edges based on CLI flags.  This ensures **ablation
    correctness**: ``--mirna-sim-edges`` alone means only miRNA-miRNA edges
    are added, regardless of what similarity data exists in the .pt file.
    """
    use_mirna = args.mirna_sim_edges and graph.similarity_edge_index_mirna is not None
    use_mrna = args.mrna_sim_edges and graph.similarity_edge_index_mrna is not None

    if not (use_mirna or use_mrna):
        return graph.edge_index, None

    # Start from the original undirected interaction edges
    half = graph.edge_index.size(1) // 2
    device = graph.edge_index.device
    ei_parts: list[Tensor] = [graph.edge_index]
    ew_parts: list[Tensor] = [torch.ones(graph.edge_index.size(1), dtype=torch.float32, device=device)]

    if use_mirna:
        ei_parts.append(graph.similarity_edge_index_mirna)
        if graph.similarity_edge_weight_mirna is not None:
            ew_parts.append(graph.similarity_edge_weight_mirna.to(device=device, dtype=torch.float32))
        else:
            ew_parts.append(torch.ones(graph.similarity_edge_index_mirna.size(1), dtype=torch.float32, device=device))

    if use_mrna:
        ei_parts.append(graph.similarity_edge_index_mrna)
        if graph.similarity_edge_weight_mrna is not None:
            ew_parts.append(graph.similarity_edge_weight_mrna.to(device=device, dtype=torch.float32))
        else:
            ew_parts.append(torch.ones(graph.similarity_edge_index_mrna.size(1), dtype=torch.float32, device=device))

    return torch.cat(ei_parts, dim=1), torch.cat(ew_parts, dim=0)


def eval_negative_strategy(args: argparse.Namespace) -> str:
    return args.neg_strategy if args.eval_neg_strategy == "same" else args.eval_neg_strategy


def safe_ratio_name(value: float) -> str:
    return str(value).replace(".", "p")


def fixed_negative_path(args: argparse.Namespace, species: str, split: str) -> Path:
    strategy = eval_negative_strategy(args)
    name = f"{split}_neg_{strategy}_r{safe_ratio_name(args.eval_neg_ratio)}_seed{args.seed}.pt"
    return args.processed_dir / species / "fixed_negatives" / name


def load_or_create_fixed_negatives(
    species: str,
    graph: GraphBundle,
    split: str,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.Tensor:
    path = fixed_negative_path(args, species, split)
    if path.exists() and not args.refresh_fixed_negatives:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        edge_index = payload["edge_index"] if isinstance(payload, dict) else payload
        return edge_index.to(device=device, dtype=torch.long)

    path.parent.mkdir(parents=True, exist_ok=True)
    generator = generator_for(
        device,
        stable_seed(args.seed, "fixed_eval_negative", species, split, eval_negative_strategy(args), str(args.eval_neg_ratio)),
    )
    edge_index = sample_negative_edges(
        graph.split_pos_edge_index[split],
        graph.node_type,
        graph.all_positive_edge_index,
        neg_ratio=args.eval_neg_ratio,
        strategy=eval_negative_strategy(args),
        generator=generator,
        node_sequences=graph.node_sequences,
        blocked_pairs=graph.positive_pair_cache,
    )
    torch.save(
        {
            "edge_index": edge_index.detach().cpu(),
            "species": species,
            "split": split,
            "strategy": eval_negative_strategy(args),
            "eval_neg_ratio": args.eval_neg_ratio,
            "seed": args.seed,
        },
        path,
    )
    return edge_index


def fixed_eval_negatives(
    species: str,
    graph: GraphBundle,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not args.fixed_eval_negatives:
        return {}
    return {
        "val": load_or_create_fixed_negatives(species, graph, "val", args, device),
        "test": load_or_create_fixed_negatives(species, graph, "test", args, device),
    }


def configure_finetune_parameters(model: GraphMMILinkPredictor, strategy: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
    if strategy == "full":
        return
    for parameter in model.input_encoder.parameters():
        parameter.requires_grad = False
    for parameter in model.gnn.parameters():
        parameter.requires_grad = False
    if strategy == "last_layer":
        layers = getattr(model.gnn, "layers", None)
        if layers is not None and len(layers) > 0:
            for parameter in layers[-1].parameters():
                parameter.requires_grad = True
        norms = getattr(model.gnn, "norms", None)
        if norms is not None and len(norms) > 0:
            for parameter in norms[-1].parameters():
                parameter.requires_grad = True
    for parameter in model.decoder.parameters():
        parameter.requires_grad = True


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def roc_auc_score_np(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    rank_sum_pos = float(ranks[labels == 1].sum())
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def average_precision_np(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / (np.arange(len(sorted_labels)) + 1)
    return float((precision * sorted_labels).sum() / n_pos)


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (scores >= threshold).astype(np.int64)
    labels = labels.astype(np.int64)
    tp = float(((pred == 1) & (labels == 1)).sum())
    tn = float(((pred == 0) & (labels == 0)).sum())
    fp = float(((pred == 1) & (labels == 0)).sum())
    fn = float(((pred == 0) & (labels == 1)).sum())
    acc = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    denom = math.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-12))
    mcc = ((tp * tn) - (fp * fn)) / denom
    return {"acc": acc, "f1": f1, "mcc": mcc}


def select_best_threshold(labels: torch.Tensor, logits: torch.Tensor, metric: str, fallback: float) -> tuple[float, dict[str, float]]:
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    scores_np = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
    candidates = np.unique(np.concatenate([scores_np, np.asarray([0.0, fallback, 1.0], dtype=np.float64)]))
    best_threshold = float(fallback)
    best_binary = binary_metrics(labels_np, scores_np, best_threshold)
    best_value = best_binary[metric]
    for threshold in candidates:
        current = binary_metrics(labels_np, scores_np, float(threshold))
        current_value = current[metric]
        if current_value > best_value + 1e-12:
            best_threshold = float(threshold)
            best_binary = current
            best_value = current_value
        elif abs(current_value - best_value) <= 1e-12 and abs(float(threshold) - fallback) < abs(best_threshold - fallback):
            best_threshold = float(threshold)
            best_binary = current
    return best_threshold, best_binary


def compute_metrics(labels: torch.Tensor, logits: torch.Tensor, threshold: float) -> dict[str, float]:
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    probs_np = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float64)
    out = {
        "auc": float(roc_auc_score_np(labels_np, probs_np)),
        "aupr": float(average_precision_np(labels_np, probs_np)),
    }
    out.update(binary_metrics(labels_np, probs_np, threshold))
    return out


def generator_for(device: torch.device, seed: int) -> torch.Generator:
    if device.type == "cuda":
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def build_batch(
    graph: GraphBundle,
    split: str,
    neg_ratio: float,
    seed: int,
    edge_attr_mode: str,
    device: torch.device,
    neg_strategy: str,
    fixed_neg_edge_index: torch.Tensor | None = None,
    edge_attr_version: str = "v1",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if fixed_neg_edge_index is not None:
        cache_key = f"{split}:{edge_attr_mode}:{edge_attr_version}:{id(fixed_neg_edge_index)}"
        cached = graph.batch_cache.get(cache_key)
        if cached is not None:
            return cached
    else:
        cache_key = ""

    generator = generator_for(device, seed)
    pos_edge_index = graph.split_pos_edge_index[split]
    if fixed_neg_edge_index is None:
        neg_edge_index = sample_negative_edges(
            pos_edge_index,
            graph.node_type,
            graph.all_positive_edge_index,
            neg_ratio=neg_ratio,
            strategy=neg_strategy,
            generator=generator,
            node_sequences=graph.node_sequences,
            blocked_pairs=graph.positive_pair_cache,
        )
    else:
        neg_edge_index = fixed_neg_edge_index.to(device=pos_edge_index.device, dtype=torch.long)
    edge_label_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
    labels = torch.cat(
        [
            torch.ones(pos_edge_index.size(1), dtype=torch.float32, device=pos_edge_index.device),
            torch.zeros(neg_edge_index.size(1), dtype=torch.float32, device=pos_edge_index.device),
        ],
        dim=0,
    )
    edge_attr = pair_feature_matrix(edge_label_index, graph, version=edge_attr_version) if edge_attr_mode == "pair" else None
    if edge_attr is not None and edge_attr_version == "v3":
        assert edge_attr.size(1) == 40, f"v3 pair dim mismatch: {edge_attr.size(1)} != 40"
    if edge_attr is not None and edge_attr_version == "v2":
        assert edge_attr.size(1) == 28, f"v2 pair dim mismatch: {edge_attr.size(1)} != 28"
    order = torch.randperm(labels.numel(), device=labels.device)
    edge_label_index = edge_label_index[:, order]
    labels = labels[order]
    if edge_attr is not None:
        edge_attr = edge_attr[order]
    if fixed_neg_edge_index is not None:
        graph.batch_cache[cache_key] = (edge_label_index, labels, edge_attr)
    return edge_label_index, labels, edge_attr


@torch.no_grad()
def predict_split(
    model: GraphMMILinkPredictor,
    graph: GraphBundle,
    split: str,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    fixed_neg_edge_index: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    model.eval()
    edge_label_index, labels, edge_attr = build_batch(
        graph,
        split=split,
        neg_ratio=args.eval_neg_ratio,
        seed=seed,
        edge_attr_mode=args.edge_attr_mode,
        device=device,
        neg_strategy=eval_negative_strategy(args) if split in {"val", "test"} else args.neg_strategy,
        fixed_neg_edge_index=fixed_neg_edge_index,
        edge_attr_version=args.pair_feature_version,
    )
    mp_edge_index, mp_edge_weight = resolve_edge_index_for_training(graph, args)
    logits = model(graph.x, graph.node_type, graph.species_id, mp_edge_index, edge_label_index, edge_attr, edge_weight=mp_edge_weight)
    loss = F.binary_cross_entropy_with_logits(logits, labels)
    return labels, logits, float(loss.item())


@torch.no_grad()
def evaluate(
    model: GraphMMILinkPredictor,
    graph: GraphBundle,
    split: str,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    threshold: float | None = None,
    fixed_neg_edge_index: torch.Tensor | None = None,
) -> dict[str, float]:
    labels, logits, loss = predict_split(model, graph, split, args, seed, device, fixed_neg_edge_index=fixed_neg_edge_index)
    metrics = compute_metrics(labels, logits, args.threshold if threshold is None else threshold)
    metrics["loss"] = loss
    return metrics


@torch.no_grad()
def choose_threshold(
    model: GraphMMILinkPredictor,
    graph: GraphBundle,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    fixed_neg_edge_index: torch.Tensor | None = None,
) -> tuple[float, dict[str, float]]:
    labels, logits, loss = predict_split(model, graph, "val", args, seed, device, fixed_neg_edge_index=fixed_neg_edge_index)
    threshold, binary = select_best_threshold(labels, logits, args.threshold_metric, args.threshold)
    metrics = compute_metrics(labels, logits, threshold)
    metrics["loss"] = loss
    metrics["threshold"] = threshold
    metrics["selection_metric"] = args.threshold_metric
    metrics[f"selected_{args.threshold_metric}"] = binary[args.threshold_metric]
    return threshold, metrics


def train_on_graph(
    args: argparse.Namespace,
    model: GraphMMILinkPredictor,
    graph: GraphBundle,
    encoder: str,
    setting: str,
    species: str,
    epochs: int,
    patience: int,
    lr: float,
    seed: int,
    device: torch.device,
    phase: str,
    eval_negatives: dict[str, torch.Tensor] | None = None,
) -> TrainResult:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("No trainable parameters are left after applying the finetune strategy.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr, weight_decay=args.weight_decay)
    best_state = clone_state_dict(model)
    best_epoch = 0
    best_val = {"aupr": -float("inf"), "auc": -float("inf")}
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        # Curriculum: ramp sequence-aware negatives by training length.
        # This keeps short fine-tuning runs from spending almost all epochs on
        # endpoint_corrupt negatives while preserving the old 40-epoch schedule.
        if args.neg_strategy == "sequence_aware":
            warmup_epochs = max(1, int(round(epochs * 0.25)))
            mixed_epochs = max(warmup_epochs + 1, int(round(epochs * 0.65)))
            if epoch <= warmup_epochs:
                cur_strategy = "endpoint_corrupt"
            elif epoch <= mixed_epochs:
                cur_strategy = "sequence_aware" if epoch % 2 == 0 else "endpoint_corrupt"
            else:
                cur_strategy = "sequence_aware"
        else:
            cur_strategy = args.neg_strategy
        edge_label_index, labels, edge_attr = build_batch(
            graph,
            split="train",
            neg_ratio=args.neg_ratio,
            seed=stable_seed(seed, phase, species, str(epoch)),
            edge_attr_mode=args.edge_attr_mode,
            device=device,
            neg_strategy=cur_strategy,
            edge_attr_version=args.pair_feature_version,
        )
        optimizer.zero_grad(set_to_none=True)
        mp_edge_index, mp_edge_weight = resolve_edge_index_for_training(graph, args)
        logits = model(graph.x, graph.node_type, graph.species_id, mp_edge_index, edge_label_index, edge_attr, edge_weight=mp_edge_weight)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        val_metrics = evaluate(
            model,
            graph,
            split="val",
            args=args,
            seed=stable_seed(seed, phase, species, "val", str(epoch)),
            device=device,
            fixed_neg_edge_index=(eval_negatives or {}).get("val"),
        )
        improved = val_metrics["aupr"] > best_val["aupr"]
        if improved:
            best_state = clone_state_dict(model)
            best_epoch = epoch
            best_val = {"aupr": val_metrics["aupr"], "auc": val_metrics["auc"]}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch == epochs or improved:
            print(
                f"[{encoder}/{setting}/{phase}/{species}] "
                f"epoch={epoch:03d} loss={loss.item():.4f} "
                f"val_aupr={val_metrics['aupr']:.4f} val_auc={val_metrics['auc']:.4f}"
            )
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    return TrainResult(model=model, best_epoch=best_epoch, best_val_aupr=best_val["aupr"], best_val_auc=best_val["auc"])


def load_one_graph(species: str, args: argparse.Namespace, device: torch.device) -> GraphBundle:
    path = args.processed_dir / species / "graph_inputs.pt"
    graph = load_graph_bundle(path, device=device, load_edge_attr=args.keep_edge_attr_in_memory)
    graph.positive_pair_cache = positive_pair_set(graph.all_positive_edge_index)
    return graph


@torch.no_grad()
def threshold_for_setting(
    model: GraphMMILinkPredictor,
    graph: GraphBundle,
    setting: str,
    args: argparse.Namespace,
    seed: int,
    device: torch.device,
    eval_negatives: dict[str, torch.Tensor],
) -> tuple[float, dict[str, float]]:
    if setting == "strict_zero_shot":
        metrics = evaluate(
            model,
            graph,
            "val",
            args,
            seed,
            device,
            threshold=args.threshold,
            fixed_neg_edge_index=eval_negatives.get("val"),
        )
        metrics["threshold"] = args.threshold
        metrics["selection_metric"] = "fixed"
        return args.threshold, metrics
    return choose_threshold(
        model,
        graph,
        args,
        seed,
        device,
        fixed_neg_edge_index=eval_negatives.get("val"),
    )


def _free_graph(graph: GraphBundle | None) -> None:
    """Move a graph bundle back to CPU so Python GC can reclaim GPU/RAM."""
    if graph is None:
        return
    graph.batch_cache.clear()
    graph.pair_feature_cache.clear()
    graph.positive_pair_cache = None
    for attr in ("x", "node_type", "species_id", "edge_index", "all_positive_edge_index"):
        t = getattr(graph, attr, None)
        if isinstance(t, Tensor):
            setattr(graph, attr, t.cpu())
    for split in ("train", "val", "test"):
        t = graph.split_pos_edge_index.get(split)
        if isinstance(t, Tensor):
            graph.split_pos_edge_index[split] = t.cpu()
        t = graph.split_pos_edge_attr.get(split)
        if isinstance(t, Tensor):
            graph.split_pos_edge_attr[split] = t.cpu()


def result_row(
    args: argparse.Namespace,
    encoder: str,
    setting: str,
    source: str,
    target: str,
    phase: str,
    source_train: TrainResult | TrainSummary,
    target_train: TrainResult | None,
    metrics: dict[str, float],
    threshold: float,
    threshold_val_metrics: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "encoder": encoder,
        "setting": setting,
        "source": source,
        "target": target,
        "phase": phase,
        "train_neg_strategy": args.neg_strategy,
        "eval_neg_strategy": eval_negative_strategy(args),
        "fixed_eval_negatives": args.fixed_eval_negatives,
        "finetune_strategy": args.finetune_strategy if setting == "finetune" else "",
        "source_best_epoch": source_train.best_epoch,
        "source_best_val_aupr": source_train.best_val_aupr,
        "source_best_val_auc": source_train.best_val_auc,
        "target_best_epoch": target_train.best_epoch if target_train else 0,
        "target_best_val_aupr": target_train.best_val_aupr if target_train else float("nan"),
        "target_best_val_auc": target_train.best_val_auc if target_train else float("nan"),
        "selected_threshold": threshold,
        "threshold_metric": threshold_val_metrics.get("selection_metric", ""),
        "threshold_val_auc": threshold_val_metrics.get("auc", float("nan")),
        "threshold_val_aupr": threshold_val_metrics.get("aupr", float("nan")),
        "threshold_val_acc": threshold_val_metrics.get("acc", float("nan")),
        "threshold_val_f1": threshold_val_metrics.get("f1", float("nan")),
        "threshold_val_mcc": threshold_val_metrics.get("mcc", float("nan")),
    }
    row.update(metrics)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def metric_matrix(rows: list[dict[str, Any]], encoder: str, setting: str, metric: str, species: list[str]) -> np.ndarray:
    matrix = np.full((len(species), len(species)), np.nan, dtype=np.float64)
    for row in rows:
        if row["encoder"] != encoder or row["setting"] != setting:
            continue
        i = species.index(row["source"])
        j = species.index(row["target"])
        matrix[i, j] = float(row[metric])
    return matrix


def draw_heatmap(
    matrix: np.ndarray,
    species: list[str],
    title: str,
    metric: str,
    output_path: Path,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    own_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.6), dpi=160)
    vmin, vmax = (0.0, 1.0) if metric != "mcc" else (-1.0, 1.0)
    image = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis" if metric != "mcc" else "coolwarm")
    ax.set_xticks(np.arange(len(species)), labels=species, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(species)), labels=species)
    ax.set_xlabel("target species")
    ax.set_ylabel("source species")
    ax.set_title(title)
    for i in range(len(species)):
        for j in range(len(species)):
            value = matrix[i, j]
            text = "nan" if np.isnan(value) else f"{value:.3f}"
            ax.text(j, i, text, ha="center", va="center", color="white" if metric != "mcc" else "black", fontsize=8)
    if own_fig:
        cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_ylabel(metric.upper(), rotation=-90, va="bottom")
        ax.figure.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(output_path, bbox_inches="tight")
        plt.close(ax.figure)
    return ax


def draw_all_heatmaps(rows: list[dict[str, Any]], args: argparse.Namespace, run_dir: Path) -> None:
    heatmap_dir = run_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    for encoder in args.encoders:
        for setting in args.settings:
            fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.0), dpi=160)
            if len(METRICS) == 1:
                axes = [axes]
            for ax, metric in zip(axes, METRICS):
                matrix = metric_matrix(rows, encoder, setting, metric, args.species)
                draw_heatmap(
                    matrix,
                    args.species,
                    f"{encoder} {setting} {metric.upper()}",
                    metric,
                    heatmap_dir / f"{encoder}_{setting}_{metric}.png",
                )
                draw_heatmap(
                    matrix,
                    args.species,
                    f"{metric.upper()}",
                    metric,
                    heatmap_dir / "_unused.png",
                    ax=ax,
                )
            fig.suptitle(f"{encoder} / {setting}", y=1.02)
            fig.tight_layout()
            fig.savefig(heatmap_dir / f"{encoder}_{setting}_all_metrics.png", bbox_inches="tight")
            plt.close(fig)


def run_experiments(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    device = torch.device(args.device)
    rows: list[dict[str, Any]] = []

    for encoder in args.encoders:
        for setting in args.settings:
            source_summaries: dict[str, TrainSummary] = {}
            source_states: dict[str, dict[str, torch.Tensor]] = {}

            # --- source training (load one graph at a time) ---
            for source in args.species:
                src_graph = load_one_graph(source, args, device)
                src_eval_negatives = fixed_eval_negatives(source, src_graph, args, device)
                set_seed(stable_seed(args.seed, encoder, setting, source, "source"))
                model = build_model(args, encoder, setting, src_graph, device)
                if args.pretrained_encoder is not None:
                    _load_pretrained_encoder(model, args.pretrained_encoder, source, device)
                train_result = train_on_graph(
                    args=args, model=model, graph=src_graph,
                    encoder=encoder, setting=setting, species=source,
                    epochs=args.epochs, patience=args.patience, lr=args.lr,
                    seed=stable_seed(args.seed, encoder, setting, source),
                    device=device, phase="source",
                    eval_negatives=src_eval_negatives,
                )
                source_states[source] = clone_state_dict(train_result.model)
                source_summaries[source] = train_summary(train_result)
                if args.save_models:
                    model_dir = run_dir / "models" / encoder / setting
                    model_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(source_states[source], model_dir / f"{source}_source.pt")
                del model, train_result, src_eval_negatives
                _free_graph(src_graph)
                del src_graph
                collect_memory(device)

            # --- evaluation (load target graphs one at a time) ---
            for source in args.species:
                for target in args.species:
                    tgt_graph = load_one_graph(target, args, device)
                    tgt_eval_negatives = fixed_eval_negatives(target, tgt_graph, args, device)
                    eval_seed = stable_seed(args.seed, encoder, setting, source, target, "test")
                    if setting in ZERO_SHOT_SETTINGS:
                        model = build_model(args, encoder, setting, tgt_graph, device)
                        state = compatible_state_dict(source_states[source], model)
                        model.load_state_dict(state, strict=False)
                        threshold, threshold_val_metrics = threshold_for_setting(
                            model, tgt_graph,
                            setting,
                            args,
                            stable_seed(args.seed, encoder, setting, source, target, "threshold"),
                            device,
                            tgt_eval_negatives,
                        )
                        metrics = evaluate(
                            model,
                            tgt_graph,
                            "test",
                            args,
                            eval_seed,
                            device,
                            threshold=threshold,
                            fixed_neg_edge_index=tgt_eval_negatives.get("test"),
                        )
                        rows.append(result_row(args, encoder, setting, source, target,
                            "zero_shot_eval", source_summaries[source], None,
                            metrics, threshold, threshold_val_metrics))
                        del model
                    else:
                        set_seed(stable_seed(args.seed, encoder, setting, source, target, "finetune"))
                        target_model = build_model(args, encoder, setting, tgt_graph, device)
                        skip_prefixes = (
                            ("input_encoder.id_embedding", "input_encoder.species_embedding")
                            if source != target
                            else ()
                        )
                        state = compatible_state_dict(source_states[source], target_model, skip_prefixes=skip_prefixes)
                        target_model.load_state_dict(state, strict=False)
                        configure_finetune_parameters(target_model, args.finetune_strategy)
                        target_train = train_on_graph(
                            args=args, model=target_model, graph=tgt_graph,
                            encoder=encoder, setting=setting,
                            species=f"{source}->{target}",
                            epochs=args.finetune_epochs, patience=args.finetune_patience,
                            lr=args.finetune_lr,
                            seed=stable_seed(args.seed, encoder, setting, source, target),
                            device=device, phase="finetune",
                            eval_negatives=tgt_eval_negatives,
                        )
                        threshold, threshold_val_metrics = threshold_for_setting(
                            target_train.model, tgt_graph,
                            setting,
                            args,
                            stable_seed(args.seed, encoder, setting, source, target, "threshold"),
                            device,
                            tgt_eval_negatives,
                        )
                        metrics = evaluate(
                            target_train.model,
                            tgt_graph,
                            "test",
                            args,
                            eval_seed,
                            device,
                            threshold=threshold,
                            fixed_neg_edge_index=tgt_eval_negatives.get("test"),
                        )
                        rows.append(result_row(args, encoder, setting, source, target,
                            "finetune_eval", source_summaries[source], target_train,
                            metrics, threshold, threshold_val_metrics))
                        del target_model, target_train
                    print(
                        f"[RESULT] {encoder}/{setting} {source}->{target} "
                        f"AUC={rows[-1]['auc']:.4f} AUPR={rows[-1]['aupr']:.4f} "
                        f"F1={rows[-1]['f1']:.4f} MCC={rows[-1]['mcc']:.4f} "
                        f"thr={rows[-1]['selected_threshold']:.4f}"
                    )
                    # Free target graph after each evaluation pair
                    del tgt_eval_negatives
                    _free_graph(tgt_graph)
                    del tgt_graph
                    collect_memory(device)

            # --- free source results before next setting to save RAM ---
            del source_summaries, source_states
            collect_memory(device)
    return rows


def main() -> None:
    args = parse_args()
    if args.use_edge_attr:
        args.edge_attr_mode = "pair"
    set_seed(args.seed)
    if not args.skip_preprocess:
        raise RuntimeError(
            "For final experiments, run final_embedding.py manually first, "
            "then pass --skip-preprocess.  The built-in ensure_preprocessed() "
            "runs the OLD preprocess_graph_data.py, not final_embedding.py.")
    ensure_preprocessed(args)

    print(f"[CONFIG] pair_feature_version={args.pair_feature_version} "
          f"dim={_pair_feature_dim_for_version(args.pair_feature_version)}", flush=True)
    run_dir = args.run_root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(json_safe(vars(args)), indent=2), encoding="utf-8")

    rows = run_experiments(args, run_dir)
    write_csv(run_dir / "transfer_metrics.csv", rows)
    (run_dir / "transfer_metrics.json").write_text(json.dumps(json_safe(rows), indent=2), encoding="utf-8")
    if not args.no_heatmaps:
        draw_all_heatmaps(rows, args, run_dir)
    print(f"Saved GNN transfer run to: {run_dir}")


if __name__ == "__main__":
    main()
