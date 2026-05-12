#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FILE_RE = re.compile(r"^(?P<dataset_id>(?P<species>[A-Za-z]+)\d+)_pos\.csv$")
SPECIES_ORDER = ["human", "mouse", "worm", "cow"]

BASES = ["A", "C", "G", "U"]
DINUCS = [a + b for a in BASES for b in BASES]
NODE_FEATURE_COLUMNS = (
    [f"seq_1mer_{base}" for base in BASES]
    + [f"seq_2mer_{dinuc}" for dinuc in DINUCS]
    + ["seq_length", "seq_gc_content"]
)

REQUIRED_COLUMNS = [
    "microRNA_name",
    "miRNA sequence",
    "mRNA_name",
    "full_mrna",
]


@dataclass(frozen=True)
class RawPosFile:
    path: Path
    species: str
    dataset_id: str
    columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe miRNA-mRNA GraphSAGE inputs from positive interactions "
            "and freshly sampled negative edges."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=Path("data/external"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/graphsage_mrna/random"))
    parser.add_argument(
        "--split-strategy",
        choices=["random", "cold_mirna", "cold_mrna"],
        default="random",
        help="Positive edge split strategy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.72)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--neg-ratio", type=float, default=1.0)
    parser.add_argument(
        "--negative-strategy",
        choices=["endpoint_corrupt", "uniform"],
        default="endpoint_corrupt",
        help=(
            "endpoint_corrupt keeps one endpoint from a positive edge when sampling negatives, "
            "which reduces degree leakage versus uniform pair sampling."
        ),
    )
    return parser.parse_args()


def clean_text(value: object) -> str:
    if pd.isna(value):
        return "NA"
    text = str(value).strip()
    return text if text else "NA"


def normalize_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    seq = str(value).upper().replace("T", "U")
    return re.sub(r"\s+", "", seq)


def short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def discover_pos_files(input_dir: Path) -> list[RawPosFile]:
    files: list[RawPosFile] = []
    for path in sorted(input_dir.glob("*_pos.csv")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        files.append(
            RawPosFile(
                path=path,
                species=match.group("species").lower(),
                dataset_id=match.group("dataset_id").lower(),
                columns=columns,
            )
        )
    if not files:
        raise FileNotFoundError(f"No *_pos.csv files found in {input_dir}")
    return files


def sequence_feature_matrix(sequences: Iterable[str]) -> np.ndarray:
    base_index = {base: i for i, base in enumerate(BASES)}
    dinuc_index = {dinuc: i for i, dinuc in enumerate(DINUCS)}
    sequences = list(sequences)
    features = np.zeros((len(sequences), len(NODE_FEATURE_COLUMNS)), dtype=np.float32)

    for row_idx, seq in enumerate(sequences):
        valid_bases = [base for base in seq if base in base_index]
        length = len(valid_bases)
        if length:
            counts = np.zeros(len(BASES), dtype=np.float32)
            for base in valid_bases:
                counts[base_index[base]] += 1.0
            features[row_idx, : len(BASES)] = counts / float(length)
            features[row_idx, -2] = float(length)
            features[row_idx, -1] = float(counts[base_index["G"]] + counts[base_index["C"]]) / float(length)

        dinuc_counts = np.zeros(len(DINUCS), dtype=np.float32)
        dinuc_total = 0
        for left, right in zip(seq, seq[1:]):
            dinuc = left + right
            if dinuc in dinuc_index:
                dinuc_counts[dinuc_index[dinuc]] += 1.0
                dinuc_total += 1
        if dinuc_total:
            start = len(BASES)
            features[row_idx, start : start + len(DINUCS)] = dinuc_counts / float(dinuc_total)

    return features


def read_positive_file(raw_file: RawPosFile) -> pd.DataFrame:
    usecols = REQUIRED_COLUMNS
    df = pd.read_csv(raw_file.path, usecols=usecols, low_memory=False)
    row_numbers = np.arange(len(df))
    df["species"] = raw_file.species
    df["dataset_id"] = raw_file.dataset_id
    df["sample_id"] = [f"{raw_file.dataset_id}_pos_{i:07d}" for i in row_numbers]

    mirna_name = df["microRNA_name"].map(clean_text)
    mrna_name = df["mRNA_name"].map(clean_text)
    df["mirna_sequence"] = df["miRNA sequence"].map(normalize_sequence)
    df["mrna_sequence"] = df["full_mrna"].map(normalize_sequence)
    df["mirna_id"] = raw_file.species + "|" + mirna_name
    df["mrna_id"] = raw_file.species + "|" + mrna_name
    df["label"] = 1
    return df


def split_random_edges(
    edges: pd.DataFrame,
    rng: np.random.Generator,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.Series:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    indices = edges.index.to_numpy()
    rng.shuffle(indices)
    n_total = len(indices)
    n_val = int(round(n_total * val_ratio))
    n_test = int(round(n_total * test_ratio))
    n_train = n_total - n_val - n_test
    split = pd.Series(index=edges.index, dtype="object")
    split.loc[indices[:n_train]] = "train"
    split.loc[indices[n_train : n_train + n_val]] = "val"
    split.loc[indices[n_train + n_val :]] = "test"
    return split


def split_group_cold(
    edges: pd.DataFrame,
    group_column: str,
    rng: np.random.Generator,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.Series:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    group_counts = edges.groupby(group_column).size().reset_index(name="n_edges")
    group_counts = group_counts.sample(frac=1.0, random_state=int(rng.integers(0, 2**32 - 1)))
    n_total_edges = int(group_counts["n_edges"].sum())
    target_test = int(round(n_total_edges * test_ratio))
    target_val = int(round(n_total_edges * val_ratio))

    test_groups: set[str] = set()
    val_groups: set[str] = set()
    test_edges = 0
    val_edges = 0
    for row in group_counts.itertuples(index=False):
        group = str(getattr(row, group_column))
        n_edges = int(row.n_edges)
        if test_edges < target_test:
            test_groups.add(group)
            test_edges += n_edges
        elif val_edges < target_val:
            val_groups.add(group)
            val_edges += n_edges
        else:
            break

    split = pd.Series("train", index=edges.index, dtype="object")
    split.loc[edges[group_column].astype(str).isin(test_groups)] = "test"
    split.loc[edges[group_column].astype(str).isin(val_groups)] = "val"
    return split


def build_nodes(edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    mirna_nodes = (
        edges[["mirna_id", "species", "microRNA_name", "mirna_sequence"]]
        .drop_duplicates("mirna_id", keep="first")
        .rename(columns={"mirna_id": "node_id", "microRNA_name": "source_name", "mirna_sequence": "sequence"})
    )
    mirna_nodes["node_type"] = "mirna"
    mirna_nodes["node_type_id"] = 0

    mrna_nodes = (
        edges[["mrna_id", "species", "mRNA_name", "mrna_sequence"]]
        .drop_duplicates("mrna_id", keep="first")
        .rename(columns={"mrna_id": "node_id", "mRNA_name": "source_name", "mrna_sequence": "sequence"})
    )
    mrna_nodes["node_type"] = "mrna"
    mrna_nodes["node_type_id"] = 1

    nodes = pd.concat([mirna_nodes, mrna_nodes], ignore_index=True, sort=False)
    nodes.insert(0, "node_idx", np.arange(len(nodes), dtype=np.int64))
    nodes["sequence_hash"] = nodes["sequence"].fillna("").astype(str).map(short_hash)

    node_features = sequence_feature_matrix(nodes["sequence"].fillna("").astype(str))
    for idx, feature_name in enumerate(NODE_FEATURE_COLUMNS):
        nodes[feature_name] = node_features[:, idx]

    node_index = dict(zip(nodes["node_id"], nodes["node_idx"]))
    mirna_index = {
        node_id: node_idx
        for node_id, node_idx in node_index.items()
        if nodes.loc[node_idx, "node_type"] == "mirna"
    }
    mrna_index = {
        node_id: node_idx
        for node_id, node_idx in node_index.items()
        if nodes.loc[node_idx, "node_type"] == "mrna"
    }
    return nodes, mirna_index, mrna_index


def sample_uniform_negative_pairs(
    mirna_pool: list[str],
    mrna_pool: list[str],
    positive_pairs: set[tuple[str, str]],
    n_samples: int,
    rng: np.random.Generator,
    used_negative_pairs: set[tuple[str, str]],
) -> list[tuple[str, str]]:
    if not mirna_pool or not mrna_pool:
        raise ValueError("Negative sampling pools must not be empty.")

    possible = len(mirna_pool) * len(mrna_pool)
    blocked_in_pool = sum(1 for pair in positive_pairs if pair[0] in mirna_pool and pair[1] in mrna_pool)
    available = possible - blocked_in_pool - sum(
        1 for pair in used_negative_pairs if pair[0] in mirna_pool and pair[1] in mrna_pool
    )
    if available < n_samples:
        raise ValueError(
            f"Not enough negative candidates: requested={n_samples}, available={available}, "
            f"mirnas={len(mirna_pool)}, mrnas={len(mrna_pool)}"
        )

    sampled: list[tuple[str, str]] = []
    local_seen: set[tuple[str, str]] = set()
    max_attempts = max(10000, n_samples * 100)
    attempts = 0
    while len(sampled) < n_samples and attempts < max_attempts:
        attempts += 1
        mirna_id = mirna_pool[int(rng.integers(0, len(mirna_pool)))]
        mrna_id = mrna_pool[int(rng.integers(0, len(mrna_pool)))]
        pair = (mirna_id, mrna_id)
        if pair in positive_pairs or pair in used_negative_pairs or pair in local_seen:
            continue
        sampled.append(pair)
        local_seen.add(pair)

    if len(sampled) < n_samples:
        candidates = [
            (mirna_id, mrna_id)
            for mirna_id in mirna_pool
            for mrna_id in mrna_pool
            if (mirna_id, mrna_id) not in positive_pairs
            and (mirna_id, mrna_id) not in used_negative_pairs
            and (mirna_id, mrna_id) not in local_seen
        ]
        rng.shuffle(candidates)
        sampled.extend(candidates[: n_samples - len(sampled)])

    if len(sampled) != n_samples:
        raise RuntimeError(f"Failed to sample {n_samples} negatives; sampled {len(sampled)}")
    used_negative_pairs.update(sampled)
    return sampled


def sample_endpoint_corrupt_negative_pairs(
    split_pos: pd.DataFrame,
    mirna_pool: list[str],
    mrna_pool: list[str],
    positive_pairs: set[tuple[str, str]],
    n_samples: int,
    rng: np.random.Generator,
    used_negative_pairs: set[tuple[str, str]],
    split_strategy: str,
) -> list[tuple[str, str]]:
    if not mirna_pool or not mrna_pool:
        raise ValueError("Negative sampling pools must not be empty.")
    if split_pos.empty and n_samples:
        raise ValueError("Cannot endpoint-corrupt without positive edges in the split.")

    pos_records = split_pos[["mirna_id", "mrna_id"]].to_records(index=False)
    sampled: list[tuple[str, str]] = []
    local_seen: set[tuple[str, str]] = set()
    max_attempts = max(10000, n_samples * 200)
    attempts = 0

    while len(sampled) < n_samples and attempts < max_attempts:
        attempts += 1
        pos_mirna, pos_mrna = pos_records[int(rng.integers(0, len(pos_records)))]

        if split_strategy == "cold_mirna":
            corrupt_mrna = True
        elif split_strategy == "cold_mrna":
            corrupt_mrna = False
        else:
            corrupt_mrna = bool(rng.integers(0, 2))

        if corrupt_mrna:
            pair = (str(pos_mirna), mrna_pool[int(rng.integers(0, len(mrna_pool)))])
        else:
            pair = (mirna_pool[int(rng.integers(0, len(mirna_pool)))], str(pos_mrna))

        if pair in positive_pairs or pair in used_negative_pairs or pair in local_seen:
            continue
        sampled.append(pair)
        local_seen.add(pair)

    if len(sampled) < n_samples:
        # Fallback keeps the script robust for dense local neighborhoods.
        sampled.extend(
            sample_uniform_negative_pairs(
                mirna_pool=mirna_pool,
                mrna_pool=mrna_pool,
                positive_pairs=positive_pairs,
                n_samples=n_samples - len(sampled),
                rng=rng,
                used_negative_pairs=used_negative_pairs | local_seen,
            )
        )
        local_seen.update(sampled)

    if len(sampled) != n_samples:
        raise RuntimeError(f"Failed to sample {n_samples} endpoint-corrupt negatives; sampled {len(sampled)}")
    used_negative_pairs.update(sampled)
    return sampled


def build_candidate_edges(
    pos_edges: pd.DataFrame,
    nodes: pd.DataFrame,
    mirna_index: dict[str, int],
    mrna_index: dict[str, int],
    split_strategy: str,
    negative_strategy: str,
    neg_ratio: float,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict[str, object]]:
    positive_pairs = set(zip(pos_edges["mirna_id"], pos_edges["mrna_id"]))
    all_mirnas = sorted(mirna_index)
    all_mrnas = sorted(mrna_index)
    used_negative_pairs: set[tuple[str, str]] = set()
    candidate_parts: list[pd.DataFrame] = []
    neg_report: dict[str, object] = {}

    for split_name in ["train", "val", "test"]:
        split_pos = pos_edges[pos_edges["split"].eq(split_name)].copy()
        n_neg = int(round(len(split_pos) * neg_ratio))

        if split_strategy == "cold_mirna":
            mirna_pool = sorted(split_pos["mirna_id"].unique().tolist())
            mrna_pool = all_mrnas
        elif split_strategy == "cold_mrna":
            mirna_pool = all_mirnas
            mrna_pool = sorted(split_pos["mrna_id"].unique().tolist())
        else:
            mirna_pool = all_mirnas
            mrna_pool = all_mrnas

        if negative_strategy == "endpoint_corrupt":
            neg_pairs = sample_endpoint_corrupt_negative_pairs(
                split_pos=split_pos,
                mirna_pool=mirna_pool,
                mrna_pool=mrna_pool,
                positive_pairs=positive_pairs,
                n_samples=n_neg,
                rng=rng,
                used_negative_pairs=used_negative_pairs,
                split_strategy=split_strategy,
            )
        else:
            neg_pairs = sample_uniform_negative_pairs(
                mirna_pool=mirna_pool,
                mrna_pool=mrna_pool,
                positive_pairs=positive_pairs,
                n_samples=n_neg,
                rng=rng,
                used_negative_pairs=used_negative_pairs,
            )
        split_pos["label"] = 1
        split_pos["edge_source"] = "positive"
        split_neg = pd.DataFrame(neg_pairs, columns=["mirna_id", "mrna_id"])
        split_neg["species"] = split_pos["species"].iloc[0]
        split_neg["dataset_id"] = "negative_sampling"
        split_neg["sample_id"] = [f"{split_name}_neg_{i:07d}" for i in range(len(split_neg))]
        split_neg["split"] = split_name
        split_neg["label"] = 0
        split_neg["edge_source"] = "negative_sampling"

        candidate_parts.extend([split_pos, split_neg])
        neg_report[split_name] = {
            "positive_edges": int(len(split_pos)),
            "negative_edges": int(len(split_neg)),
            "mirna_pool_size": int(len(mirna_pool)),
            "mrna_pool_size": int(len(mrna_pool)),
            "negative_strategy": negative_strategy,
        }

    candidates = pd.concat(candidate_parts, ignore_index=True, sort=False)
    candidates["src_idx"] = candidates["mirna_id"].map(mirna_index).astype("int64")
    candidates["dst_idx"] = candidates["mrna_id"].map(mrna_index).astype("int64")
    return candidates, neg_report


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


def pure_node_report(edges: pd.DataFrame) -> dict[str, object]:
    reports: dict[str, object] = {}
    for split_name in ["overall", "train", "val", "test"]:
        subset = edges if split_name == "overall" else edges[edges["split"].eq(split_name)]
        node_labels: dict[int, set[int]] = {}
        for row in subset[["src_idx", "dst_idx", "label"]].itertuples(index=False):
            label = int(row.label)
            node_labels.setdefault(int(row.src_idx), set()).add(label)
            node_labels.setdefault(int(row.dst_idx), set()).add(label)
        pure_nodes = sum(1 for labels in node_labels.values() if len(labels) == 1)
        total_nodes = len(node_labels)

        pos = subset[subset["label"].eq(1)]
        neg = subset[subset["label"].eq(0)]
        pos_mirnas = set(pos["mirna_id"])
        neg_mirnas = set(neg["mirna_id"])
        pos_mrnas = set(pos["mrna_id"])
        neg_mrnas = set(neg["mrna_id"])
        reports[split_name] = {
            "candidate_edges": int(len(subset)),
            "positive_edges": int(len(pos)),
            "negative_edges": int(len(neg)),
            "nodes_in_candidates": int(total_nodes),
            "pure_nodes": int(pure_nodes),
            "pure_node_ratio": float(pure_nodes / total_nodes) if total_nodes else None,
            "positive_mirna_count": int(len(pos_mirnas)),
            "negative_mirna_count": int(len(neg_mirnas)),
            "positive_negative_mirna_overlap": int(len(pos_mirnas & neg_mirnas)),
            "positive_negative_mirna_overlap_ratio_vs_positive": float(len(pos_mirnas & neg_mirnas) / len(pos_mirnas))
            if pos_mirnas
            else None,
            "positive_mrna_count": int(len(pos_mrnas)),
            "negative_mrna_count": int(len(neg_mrnas)),
            "positive_negative_mrna_overlap": int(len(pos_mrnas & neg_mrnas)),
            "positive_negative_mrna_overlap_ratio_vs_positive": float(len(pos_mrnas & neg_mrnas) / len(pos_mrnas))
            if pos_mrnas
            else None,
        }
    return reports


def degree_leakage_report(pos_edges: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, object]:
    train_pos = pos_edges[pos_edges["split"].eq("train")]
    degree: dict[int, int] = {}
    for row in train_pos[["src_idx", "dst_idx"]].itertuples(index=False):
        degree[int(row.src_idx)] = degree.get(int(row.src_idx), 0) + 1
        degree[int(row.dst_idx)] = degree.get(int(row.dst_idx), 0) + 1

    reports: dict[str, object] = {}
    for split_name in ["train", "val", "test"]:
        subset = candidates[candidates["split"].eq(split_name)]
        scores = np.asarray(
            [degree.get(int(row.src_idx), 0) + degree.get(int(row.dst_idx), 0) for row in subset.itertuples(index=False)],
            dtype=np.float32,
        )
        labels = subset["label"].to_numpy(dtype=np.int64)
        reports[split_name] = {
            "degree_sum_auc": binary_auc(labels, scores),
            "mean_score_positive": float(scores[labels == 1].mean()) if np.any(labels == 1) else None,
            "mean_score_negative": float(scores[labels == 0].mean()) if np.any(labels == 0) else None,
        }
    return reports


def node_sequence_conflicts(edges: pd.DataFrame) -> dict[str, int]:
    mirna_conflicts = (
        edges.groupby("mirna_id")["mirna_sequence"].nunique(dropna=False).gt(1).sum()
    )
    mrna_conflicts = (
        edges.groupby("mrna_id")["mrna_sequence"].nunique(dropna=False).gt(1).sum()
    )
    return {
        "mirna_nodes_with_multiple_sequences": int(mirna_conflicts),
        "mrna_nodes_with_multiple_sequences": int(mrna_conflicts),
        "rows_missing_mirna_sequence": int(edges["mirna_sequence"].eq("").sum()),
        "rows_missing_mrna_sequence": int(edges["mrna_sequence"].eq("").sum()),
    }


def export_edges_csv(edges: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "species",
        "dataset_id",
        "sample_id",
        "split",
        "label",
        "edge_source",
        "mirna_id",
        "mrna_id",
        "src_idx",
        "dst_idx",
    ]
    present = [column for column in columns if column in edges.columns]
    edges[present].to_csv(path, index=False)


def export_graphsage_npz(
    graph_dir: Path,
    nodes: pd.DataFrame,
    pos_edges: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, str]:
    graph_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = graph_dir / "nodes.csv"
    pos_path = graph_dir / "positive_edges.csv"
    all_candidates_path = graph_dir / "candidate_edges.csv"
    nodes.to_csv(nodes_path, index=False)
    export_edges_csv(pos_edges, pos_path)
    export_edges_csv(candidates, all_candidates_path)

    paths = {
        "nodes": str(nodes_path),
        "positive_edges": str(pos_path),
        "candidate_edges": str(all_candidates_path),
    }

    x = nodes[NODE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    node_type = nodes["node_type_id"].to_numpy(dtype=np.int8)
    train_pos = pos_edges[pos_edges["split"].eq("train")]
    edge_index = train_pos[["src_idx", "dst_idx"]].to_numpy(dtype=np.int64).T
    edge_index_undirected = np.concatenate([edge_index, edge_index[::-1]], axis=1)

    arrays: dict[str, np.ndarray] = {
        "x": x,
        "node_type": node_type,
        "edge_index": edge_index,
        "edge_index_undirected": edge_index_undirected,
        "node_feature_names": np.asarray(NODE_FEATURE_COLUMNS, dtype=str),
        "edge_feature_names": np.asarray([], dtype=str),
        "node_ids": nodes["node_id"].astype(str).to_numpy(dtype=str),
    }

    edge_index_path = graph_dir / "edge_index_train_pos.csv"
    train_pos[["src_idx", "dst_idx"]].to_csv(edge_index_path, index=False)
    paths["edge_index_train_pos"] = str(edge_index_path)

    for split_name in ["train", "val", "test"]:
        split_candidates = candidates[candidates["split"].eq(split_name)].copy()
        split_path = graph_dir / f"{split_name}_edges.csv"
        split_pos_path = graph_dir / f"{split_name}_pos_edges.csv"
        split_neg_path = graph_dir / f"{split_name}_neg_edges.csv"
        export_edges_csv(split_candidates, split_path)
        export_edges_csv(split_candidates[split_candidates["label"].eq(1)], split_pos_path)
        export_edges_csv(split_candidates[split_candidates["label"].eq(0)], split_neg_path)
        paths[f"{split_name}_edges"] = str(split_path)
        paths[f"{split_name}_pos_edges"] = str(split_pos_path)
        paths[f"{split_name}_neg_edges"] = str(split_neg_path)

        arrays[f"{split_name}_edge_label_index"] = split_candidates[["src_idx", "dst_idx"]].to_numpy(
            dtype=np.int64
        ).T
        arrays[f"{split_name}_edge_label"] = split_candidates["label"].to_numpy(dtype=np.int64)
        arrays[f"{split_name}_edge_attr"] = np.zeros((len(split_candidates), 0), dtype=np.float32)
        arrays[f"{split_name}_sample_id"] = split_candidates["sample_id"].astype(str).to_numpy(dtype=str)

    npz_path = graph_dir / "graphsage_inputs.npz"
    np.savez_compressed(npz_path, **arrays)
    paths["graphsage_inputs_npz"] = str(npz_path)
    return paths


def process_species(
    species: str,
    files: list[RawPosFile],
    args: argparse.Namespace,
) -> dict[str, object]:
    rng = np.random.default_rng(args.seed + sum(ord(ch) for ch in species))
    graph_dir = args.output_dir / species

    raw_frames = [read_positive_file(raw_file) for raw_file in sorted(files, key=lambda item: item.path.name)]
    raw_edges = pd.concat(raw_frames, ignore_index=True, sort=False)
    raw_count = int(len(raw_edges))
    seq_conflicts_before = node_sequence_conflicts(raw_edges)

    pos_edges = raw_edges.sort_values(["dataset_id", "sample_id"]).drop_duplicates(
        ["mirna_id", "mrna_id"], keep="first"
    )
    pos_edges = pos_edges.reset_index(drop=True)

    if args.split_strategy == "random":
        pos_edges["split"] = split_random_edges(
            pos_edges, rng, args.train_ratio, args.val_ratio, args.test_ratio
        )
    elif args.split_strategy == "cold_mirna":
        pos_edges["split"] = split_group_cold(
            pos_edges, "mirna_id", rng, args.train_ratio, args.val_ratio, args.test_ratio
        )
    else:
        pos_edges["split"] = split_group_cold(
            pos_edges, "mrna_id", rng, args.train_ratio, args.val_ratio, args.test_ratio
        )

    nodes, mirna_index, mrna_index = build_nodes(pos_edges)
    pos_edges["src_idx"] = pos_edges["mirna_id"].map(mirna_index).astype("int64")
    pos_edges["dst_idx"] = pos_edges["mrna_id"].map(mrna_index).astype("int64")
    pos_edges["edge_source"] = "positive"

    candidates, neg_report = build_candidate_edges(
        pos_edges=pos_edges,
        nodes=nodes,
        mirna_index=mirna_index,
        mrna_index=mrna_index,
        split_strategy=args.split_strategy,
        negative_strategy=args.negative_strategy,
        neg_ratio=args.neg_ratio,
        rng=rng,
    )
    paths = export_graphsage_npz(graph_dir, nodes, pos_edges, candidates)

    split_counts = {}
    for split_name in ["train", "val", "test"]:
        split_pos = pos_edges[pos_edges["split"].eq(split_name)]
        split_candidates = candidates[candidates["split"].eq(split_name)]
        split_counts[split_name] = {
            "positive_edges": int(len(split_pos)),
            "negative_edges": int((split_candidates["label"] == 0).sum()),
            "candidate_edges": int(len(split_candidates)),
            "unique_mirnas_positive": int(split_pos["mirna_id"].nunique()),
            "unique_mrnas_positive": int(split_pos["mrna_id"].nunique()),
        }

    leakage_report = {
        "species": species,
        "split_strategy": args.split_strategy,
        "node_overlap_and_purity": pure_node_report(candidates),
        "degree_leakage": degree_leakage_report(pos_edges, candidates),
    }
    leakage_path = graph_dir / "leakage_report.json"
    leakage_path.write_text(json.dumps(leakage_report, indent=2), encoding="utf-8")
    paths["leakage_report"] = str(leakage_path)

    metadata = {
        "species": species,
        "split_strategy": args.split_strategy,
        "negative_strategy": args.negative_strategy,
        "raw_positive_rows": raw_count,
        "deduplicated_positive_edges": int(len(pos_edges)),
        "duplicate_positive_rows_removed": int(raw_count - len(pos_edges)),
        "num_nodes": int(len(nodes)),
        "num_mirna_nodes": int(len(mirna_index)),
        "num_mrna_nodes": int(len(mrna_index)),
        "num_node_features": int(len(NODE_FEATURE_COLUMNS)),
        "num_edge_features": 0,
        "node_feature_columns": NODE_FEATURE_COLUMNS,
        "sequence_conflicts_before_dedup": seq_conflicts_before,
        "sequence_conflicts_after_dedup": node_sequence_conflicts(pos_edges),
        "split_counts": split_counts,
        "negative_sampling": neg_report,
        "paths": paths,
    }
    metadata_path = graph_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["paths"]["metadata"] = str(metadata_path)
    return metadata


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
        category=FutureWarning,
    )
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_pos_files(args.input_dir)
    by_species: dict[str, list[RawPosFile]] = {}
    for raw_file in files:
        by_species.setdefault(raw_file.species, []).append(raw_file)

    summary: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "split_strategy": args.split_strategy,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "neg_ratio": args.neg_ratio,
        "negative_strategy": args.negative_strategy,
        "note": "Only *_pos.csv files are used. Negative edges are freshly sampled from true miRNA/mRNA nodes.",
        "species": {},
    }

    ordered_species = [species for species in SPECIES_ORDER if species in by_species]
    ordered_species.extend(sorted(set(by_species) - set(ordered_species)))
    for species in ordered_species:
        print(f"[{species}] processing {len(by_species[species])} positive files")
        metadata = process_species(species, by_species[species], args)
        summary["species"][species] = metadata
        split_counts = metadata["split_counts"]
        print(
            f"[{species}] pos={metadata['deduplicated_positive_edges']} "
            f"nodes={metadata['num_nodes']} "
            f"train={split_counts['train']['candidate_edges']} "
            f"val={split_counts['val']['candidate_edges']} "
            f"test={split_counts['test']['candidate_edges']}"
        )

    summary_path = args.output_dir / "preprocess_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
