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


FILE_RE = re.compile(r"^(?P<dataset_id>(?P<species>[A-Za-z]+)\d+)_(?P<label>pos|neg)\.csv$")
SPECIES_ORDER = ["human", "mouse", "worm", "cow"]

INPUT_DROP_COLUMNS = {
    "Unnamed: 0",
    "Source",
    "Organism",
    "GI_ID",
    "number of reads",
    "full_mrna",
}

ID_AND_NODE_SOURCE_COLUMNS = {
    "microRNA_name",
    "miRNA sequence",
    "target sequence",
    "mRNA_name",
    "mRNA_start",
    "mRNA_end",
}

INTERNAL_COLUMNS = {
    "_mirna_seq_norm",
    "_target_seq_norm",
    "_mRNA_start_str",
    "_mRNA_end_str",
}

BASES = ["A", "C", "G", "U"]
DINUCS = [a + b for a in BASES for b in BASES]
NODE_FEATURE_COLUMNS = (
    [f"seq_1mer_{base}" for base in BASES]
    + [f"seq_2mer_{dinuc}" for dinuc in DINUCS]
    + ["seq_length", "seq_gc_content"]
)


@dataclass(frozen=True)
class RawFile:
    path: Path
    species: str
    dataset_id: str
    label_name: str
    label: int
    columns: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build species-level miRNA-target_site graph preprocessing outputs for GraphSAGE."
    )
    parser.add_argument("--input-dir", default="data/external", type=Path)
    parser.add_argument("--output-dir", default="data/processed", type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--train-ratio", default=0.72, type=float)
    parser.add_argument("--val-ratio", default=0.08, type=float)
    parser.add_argument("--test-ratio", default=0.20, type=float)
    parser.add_argument(
        "--conflict-policy",
        default="prefer_positive",
        choices=["prefer_positive", "drop_all"],
        help="How to handle the same miRNA-site pair appearing with both labels.",
    )
    parser.add_argument(
        "--no-balance-labels",
        action="store_true",
        help="Keep all post-dedup rows instead of downsampling to equal positive/negative counts.",
    )
    parser.add_argument(
        "--no-dataset-edges",
        action="store_true",
        help="Skip writing per-dataset merged edge CSV files.",
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


def format_coord(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "NA"
        if float(value).is_integer():
            return str(int(value))
        return f"{float(value):g}"
    text = str(value).strip()
    if not text:
        return "NA"
    try:
        number = float(text)
    except ValueError:
        return text
    if math.isfinite(number) and number.is_integer():
        return str(int(number))
    if math.isfinite(number):
        return f"{number:g}"
    return "NA"


def discover_files(input_dir: Path) -> list[RawFile]:
    files: list[RawFile] = []
    for path in sorted(input_dir.glob("*.csv")):
        match = FILE_RE.match(path.name)
        if not match:
            continue
        label_name = match.group("label")
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        files.append(
            RawFile(
                path=path,
                species=match.group("species").lower(),
                dataset_id=match.group("dataset_id").lower(),
                label_name=label_name,
                label=1 if label_name == "pos" else 0,
                columns=columns,
            )
        )
    if not files:
        raise FileNotFoundError(f"No input CSV files matching xxx_pos/neg.csv found in {input_dir}")
    return files


def ordered_union_columns(files: Iterable[RawFile]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    ordered_files = sorted(
        files,
        key=lambda raw_file: (
            0 if raw_file.label_name == "pos" else 1,
            -len(raw_file.columns),
            raw_file.path.name,
        ),
    )
    for raw_file in ordered_files:
        for column in raw_file.columns:
            if column in INPUT_DROP_COLUMNS or column in seen:
                continue
            seen.add(column)
            ordered.append(column)
    return ordered


def edge_feature_columns(input_columns: Iterable[str]) -> list[str]:
    return [column for column in input_columns if column not in ID_AND_NODE_SOURCE_COLUMNS]


def add_missing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = pd.NA
    return df[columns]


def read_standard_file(raw_file: RawFile, input_columns: list[str]) -> pd.DataFrame:
    usecols = [column for column in input_columns if column in raw_file.columns]
    df = pd.read_csv(raw_file.path, usecols=usecols, low_memory=False)
    df = add_missing_columns(df, input_columns)

    row_numbers = np.arange(len(df))
    df["species"] = raw_file.species
    df["dataset_id"] = raw_file.dataset_id
    df["label"] = raw_file.label
    df["sample_id"] = [f"{raw_file.dataset_id}_{raw_file.label_name}_{i:07d}" for i in row_numbers]

    mirna_name = df["microRNA_name"].map(clean_text)
    mrna_name = df["mRNA_name"].map(clean_text)
    mirna_seq = df["miRNA sequence"].map(normalize_sequence)
    target_seq = df["target sequence"].map(normalize_sequence)
    mirna_hash = mirna_seq.map(short_hash)
    target_hash = target_seq.map(short_hash)
    start_str = df["mRNA_start"].map(format_coord)
    end_str = df["mRNA_end"].map(format_coord)

    df["_mirna_seq_norm"] = mirna_seq
    df["_target_seq_norm"] = target_seq
    df["_mRNA_start_str"] = start_str
    df["_mRNA_end_str"] = end_str
    df["mirna_sequence_hash"] = mirna_hash
    df["target_sequence_hash"] = target_hash
    df["mirna_id"] = raw_file.species + "|" + mirna_name + "|" + mirna_hash
    df["site_id"] = raw_file.species + "|" + mrna_name + "|" + start_str + "|" + end_str + "|" + target_hash
    return df


def export_edge_csv(
    edges: pd.DataFrame,
    path: Path,
    feature_columns: list[str],
    feature_values: pd.DataFrame | None = None,
    include_split: bool = False,
    include_indices: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata_columns = [
        "species",
        "dataset_id",
        "sample_id",
        "label",
        "mirna_id",
        "site_id",
        "microRNA_name",
        "mRNA_name",
        "_mRNA_start_str",
        "_mRNA_end_str",
        "mirna_sequence_hash",
        "target_sequence_hash",
    ]
    if include_split:
        metadata_columns.insert(4, "split")
    if include_indices:
        metadata_columns.extend(["src_idx", "dst_idx"])

    present_metadata = [column for column in metadata_columns if column in edges.columns]
    exported = edges[present_metadata].rename(
        columns={"_mRNA_start_str": "mRNA_start", "_mRNA_end_str": "mRNA_end"}
    )
    if feature_values is None:
        features = edges[[column for column in feature_columns if column in edges.columns]].copy()
    else:
        features = feature_values[[column for column in feature_columns if column in feature_values.columns]].copy()
    exported = pd.concat([exported.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    exported.to_csv(path, index=False)


def pair_key(edges: pd.DataFrame) -> pd.Series:
    return edges["mirna_id"].astype(str) + "\t" + edges["site_id"].astype(str)


def resolve_label_conflicts(
    edges: pd.DataFrame, conflict_policy: str
) -> tuple[pd.DataFrame, dict[str, int]]:
    before_rows = len(edges)
    keys = pair_key(edges)
    label_counts = edges.assign(_pair_key=keys).groupby("_pair_key")["label"].agg(["nunique", "size"])
    conflict_keys = set(label_counts.index[label_counts["nunique"] > 1])
    duplicate_pair_rows = int(label_counts["size"].sub(1).clip(lower=0).sum())

    if conflict_policy == "drop_all":
        keep_mask = ~keys.isin(conflict_keys)
    else:
        keep_mask = ~(keys.isin(conflict_keys) & edges["label"].eq(0))
    conflict_removed = int((~keep_mask).sum())
    edges = edges.loc[keep_mask].copy()

    before_dedup = len(edges)
    edges = edges.sort_values(["label", "dataset_id", "sample_id"], ascending=[False, True, True])
    edges = edges.drop_duplicates(["mirna_id", "site_id"], keep="first").copy()
    duplicate_removed = before_dedup - len(edges)

    return edges.reset_index(drop=True), {
        "rows_before_conflict_resolution": int(before_rows),
        "conflicting_pairs": int(len(conflict_keys)),
        "rows_removed_by_conflict_policy": int(conflict_removed),
        "duplicate_pair_extra_rows_before_resolution": int(duplicate_pair_rows),
        "duplicate_rows_removed_after_conflict_resolution": int(duplicate_removed),
        "rows_after_conflict_and_dedup": int(len(edges)),
    }


def balance_labels(edges: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, dict[str, int]]:
    counts = edges["label"].value_counts().to_dict()
    pos_count = int(counts.get(1, 0))
    neg_count = int(counts.get(0, 0))
    if pos_count == 0 or neg_count == 0:
        raise ValueError("Both positive and negative labels are required after conflict resolution.")

    target = min(pos_count, neg_count)
    keep_indices: list[int] = []
    dropped = {0: 0, 1: 0}
    for label in [1, 0]:
        label_indices = edges.index[edges["label"].eq(label)].to_numpy()
        if len(label_indices) > target:
            chosen = rng.choice(label_indices, size=target, replace=False)
            dropped[label] = int(len(label_indices) - target)
        else:
            chosen = label_indices
        keep_indices.extend(chosen.tolist())

    balanced = edges.loc[sorted(keep_indices)].reset_index(drop=True)
    return balanced, {
        "positive_before_balance": pos_count,
        "negative_before_balance": neg_count,
        "positive_after_balance": int((balanced["label"] == 1).sum()),
        "negative_after_balance": int((balanced["label"] == 0).sum()),
        "positive_rows_dropped_for_balance": dropped[1],
        "negative_rows_dropped_for_balance": dropped[0],
    }


def stratified_split(
    edges: pd.DataFrame,
    rng: np.random.Generator,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    split = pd.Series(index=edges.index, dtype="object")
    split_report: dict[str, dict[str, int]] = {}
    for label in [1, 0]:
        label_indices = edges.index[edges["label"].eq(label)].to_numpy()
        rng.shuffle(label_indices)
        n_total = len(label_indices)
        n_val = int(round(n_total * val_ratio))
        n_test = int(round(n_total * test_ratio))
        n_train = n_total - n_val - n_test
        if n_train <= 0:
            raise ValueError(f"Not enough rows for label {label} to create train/val/test split.")

        train_idx = label_indices[:n_train]
        val_idx = label_indices[n_train : n_train + n_val]
        test_idx = label_indices[n_train + n_val :]
        split.loc[train_idx] = "train"
        split.loc[val_idx] = "val"
        split.loc[test_idx] = "test"
        split_report[str(label)] = {
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx)),
            "total": int(n_total),
        }

    edges = edges.copy()
    edges["split"] = split
    return edges, split_report


def count_node_sequence_conflicts(edges: pd.DataFrame) -> dict[str, int]:
    mirna_node_conflicts = (
        edges.groupby("mirna_id")["_mirna_seq_norm"].nunique(dropna=False).gt(1).sum()
    )
    site_node_conflicts = (
        edges.groupby("site_id")["_target_seq_norm"].nunique(dropna=False).gt(1).sum()
    )
    mirna_name_variants = (
        edges.groupby(["species", "microRNA_name"])["mirna_sequence_hash"].nunique(dropna=False).gt(1).sum()
    )
    site_coordinate_variants = (
        edges.groupby(["species", "mRNA_name", "_mRNA_start_str", "_mRNA_end_str"])["target_sequence_hash"]
        .nunique(dropna=False)
        .gt(1)
        .sum()
    )
    return {
        "mirna_node_ids_with_multiple_sequences": int(mirna_node_conflicts),
        "site_node_ids_with_multiple_sequences": int(site_node_conflicts),
        "mirna_names_with_multiple_sequence_hashes": int(mirna_name_variants),
        "site_coordinate_keys_with_multiple_target_hashes": int(site_coordinate_variants),
        "rows_missing_mirna_sequence": int(edges["_mirna_seq_norm"].eq("").sum()),
        "rows_missing_target_sequence": int(edges["_target_seq_norm"].eq("").sum()),
    }


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


def build_nodes(edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    mirna_nodes = (
        edges[
            [
                "mirna_id",
                "species",
                "microRNA_name",
                "_mirna_seq_norm",
                "mirna_sequence_hash",
            ]
        ]
        .drop_duplicates("mirna_id", keep="first")
        .rename(
            columns={
                "mirna_id": "node_id",
                "microRNA_name": "source_name",
                "_mirna_seq_norm": "sequence",
                "mirna_sequence_hash": "sequence_hash",
            }
        )
    )
    mirna_nodes["node_type"] = "mirna"
    mirna_nodes["node_type_id"] = 0
    mirna_nodes["mRNA_start"] = pd.NA
    mirna_nodes["mRNA_end"] = pd.NA

    site_nodes = (
        edges[
            [
                "site_id",
                "species",
                "mRNA_name",
                "_mRNA_start_str",
                "_mRNA_end_str",
                "_target_seq_norm",
                "target_sequence_hash",
            ]
        ]
        .drop_duplicates("site_id", keep="first")
        .rename(
            columns={
                "site_id": "node_id",
                "mRNA_name": "source_name",
                "_mRNA_start_str": "mRNA_start",
                "_mRNA_end_str": "mRNA_end",
                "_target_seq_norm": "sequence",
                "target_sequence_hash": "sequence_hash",
            }
        )
    )
    site_nodes["node_type"] = "target_site"
    site_nodes["node_type_id"] = 1

    nodes = pd.concat([mirna_nodes, site_nodes], ignore_index=True, sort=False)
    nodes.insert(0, "node_idx", np.arange(len(nodes), dtype=np.int64))

    node_features = sequence_feature_matrix(nodes["sequence"].fillna("").astype(str))
    for idx, feature_name in enumerate(NODE_FEATURE_COLUMNS):
        nodes[feature_name] = node_features[:, idx]

    node_index = dict(zip(nodes["node_id"], nodes["node_idx"]))
    return nodes, node_index


def is_bool_like(series: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(series):
        return True
    non_null = series.dropna()
    if non_null.empty or pd.api.types.is_numeric_dtype(non_null):
        return False
    values = {str(value).strip().lower() for value in non_null.unique()}
    return values.issubset({"true", "false", "yes", "no"})


def coerce_edge_features(
    edges: pd.DataFrame, feature_columns: list[str]
) -> tuple[pd.DataFrame, list[str], list[str]]:
    value_columns: dict[str, pd.Series | float] = {}
    bool_columns: list[str] = []
    all_nan_columns: list[str] = []
    bool_map = {"true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0}

    for column in feature_columns:
        if column not in edges.columns:
            value_columns[column] = np.nan
            all_nan_columns.append(column)
            continue
        series = edges[column]
        if is_bool_like(series):
            bool_columns.append(column)
            if pd.api.types.is_bool_dtype(series):
                coerced = series.astype("float32")
            else:
                coerced = series.map(lambda value: bool_map.get(str(value).strip().lower(), np.nan))
        else:
            coerced = pd.to_numeric(series, errors="coerce")
        value_columns[column] = coerced
        if pd.Series(coerced, index=edges.index).isna().all():
            all_nan_columns.append(column)

    values = pd.DataFrame(value_columns, index=edges.index)
    return values, bool_columns, all_nan_columns


def fit_transform_edge_features(
    raw_features: pd.DataFrame, split: pd.Series, bool_columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_mask = split.eq("train")
    transformed_columns: dict[str, pd.Series] = {}
    stats_rows: list[dict[str, object]] = []
    bool_set = set(bool_columns)

    for column in raw_features.columns:
        train_values = raw_features.loc[train_mask, column]
        if train_values.notna().any():
            median = train_values.median(skipna=True)
        else:
            median = 0.0
        if pd.isna(median) or not math.isfinite(float(median)):
            median = 0.0
        filled = raw_features[column].fillna(float(median)).astype("float32")

        if column in bool_set:
            transformed_columns[column] = filled
            mean = 0.0
            std = 1.0
            standardized = False
        else:
            train_filled = train_values.fillna(float(median)).astype("float32")
            mean = float(train_filled.mean(skipna=True))
            std = float(train_filled.std(skipna=True, ddof=0))
            if not math.isfinite(mean):
                mean = 0.0
            if not math.isfinite(std) or std == 0.0:
                std = 1.0
            transformed_columns[column] = ((filled - mean) / std).astype("float32")
            standardized = True

        stats_rows.append(
            {
                "feature": column,
                "is_bool": column in bool_set,
                "fill_median_from_train": float(median),
                "mean_from_train_after_fill": mean,
                "std_from_train_after_fill": std,
                "standardized": standardized,
            }
        )

    transformed = pd.DataFrame(transformed_columns, index=raw_features.index)
    return transformed, pd.DataFrame(stats_rows)


def add_node_indices(edges: pd.DataFrame, node_index: dict[str, int]) -> pd.DataFrame:
    edges = edges.copy()
    edges["src_idx"] = edges["mirna_id"].map(node_index).astype("int64")
    edges["dst_idx"] = edges["site_id"].map(node_index).astype("int64")
    return edges


def split_counts(edges: pd.DataFrame) -> dict[str, dict[str, int]]:
    report: dict[str, dict[str, int]] = {}
    for split_name in ["train", "val", "test"]:
        subset = edges[edges["split"].eq(split_name)]
        report[split_name] = {
            "positive": int((subset["label"] == 1).sum()),
            "negative": int((subset["label"] == 0).sum()),
            "total": int(len(subset)),
        }
    return report


def export_graphsage_files(
    graph_dir: Path,
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    raw_features: pd.DataFrame,
    standardized_features: pd.DataFrame,
    feature_columns: list[str],
) -> dict[str, str]:
    graph_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    nodes_path = graph_dir / "nodes.csv"
    nodes.to_csv(nodes_path, index=False)
    paths["nodes"] = str(nodes_path)

    for split_name in ["train", "val", "test"]:
        split_mask = edges["split"].eq(split_name)
        split_path = graph_dir / f"{split_name}_edges.csv"
        export_edge_csv(
            edges.loc[split_mask],
            split_path,
            feature_columns,
            feature_values=standardized_features.loc[split_mask],
            include_split=True,
            include_indices=True,
        )
        paths[f"{split_name}_edges"] = str(split_path)

        for label_name, label_value in [("pos", 1), ("neg", 0)]:
            mask = split_mask & edges["label"].eq(label_value)
            label_path = graph_dir / f"{split_name}_{label_name}_edges.csv"
            export_edge_csv(
                edges.loc[mask],
                label_path,
                feature_columns,
                feature_values=standardized_features.loc[mask],
                include_split=True,
                include_indices=True,
            )
            paths[f"{split_name}_{label_name}_edges"] = str(label_path)

    train_pos_mask = edges["split"].eq("train") & edges["label"].eq(1)
    edge_index_path = graph_dir / "edge_index_train_pos.csv"
    edges.loc[train_pos_mask, ["src_idx", "dst_idx"]].to_csv(edge_index_path, index=False)
    paths["edge_index_train_pos"] = str(edge_index_path)

    raw_feature_path = graph_dir / "edge_features_raw.csv"
    pd.concat(
        [
            edges[["sample_id", "split", "label", "src_idx", "dst_idx"]].reset_index(drop=True),
            raw_features.reset_index(drop=True),
        ],
        axis=1,
    ).to_csv(raw_feature_path, index=False)
    paths["edge_features_raw"] = str(raw_feature_path)

    std_feature_path = graph_dir / "edge_features_standardized.csv"
    pd.concat(
        [
            edges[["sample_id", "split", "label", "src_idx", "dst_idx"]].reset_index(drop=True),
            standardized_features.reset_index(drop=True),
        ],
        axis=1,
    ).to_csv(std_feature_path, index=False)
    paths["edge_features_standardized"] = str(std_feature_path)

    x = nodes[NODE_FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    node_type = nodes["node_type_id"].to_numpy(dtype=np.int8)
    edge_index = edges.loc[train_pos_mask, ["src_idx", "dst_idx"]].to_numpy(dtype=np.int64).T
    edge_index_undirected = np.concatenate([edge_index, edge_index[::-1]], axis=1)

    arrays: dict[str, np.ndarray] = {
        "x": x,
        "node_type": node_type,
        "edge_index": edge_index,
        "edge_index_undirected": edge_index_undirected,
        "node_feature_names": np.asarray(NODE_FEATURE_COLUMNS, dtype=str),
        "edge_feature_names": np.asarray(feature_columns, dtype=str),
        "node_ids": nodes["node_id"].astype(str).to_numpy(dtype=str),
    }
    for split_name in ["train", "val", "test"]:
        mask = edges["split"].eq(split_name)
        arrays[f"{split_name}_edge_label_index"] = edges.loc[mask, ["src_idx", "dst_idx"]].to_numpy(
            dtype=np.int64
        ).T
        arrays[f"{split_name}_edge_label"] = edges.loc[mask, "label"].to_numpy(dtype=np.int64)
        arrays[f"{split_name}_edge_attr"] = standardized_features.loc[mask].to_numpy(dtype=np.float32)
        arrays[f"{split_name}_sample_id"] = edges.loc[mask, "sample_id"].astype(str).to_numpy(dtype=str)

    arrays_path = graph_dir / "graphsage_inputs.npz"
    np.savez_compressed(arrays_path, **arrays)
    paths["graphsage_inputs_npz"] = str(arrays_path)
    return paths


def process_species(
    species: str,
    species_files: list[RawFile],
    input_columns: list[str],
    feature_columns: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, object]:
    rng = np.random.default_rng(args.seed + sum(ord(ch) for ch in species))
    dataset_edge_dir = output_dir / "dataset_edges"
    graph_dir = output_dir / "graphsage" / species

    dataset_reports: list[dict[str, object]] = []
    species_parts: list[pd.DataFrame] = []
    files_by_dataset: dict[str, list[RawFile]] = {}
    for raw_file in species_files:
        files_by_dataset.setdefault(raw_file.dataset_id, []).append(raw_file)

    for dataset_id in sorted(files_by_dataset):
        raw_files = sorted(files_by_dataset[dataset_id], key=lambda item: item.label, reverse=True)
        labels_present = {item.label_name for item in raw_files}
        if labels_present != {"pos", "neg"}:
            raise ValueError(f"{dataset_id} must have both pos and neg files, got {labels_present}")

        frames = [read_standard_file(raw_file, input_columns) for raw_file in raw_files]
        dataset_edges = pd.concat(frames, ignore_index=True, sort=False)
        species_parts.append(dataset_edges)
        dataset_reports.append(
            {
                "dataset_id": dataset_id,
                "rows": int(len(dataset_edges)),
                "positive": int((dataset_edges["label"] == 1).sum()),
                "negative": int((dataset_edges["label"] == 0).sum()),
            }
        )
        if not args.no_dataset_edges:
            export_edge_csv(
                dataset_edges,
                dataset_edge_dir / f"{dataset_id}_edges.csv",
                feature_columns,
                include_split=False,
                include_indices=False,
            )

    edges = pd.concat(species_parts, ignore_index=True, sort=False)
    rows_before_resolution = int(len(edges))
    node_conflicts_before = count_node_sequence_conflicts(edges)
    edges, conflict_report = resolve_label_conflicts(edges, args.conflict_policy)

    if args.no_balance_labels:
        balance_report = {
            "positive_before_balance": int((edges["label"] == 1).sum()),
            "negative_before_balance": int((edges["label"] == 0).sum()),
            "positive_after_balance": int((edges["label"] == 1).sum()),
            "negative_after_balance": int((edges["label"] == 0).sum()),
            "positive_rows_dropped_for_balance": 0,
            "negative_rows_dropped_for_balance": 0,
        }
    else:
        edges, balance_report = balance_labels(edges, rng)

    edges, stratified_report = stratified_split(
        edges,
        rng,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    nodes, node_index = build_nodes(edges)
    edges = add_node_indices(edges, node_index)
    raw_features, bool_columns, all_nan_columns = coerce_edge_features(edges, feature_columns)
    standardized_features, stats = fit_transform_edge_features(raw_features, edges["split"], bool_columns)

    species_edges_path = output_dir / f"{species}_edges.csv"
    export_edge_csv(
        edges,
        species_edges_path,
        feature_columns,
        feature_values=raw_features,
        include_split=True,
        include_indices=True,
    )

    stats_path = graph_dir / "edge_feature_stats.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(stats_path, index=False)
    graph_paths = export_graphsage_files(
        graph_dir=graph_dir,
        edges=edges,
        nodes=nodes,
        raw_features=raw_features,
        standardized_features=standardized_features,
        feature_columns=feature_columns,
    )
    graph_paths["species_edges"] = str(species_edges_path)
    graph_paths["edge_feature_stats"] = str(stats_path)

    metadata = {
        "species": species,
        "datasets": dataset_reports,
        "rows_before_resolution": rows_before_resolution,
        "rows_after_all_filters": int(len(edges)),
        "conflict_resolution": conflict_report,
        "label_balance": balance_report,
        "stratified_split_by_label": stratified_report,
        "split_counts": split_counts(edges),
        "node_checks_before_resolution": node_conflicts_before,
        "node_checks_after_filters": count_node_sequence_conflicts(edges),
        "num_nodes": int(len(nodes)),
        "num_mirna_nodes": int((nodes["node_type"] == "mirna").sum()),
        "num_target_site_nodes": int((nodes["node_type"] == "target_site").sum()),
        "num_edge_features": int(len(feature_columns)),
        "num_bool_edge_features": int(len(bool_columns)),
        "bool_edge_features": bool_columns,
        "all_nan_edge_features_after_coercion": all_nan_columns,
        "num_node_features": int(len(NODE_FEATURE_COLUMNS)),
        "node_feature_columns": NODE_FEATURE_COLUMNS,
        "edge_feature_columns": feature_columns,
        "paths": graph_paths,
    }
    metadata_path = graph_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    metadata["paths"]["metadata"] = str(metadata_path)
    return metadata


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
        category=FutureWarning,
    )
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(args.input_dir)
    input_columns = ordered_union_columns(files)
    feature_columns = edge_feature_columns(input_columns)

    files_by_species: dict[str, list[RawFile]] = {}
    for raw_file in files:
        files_by_species.setdefault(raw_file.species, []).append(raw_file)

    summary: dict[str, object] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "conflict_policy": args.conflict_policy,
        "balanced_labels": not args.no_balance_labels,
        "input_file_count": len(files),
        "input_columns_after_dropping_non_model_columns": input_columns,
        "edge_feature_columns": feature_columns,
        "species": {},
    }

    ordered_species = [species for species in SPECIES_ORDER if species in files_by_species]
    ordered_species.extend(sorted(set(files_by_species) - set(ordered_species)))
    for species in ordered_species:
        print(f"[{species}] processing {len(files_by_species[species])} files")
        metadata = process_species(
            species=species,
            species_files=files_by_species[species],
            input_columns=input_columns,
            feature_columns=feature_columns,
            output_dir=output_dir,
            args=args,
        )
        summary["species"][species] = metadata
        split_report = metadata["split_counts"]
        print(
            f"[{species}] rows={metadata['rows_after_all_filters']} "
            f"nodes={metadata['num_nodes']} "
            f"train={split_report['train']['total']} "
            f"val={split_report['val']['total']} "
            f"test={split_report['test']['total']}"
        )

    summary_path = output_dir / "preprocess_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
