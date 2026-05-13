#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FILE_RE = re.compile(r"^(?P<dataset_id>(?P<species>[A-Za-z]+)\d+)_pos\.csv$")
SPECIES_ORDER = ["human", "cow", "mouse", "worm"]
BASES = ["A", "C", "G", "U"]
BOOL_TO_INT = {True: 1, False: 0, "True": 1, "False": 0}

REQUIRED_COLUMNS = [
    "microRNA_name",
    "miRNA sequence",
    "mRNA_name",
    "target sequence",
    "full_mrna",
]

METADATA_COLUMNS = {
    "Source",
    "Organism",
    "GI_ID",
    "microRNA_name",
    "miRNA sequence",
    "target sequence",
    "number of reads",
    "mRNA_name",
    "full_mrna",
}


@dataclass(frozen=True)
class RawPosFile:
    path: Path
    species: str
    dataset_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build GraphMMI leakage-safe graph inputs from positive miRNA-mRNA "
            "interactions only. Negative edges are intentionally not materialized."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/external")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/graph/random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.72)
    parser.add_argument("--val-ratio", type=float, default=0.08)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--kmer-sizes", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--drop-hot-pairing",
        action="store_true",
        default=True,
        help="Drop HotPairing* columns from edge_attr. Enabled by default.",
    )
    parser.add_argument(
        "--keep-hot-pairing",
        action="store_true",
        help="Keep HotPairing* columns in edge_attr.",
    )
    return parser.parse_args()


def json_safe(value):
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


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text


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
            )
        )
    if not files:
        raise FileNotFoundError(f"No *_pos.csv files found in {input_dir}")
    return files


def read_positive_file(raw_file: RawPosFile) -> pd.DataFrame:
    df = pd.read_csv(raw_file.path, index_col=0, low_memory=False)
    df = df.reset_index(drop=False).rename(columns={"index": "raw_index"})
    df["species"] = raw_file.species
    df["dataset_id"] = raw_file.dataset_id
    df["source_file"] = raw_file.path.name
    df["source_row"] = np.arange(len(df), dtype=np.int64)

    mirna_name = df["microRNA_name"].map(clean_text)
    mrna_name = df["mRNA_name"].map(clean_text)
    df["mirna_seq"] = df["miRNA sequence"].map(normalize_sequence)
    df["target_seq"] = df["target sequence"].map(normalize_sequence)
    df["full_mrna_seq"] = df["full_mrna"].map(normalize_sequence)
    df["mrna_seq"] = np.where(df["full_mrna_seq"].astype(str).str.len() > 0, df["full_mrna_seq"], df["target_seq"])
    df["mirna_id"] = raw_file.species + "|" + mirna_name
    df["mrna_id"] = raw_file.species + "|" + mrna_name
    df["label"] = 1
    return df


def sequence_conflict_ids(frame: pd.DataFrame, id_col: str, seq_col: str) -> set[str]:
    counts = frame.groupby(id_col)[seq_col].nunique(dropna=False)
    return set(counts[counts.gt(1)].index.astype(str))


def clean_positive_edges(raw_edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    report: dict[str, int] = {
        "raw_rows": int(len(raw_edges)),
        "dropped_missing_core_fields": 0,
        "dropped_pair_sequence_conflicts": 0,
        "dropped_node_sequence_conflicts": 0,
        "duplicate_pair_rows_removed": 0,
    }

    core_mask = (
        raw_edges["mirna_id"].astype(str).ne("")
        & raw_edges["mrna_id"].astype(str).ne("")
        & raw_edges["mirna_seq"].astype(str).ne("")
        & raw_edges["mrna_seq"].astype(str).ne("")
    )
    report["dropped_missing_core_fields"] = int((~core_mask).sum())
    edges = raw_edges[core_mask].copy()

    pair_group = edges.groupby(["mirna_id", "mrna_id"], dropna=False)
    pair_conflicts = pair_group.filter(
        lambda group: group["mirna_seq"].nunique(dropna=False) > 1
        or group["mrna_seq"].nunique(dropna=False) > 1
    )
    if not pair_conflicts.empty:
        conflict_keys = set(zip(pair_conflicts["mirna_id"], pair_conflicts["mrna_id"]))
        conflict_mask = edges[["mirna_id", "mrna_id"]].apply(tuple, axis=1).isin(conflict_keys)
        report["dropped_pair_sequence_conflicts"] = int(conflict_mask.sum())
        edges = edges[~conflict_mask].copy()

    before_dedup = len(edges)
    edges = (
        edges.sort_values(["dataset_id", "source_row"])
        .drop_duplicates(["mirna_id", "mrna_id"], keep="first")
        .reset_index(drop=True)
    )
    report["duplicate_pair_rows_removed"] = int(before_dedup - len(edges))

    mirna_conflicts = sequence_conflict_ids(edges, "mirna_id", "mirna_seq")
    mrna_conflicts = sequence_conflict_ids(edges, "mrna_id", "mrna_seq")
    if mirna_conflicts or mrna_conflicts:
        conflict_mask = edges["mirna_id"].isin(mirna_conflicts) | edges["mrna_id"].isin(mrna_conflicts)
        report["dropped_node_sequence_conflicts"] = int(conflict_mask.sum())
        edges = edges[~conflict_mask].reset_index(drop=True)

    report["clean_positive_edges"] = int(len(edges))
    return edges, report


def split_edges(edges: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> pd.Series:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")
    rng = np.random.default_rng(seed)
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


def kmers(k: int) -> list[str]:
    words = [""]
    for _ in range(k):
        words = [prefix + base for prefix in words for base in BASES]
    return words


def kmer_feature_names(kmer_sizes: Iterable[int]) -> list[str]:
    names: list[str] = []
    for k in kmer_sizes:
        names.extend([f"kmer{k}_{word}" for word in kmers(k)])
    return names


def sequence_numeric_features(seq: str, kmer_sizes: Iterable[int]) -> dict[str, float]:
    valid = [base for base in seq if base in BASES]
    length = len(valid)
    features: dict[str, float] = {
        "seq_length": float(length),
        "seq_log_length": float(np.log1p(length)),
        "seq_gc": float((valid.count("G") + valid.count("C")) / length) if length else 0.0,
    }
    for k in kmer_sizes:
        vocab = kmers(k)
        counts = {word: 0.0 for word in vocab}
        total = 0
        for idx in range(0, max(len(seq) - k + 1, 0)):
            word = seq[idx : idx + k]
            if all(base in BASES for base in word):
                counts[word] += 1.0
                total += 1
        for word in vocab:
            features[f"kmer{k}_{word}"] = counts[word] / float(total) if total else 0.0
    return features


def seed_features(seq: str) -> dict[str, float]:
    # miRNA seed regions are 1-indexed positions 2-7 and 3-8.
    regions = {
        "seed_2_7": seq[1:7],
        "seed_3_8": seq[2:8],
    }
    features: dict[str, float] = {}
    for name, region in regions.items():
        valid = [base for base in region if base in BASES]
        length = len(valid)
        features[f"{name}_len"] = float(length)
        features[f"{name}_gc"] = float((valid.count("G") + valid.count("C")) / length) if length else 0.0
        for base in BASES:
            features[f"{name}_{base}_freq"] = float(valid.count(base) / length) if length else 0.0
    return features


def build_nodes(edges: pd.DataFrame, kmer_sizes: list[int]) -> tuple[pd.DataFrame, dict[str, int]]:
    mirna_columns = ["mirna_id", "species", "microRNA_name", "mirna_seq", "dataset_id", "Source", "Organism"]
    mrna_columns = [
        "mrna_id",
        "species",
        "mRNA_name",
        "mrna_seq",
        "target_seq",
        "full_mrna_seq",
        "dataset_id",
        "Source",
        "Organism",
        "GI_ID",
    ]
    mirna_nodes = (
        edges.reindex(columns=mirna_columns)
        .drop_duplicates("mirna_id", keep="first")
        .rename(columns={"mirna_id": "node_id", "microRNA_name": "source_name", "mirna_seq": "sequence"})
    )
    mirna_nodes["node_type"] = "mirna"
    mirna_nodes["node_type_id"] = 0

    mrna_nodes = (
        edges.reindex(columns=mrna_columns)
        .drop_duplicates("mrna_id", keep="first")
        .rename(columns={"mrna_id": "node_id", "mRNA_name": "source_name", "mrna_seq": "sequence"})
    )
    mrna_nodes["node_type"] = "mrna"
    mrna_nodes["node_type_id"] = 1

    nodes = pd.concat([mirna_nodes, mrna_nodes], ignore_index=True, sort=False)
    nodes.insert(0, "node_idx", np.arange(len(nodes), dtype=np.int64))
    species_to_id = {species: idx for idx, species in enumerate(SPECIES_ORDER)}
    nodes["species_id"] = nodes["species"].map(species_to_id).fillna(-1).astype("int64")
    nodes["sequence_hash"] = nodes["sequence"].fillna("").astype(str).map(short_hash)
    nodes["mirna_len"] = np.where(nodes["node_type"].eq("mirna"), nodes["sequence"].astype(str).str.len(), np.nan)
    nodes["mrna_len"] = np.where(nodes["node_type"].eq("mrna"), nodes["sequence"].astype(str).str.len(), np.nan)

    feature_rows = []
    for row in nodes.itertuples(index=False):
        seq = str(row.sequence)
        features = sequence_numeric_features(seq, kmer_sizes)
        seeds = seed_features(seq) if row.node_type == "mirna" else {key: 0.0 for key in seed_features("").keys()}
        features.update(seeds)
        feature_rows.append(features)
    feature_df = pd.DataFrame(feature_rows).fillna(0.0).astype("float32")
    nodes = pd.concat([nodes.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    nodes["mirna_gc"] = np.where(nodes["node_type"].eq("mirna"), nodes["seq_gc"], np.nan)
    nodes["mrna_gc"] = np.where(nodes["node_type"].eq("mrna"), nodes["seq_gc"], np.nan)
    node_index = dict(zip(nodes["node_id"].astype(str), nodes["node_idx"].astype(int)))
    return nodes, node_index


def bools_to_numeric(series: pd.Series) -> pd.Series:
    def convert(value):
        try:
            return BOOL_TO_INT.get(value, value)
        except TypeError:
            return value

    return series.map(convert)


def infer_edge_attr_columns(edges: pd.DataFrame, drop_hot_pairing: bool) -> list[str]:
    forbidden = set(METADATA_COLUMNS) | {
        "raw_index",
        "species",
        "dataset_id",
        "source_file",
        "source_row",
        "mirna_id",
        "mrna_id",
        "mirna_seq",
        "target_seq",
        "full_mrna_seq",
        "mrna_seq",
        "label",
        "split",
        "src_idx",
        "dst_idx",
    }
    columns: list[str] = []
    for column in edges.columns:
        if column in forbidden:
            continue
        if drop_hot_pairing and str(column).startswith("HotPairing"):
            continue
        series = bools_to_numeric(edges[column])
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().any():
            columns.append(column)
    return columns


def edge_attr_matrix(edges: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.zeros((len(edges), 0), dtype=np.float32)
    frame = edges.reindex(columns=columns)
    frame = frame.apply(bools_to_numeric)
    frame = frame.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return frame.to_numpy(dtype=np.float32)


def standardize_edge_attr(all_attr: np.ndarray, split: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if all_attr.shape[1] == 0:
        return all_attr, np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    train_mask = split.eq("train").to_numpy()
    train_attr = all_attr[train_mask]
    mean = train_attr.mean(axis=0).astype(np.float32)
    std = train_attr.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ((all_attr - mean) / std).astype(np.float32), mean, std


def standardize_node_features(x_raw: np.ndarray, train_node_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if x_raw.shape[1] == 0:
        return x_raw, np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    if train_node_indices.size == 0:
        train_node_indices = np.arange(x_raw.shape[0], dtype=np.int64)
    train_x = x_raw[train_node_indices]
    mean = train_x.mean(axis=0).astype(np.float32)
    std = train_x.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return ((x_raw - mean) / std).astype(np.float32), mean, std


def export_edges_csv(edges: pd.DataFrame, path: Path) -> None:
    columns = [
        "species",
        "dataset_id",
        "source_file",
        "source_row",
        "split",
        "label",
        "mirna_id",
        "mrna_id",
        "src_idx",
        "dst_idx",
    ]
    present = [column for column in columns if column in edges.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    edges[present].to_csv(path, index=False)


def export_species_graph(
    species: str,
    edges: pd.DataFrame,
    args: argparse.Namespace,
) -> dict[str, object]:
    graph_dir = args.output_dir / species
    graph_dir.mkdir(parents=True, exist_ok=True)
    species_seed = args.seed + sum(ord(ch) for ch in species)
    clean_edges, clean_report = clean_positive_edges(edges)
    clean_edges["split"] = split_edges(clean_edges, species_seed, args.train_ratio, args.val_ratio, args.test_ratio)
    nodes, node_index = build_nodes(clean_edges, args.kmer_sizes)
    clean_edges["src_idx"] = clean_edges["mirna_id"].map(node_index).astype("int64")
    clean_edges["dst_idx"] = clean_edges["mrna_id"].map(node_index).astype("int64")

    drop_hot_pairing = args.drop_hot_pairing and not args.keep_hot_pairing
    edge_attr_columns = infer_edge_attr_columns(clean_edges, drop_hot_pairing=drop_hot_pairing)
    edge_attr_raw = edge_attr_matrix(clean_edges, edge_attr_columns)
    edge_attr_std, edge_attr_mean, edge_attr_scale = standardize_edge_attr(edge_attr_raw, clean_edges["split"])

    nodes_path = graph_dir / "nodes.csv"
    pos_path = graph_dir / "positive_edges.csv"
    nodes.to_csv(nodes_path, index=False)
    export_edges_csv(clean_edges, pos_path)
    for split_name in ["train", "val", "test"]:
        export_edges_csv(clean_edges[clean_edges["split"].eq(split_name)], graph_dir / f"{split_name}_pos_edges.csv")

    node_feature_columns = [
        "seq_log_length",
        "seq_gc",
        *kmer_feature_names(args.kmer_sizes),
        *list(seed_features("").keys()),
    ]
    x_raw = nodes[node_feature_columns].to_numpy(dtype=np.float32)
    node_type = nodes["node_type_id"].to_numpy(dtype=np.int64)
    species_id = nodes["species_id"].to_numpy(dtype=np.int64)
    train_edges = clean_edges[clean_edges["split"].eq("train")]
    edge_index = train_edges[["src_idx", "dst_idx"]].to_numpy(dtype=np.int64).T
    edge_index_undirected = np.concatenate([edge_index, edge_index[::-1]], axis=1) if edge_index.size else edge_index
    all_pos_edge_index = clean_edges[["src_idx", "dst_idx"]].to_numpy(dtype=np.int64).T
    train_node_indices = (
        np.unique(edge_index.reshape(-1)).astype(np.int64)
        if edge_index.size
        else np.arange(len(nodes), dtype=np.int64)
    )
    x, node_feature_mean, node_feature_scale = standardize_node_features(x_raw, train_node_indices)

    arrays: dict[str, object] = {
        "x": x,
        "x_raw": x_raw,
        "node_type": node_type,
        "species_id": species_id,
        "edge_index_train_pos": edge_index,
        "edge_index_train_pos_undirected": edge_index_undirected,
        "all_positive_edge_index": all_pos_edge_index,
        "node_ids": nodes["node_id"].astype(str).to_numpy(dtype=str),
        "node_sequences": nodes["sequence"].fillna("").astype(str).tolist(),
        "node_feature_names": np.asarray(node_feature_columns, dtype=str),
        "node_feature_mean": node_feature_mean,
        "node_feature_scale": node_feature_scale,
        "edge_attr_names": np.asarray(edge_attr_columns, dtype=str),
        "edge_attr_mean": edge_attr_mean,
        "edge_attr_scale": edge_attr_scale,
    }
    for split_name in ["train", "val", "test"]:
        split_mask = clean_edges["split"].eq(split_name).to_numpy()
        split_df = clean_edges[split_mask]
        arrays[f"{split_name}_pos_edge_index"] = split_df[["src_idx", "dst_idx"]].to_numpy(dtype=np.int64).T
        arrays[f"{split_name}_pos_edge_attr_raw"] = edge_attr_raw[split_mask]
        arrays[f"{split_name}_pos_edge_attr"] = edge_attr_std[split_mask]
        arrays[f"{split_name}_pos_label"] = np.ones((int(split_mask.sum()),), dtype=np.int64)

    npz_path = graph_dir / "graph_inputs.npz"
    # Keep long RNA strings out of compressed NPZ; they are only needed by the
    # PyTorch training pipeline for on-the-fly pair features.
    npz_arrays = {key: value for key, value in arrays.items() if key != "node_sequences"}
    np.savez_compressed(npz_path, **npz_arrays)

    try:
        import torch

        torch_path = graph_dir / "graph_inputs.pt"
        tmp_torch_path = graph_dir / "graph_inputs.pt.tmp"
        torch.save({key: value for key, value in arrays.items()}, tmp_torch_path)
        tmp_torch_path.replace(torch_path)
        torch_output = str(torch_path)
    except ModuleNotFoundError:
        torch_output = ""

    metadata = {
        "species": species,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "cleaning": clean_report,
        "num_nodes": int(len(nodes)),
        "num_mirna_nodes": int(nodes["node_type"].eq("mirna").sum()),
        "num_mrna_nodes": int(nodes["node_type"].eq("mrna").sum()),
        "num_positive_edges": int(len(clean_edges)),
        "split_counts": {
            split_name: int(clean_edges["split"].eq(split_name).sum())
            for split_name in ["train", "val", "test"]
        },
        "num_node_features": int(x.shape[1]),
        "num_edge_attr": int(edge_attr_raw.shape[1]),
        "node_feature_columns": node_feature_columns,
        "node_feature_normalization": {
            "length_feature": "seq_log_length = log1p(seq_length)",
            "standardization": "z-score fit on nodes appearing in train positive edges only",
            "num_train_graph_nodes_for_fit": int(train_node_indices.size),
        },
        "edge_attr_columns": edge_attr_columns,
        "negative_sampling": (
            "Dynamic only: sample from all miRNA-mRNA pairs excluding all known positive pairs. "
            "Negative edges are not stored and do not enter edge_index."
        ),
        "paths": {
            "nodes": str(nodes_path),
            "positive_edges": str(pos_path),
            "graph_inputs_npz": str(npz_path),
            "graph_inputs_pt": torch_output,
        },
    }
    (graph_dir / "metadata.json").write_text(json.dumps(json_safe(metadata), indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    args = parse_args()
    files = discover_pos_files(args.input_dir)
    raw_by_species: dict[str, list[pd.DataFrame]] = {species: [] for species in SPECIES_ORDER}
    for raw_file in files:
        if raw_file.species not in raw_by_species:
            raw_by_species[raw_file.species] = []
        raw_by_species[raw_file.species].append(read_positive_file(raw_file))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "input_dir": args.input_dir,
        "output_dir": args.output_dir,
        "seed": args.seed,
        "kmer_sizes": args.kmer_sizes,
        "note": "Only *_pos.csv files are used. Negative edges are sampled dynamically during training.",
        "species": {},
    }
    for species in SPECIES_ORDER:
        frames = raw_by_species.get(species, [])
        if not frames:
            continue
        species_edges = pd.concat(frames, ignore_index=True, sort=False)
        metadata = export_species_graph(species, species_edges, args)
        summary["species"][species] = metadata
        print(
            f"{species}: nodes={metadata['num_nodes']} pos_edges={metadata['num_positive_edges']} "
            f"train/val/test={metadata['split_counts']}",
            flush=True,
        )
    (args.output_dir / "preprocess_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2),
        encoding="utf-8",
    )
    print(f"Saved graph preprocessing outputs to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
