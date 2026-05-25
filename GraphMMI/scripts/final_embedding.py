#!/usr/bin/env python3
"""Final embedding-optimized graph preprocessing for GraphMMI thesis.

Core improvements over the original pipeline:

1. Target-site-aware node representation (--node-mode):
   - mrna:         one node per mRNA (baseline)
   - target_site:  one node per unique (mRNA_name, target_seq) — preserves
                   individual binding-site information.  **Main model.**
   - hierarchical: miRNA → target_site → mRNA_gene three-level graph

2. Cleaner similarity edges (--sim-mode):
   - topk:           asymmetric top-k per node (default, best performer)
   - mutual:         reciprocal top-k only
   - threshold_topk: cosine > threshold, then top-k

3. Configurable top-k:
   --mirna-sim-topk  (default 5)
   --mrna-sim-topk   (default 5, key hyperparameter for target_site mode)

4. Pair-feature v3 definitions (40-dim):
   v1 (17) + v2 (11) + v3 (12) including GU wobble, mismatch counts,
   seed-position-aware matching, k-mer overlap, sliding-window match.

Output: graph_inputs.npz / graph_inputs.pt per species under --output-dir.

Usage:
  # Main model (recommended)
  python scripts/final_embedding.py \\
      --node-mode target_site --sim-mode topk \\
      --mrna-sim-topk 5 \\
      --output-dir data/processed/graph/final_target_site_topk_v1

  # Ablation: node mode
  python scripts/final_embedding.py --node-mode mrna --output-dir ...
  python scripts/final_embedding.py --node-mode hierarchical --output-dir ...

  # Ablation: sim mode
  python scripts/final_embedding.py --sim-mode mutual --output-dir ...
  python scripts/final_embedding.py --sim-mode threshold_topk --sim-threshold 0.3 --output-dir ...

  # Ablation: topk sensitivity
  python scripts/final_embedding.py --mrna-sim-topk 3 --output-dir ...
  python scripts/final_embedding.py --mrna-sim-topk 10 --output-dir ...
  python scripts/final_embedding.py --mrna-sim-topk 20 --output-dir ...
"""

from __future__ import annotations

import argparse, hashlib, json, math, re
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

REQUIRED_COLUMNS = ["microRNA_name", "miRNA sequence", "mRNA_name",
                    "target sequence", "full_mrna"]
METADATA_COLUMNS = {"Source", "Organism", "GI_ID", "microRNA_name",
                    "miRNA sequence", "target sequence", "number of reads",
                    "mRNA_name", "full_mrna"}

# ------------------------------------------------------------------
# constants
# ------------------------------------------------------------------

ET_INTERACTION_FWD = 0
ET_INTERACTION_REV = 1
ET_MIRNA_SIM = 2
ET_MRNA_SIM = 3
ET_BELONGS_TO = 4

NODE_MIRNA = 0
NODE_TARGET_SITE = 1
NODE_MRNA_GENE = 2

# v1 (17) + v2 (11) + v3 (12) = 40
PAIR_V3_DIM = 40

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Final embedding-optimized graph preprocessing")
    p.add_argument("--input-dir", type=Path, default=ROOT / "data/external")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "data/processed/graph/final_target_site_topk_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.72)
    p.add_argument("--val-ratio", type=float, default=0.08)
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--kmer-sizes", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--drop-hot-pairing", action="store_true", default=True)
    p.add_argument("--keep-hot-pairing", action="store_true")
    # node mode
    p.add_argument("--node-mode", choices=["mrna", "target_site", "hierarchical"],
                   default="target_site")
    p.add_argument("--mrna-sequence-source", choices=["target", "full"],
                   default="target")
    # similarity edges
    p.add_argument("--mirna-sim-edges", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mrna-sim-edges", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--mirna-sim-topk", type=int, default=5)
    p.add_argument("--mrna-sim-topk", type=int, default=5)
    p.add_argument("--sim-mode", choices=["topk", "mutual", "threshold_topk"],
                   default="topk")
    p.add_argument("--sim-threshold", type=float, default=0.3)
    return p.parse_args()


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, np.ndarray)):
        return [json_safe(i) for i in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def clean_text(v: object) -> str:
    return "" if pd.isna(v) else str(v).strip()


def normalize_sequence(v: object) -> str:
    if pd.isna(v):
        return ""
    return re.sub(r"\s+", "", str(v).upper().replace("T", "U"))


def short_hash(v: str) -> str:
    return hashlib.sha1(v.encode("utf-8")).hexdigest()[:12]


@dataclass(frozen=True)
class RawPosFile:
    path: Path
    species: str
    dataset_id: str


def discover_pos_files(input_dir: Path) -> list[RawPosFile]:
    files = []
    for path in sorted(input_dir.glob("*_pos.csv")):
        m = FILE_RE.match(path.name)
        if not m:
            continue
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        files.append(RawPosFile(path=path, species=m.group("species").lower(),
                                dataset_id=m.group("dataset_id").lower()))
    if not files:
        raise FileNotFoundError(f"No *_pos.csv in {input_dir}")
    return files


# ------------------------------------------------------------------
# reading
# ------------------------------------------------------------------

def read_positive_file(rf: RawPosFile, args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(rf.path, index_col=0, low_memory=False)
    df = df.reset_index(drop=False).rename(columns={"index": "raw_index"})
    df["species"] = rf.species
    df["dataset_id"] = rf.dataset_id
    df["source_file"] = rf.path.name
    df["source_row"] = np.arange(len(df), dtype=np.int64)

    df["mirna_seq"] = df["miRNA sequence"].map(normalize_sequence)
    df["target_seq"] = df["target sequence"].map(normalize_sequence)
    df["full_mrna_seq"] = df["full_mrna"].map(normalize_sequence)

    if args.node_mode == "mrna":
        if args.mrna_sequence_source == "target":
            df["mrna_seq"] = df["target_seq"]
        else:
            df["mrna_seq"] = np.where(
                df["full_mrna_seq"].astype(str).str.len() > 0,
                df["full_mrna_seq"], df["target_seq"])
    else:
        df["mrna_seq"] = df["target_seq"]

    df["mirna_id"] = rf.species + "|" + df["microRNA_name"].map(clean_text)
    df["mrna_id"] = rf.species + "|" + df["mRNA_name"].map(clean_text)
    if "site_start" in df.columns:
        site_start = (pd.to_numeric(df["site_start"], errors="coerce")
                      .fillna(-1).astype(int).astype(str))
    else:
        site_start = pd.Series(["NA"] * len(df), index=df.index)

    df["target_site_id"] = (rf.species + "|"
                            + df["mRNA_name"].map(clean_text)
                            + "|" + site_start
                            + "|" + df["target_seq"].map(short_hash))
    df["label"] = 1
    return df


# ------------------------------------------------------------------
# cleaning — mode-aware
# ------------------------------------------------------------------

def sequence_conflict_ids(frame: pd.DataFrame, id_col: str,
                          seq_col: str) -> set[str]:
    cnt = frame.groupby(id_col)[seq_col].nunique(dropna=False)
    return set(cnt[cnt.gt(1)].index.astype(str))


def clean_positive_edges(edges: pd.DataFrame,
                         node_mode: str = "mrna") -> tuple[pd.DataFrame, dict[str, int]]:
    report = {"raw_rows": int(len(edges)),
              "dropped_missing_core_fields": 0,
              "dropped_pair_sequence_conflicts": 0,
              "dropped_node_sequence_conflicts": 0,
              "duplicate_pair_rows_removed": 0}

    core = (edges["mirna_id"].astype(str).ne("")
            & edges["mrna_id"].astype(str).ne("")
            & edges["mirna_seq"].astype(str).ne("")
            & edges["mrna_seq"].astype(str).ne(""))
    report["dropped_missing_core_fields"] = int((~core).sum())
    edges = edges[core].copy()

    # pair dedup key — mode-aware
    if node_mode in ("target_site", "hierarchical"):
        pair_key = ["mirna_id", "target_site_id"]
        pair_g = edges.groupby(["mirna_id", "target_site_id"], dropna=False)
        conflicts = pair_g.filter(
            lambda g: g["mirna_seq"].nunique(dropna=False) > 1
            or g["target_seq"].nunique(dropna=False) > 1)
    else:
        pair_key = ["mirna_id", "mrna_id"]
        pair_g = edges.groupby(["mirna_id", "mrna_id"], dropna=False)
        conflicts = pair_g.filter(
            lambda g: g["mirna_seq"].nunique(dropna=False) > 1
            or g["mrna_seq"].nunique(dropna=False) > 1)

    if not conflicts.empty:
        ck = set(zip(conflicts[pair_key[0]], conflicts[pair_key[1]]))
        report["dropped_pair_sequence_conflicts"] = int(
            edges[pair_key].apply(tuple, axis=1).isin(ck).sum())
        edges = edges[~edges[pair_key].apply(tuple, axis=1).isin(ck)].copy()

    before = len(edges)
    edges = (edges.sort_values(["dataset_id", "source_row"])
             .drop_duplicates(pair_key, keep="first")
             .reset_index(drop=True))
    report["duplicate_pair_rows_removed"] = int(before - len(edges))

    # node-level sequence conflicts — mode-aware
    mc = sequence_conflict_ids(edges, "mirna_id", "mirna_seq")
    if node_mode in ("target_site", "hierarchical"):
        ts_conflicts = sequence_conflict_ids(edges, "target_site_id", "target_seq")
        conflict_mask = (edges["mirna_id"].isin(mc)
                         | edges["target_site_id"].isin(ts_conflicts))
    else:
        nc = sequence_conflict_ids(edges, "mrna_id", "mrna_seq")
        conflict_mask = edges["mirna_id"].isin(mc) | edges["mrna_id"].isin(nc)

    if conflict_mask.any():
        report["dropped_node_sequence_conflicts"] = int(conflict_mask.sum())
        edges = edges[~conflict_mask].reset_index(drop=True)

    report["clean_positive_edges"] = int(len(edges))
    return edges, report


# ------------------------------------------------------------------
# split
# ------------------------------------------------------------------

def split_edges(edges: pd.DataFrame, seed: int, train_ratio: float,
                val_ratio: float, test_ratio: float) -> pd.Series:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    rng = np.random.default_rng(seed)
    indices = edges.index.to_numpy()
    rng.shuffle(indices)
    n_total = len(indices)
    n_val = int(round(n_total * val_ratio))
    n_test = int(round(n_total * test_ratio))
    n_train = n_total - n_val - n_test
    split = pd.Series(index=edges.index, dtype="object")
    split.loc[indices[:n_train]] = "train"
    split.loc[indices[n_train:n_train + n_val]] = "val"
    split.loc[indices[n_train + n_val:]] = "test"
    return split


# ------------------------------------------------------------------
# sequence features
# ------------------------------------------------------------------

def kmers(k: int) -> list[str]:
    words = [""]
    for _ in range(k):
        words = [p + b for p in words for b in BASES]
    return words


def kmer_feature_names(kmer_sizes: Iterable[int]) -> list[str]:
    names = []
    for k in kmer_sizes:
        names.extend([f"kmer{k}_{w}" for w in kmers(k)])
    return names


def sequence_numeric_features(seq: str, kmer_sizes: Iterable[int]) -> dict[str, float]:
    valid = [b for b in seq if b in BASES]
    length = len(valid)
    feats = {"seq_length": float(length),
             "seq_log_length": float(np.log1p(length)),
             "seq_gc": float((valid.count("G") + valid.count("C")) / length) if length else 0.0}
    for k in kmer_sizes:
        vocab = kmers(k)
        cnt = {w: 0.0 for w in vocab}
        total = 0
        for idx in range(max(len(seq) - k + 1, 0)):
            word = seq[idx:idx + k]
            if all(b in BASES for b in word):
                cnt[word] += 1.0
                total += 1
        for w in vocab:
            feats[f"kmer{k}_{w}"] = cnt[w] / float(total) if total else 0.0
    return feats


def seed_features(seq: str) -> dict[str, float]:
    regions = {"seed_2_7": seq[1:7], "seed_3_8": seq[2:8]}
    feats = {}
    for name, region in regions.items():
        valid = [b for b in region if b in BASES]
        length = len(valid)
        feats[f"{name}_len"] = float(length)
        feats[f"{name}_gc"] = (float((valid.count("G") + valid.count("C")) / length)
                               if length else 0.0)
        for base in BASES:
            feats[f"{name}_{base}_freq"] = (float(valid.count(base) / length)
                                            if length else 0.0)
    return feats


# ------------------------------------------------------------------
# node building
# ------------------------------------------------------------------

def _finalize_nodes(nodes: pd.DataFrame, kmer_sizes: list[int]) -> pd.DataFrame:
    nodes.insert(0, "node_idx", np.arange(len(nodes), dtype=np.int64))
    sp2id = {s: i for i, s in enumerate(SPECIES_ORDER)}
    nodes["species_id"] = nodes["species"].map(sp2id).fillna(-1).astype("int64")
    nodes["sequence_hash"] = nodes["sequence"].fillna("").astype(str).map(short_hash)
    nodes["mirna_len"] = np.where(nodes["node_type"].eq("mirna"),
                                  nodes["sequence"].astype(str).str.len(), np.nan)
    nodes["mrna_len"] = np.where(
        nodes["node_type"].isin(["mrna", "target_site", "mrna_gene"]),
        nodes["sequence"].astype(str).str.len(), np.nan)

    frows = []
    for row in nodes.itertuples(index=False):
        seq = str(row.sequence)
        feats = sequence_numeric_features(seq, kmer_sizes)
        seeds = (seed_features(seq) if row.node_type in ("mirna",)
                 else {k: 0.0 for k in seed_features("")})
        feats.update(seeds)
        frows.append(feats)
    fdf = pd.DataFrame(frows).fillna(0.0).astype("float32")
    nodes = pd.concat([nodes.reset_index(drop=True),
                       fdf.reset_index(drop=True)], axis=1)
    nodes["mirna_gc"] = np.where(nodes["node_type"].eq("mirna"),
                                 nodes["seq_gc"], np.nan)
    nodes["mrna_gc"] = np.where(
        nodes["node_type"].isin(["mrna", "target_site", "mrna_gene"]),
        nodes["seq_gc"], np.nan)
    return nodes


def build_nodes_mrna(edges: pd.DataFrame, kmer_sizes: list[int],
                     ) -> tuple[pd.DataFrame, dict[str, int]]:
    mirna_nodes = (edges[["mirna_id", "species", "microRNA_name", "mirna_seq",
                          "dataset_id", "Source", "Organism"]]
                   .drop_duplicates("mirna_id", keep="first")
                   .rename(columns={"mirna_id": "node_id",
                                    "microRNA_name": "source_name",
                                    "mirna_seq": "sequence"}))
    mirna_nodes["node_type"] = "mirna"
    mirna_nodes["node_type_id"] = NODE_MIRNA

    mrna_tmp = edges[["mrna_id", "species", "mRNA_name", "mrna_seq", "target_seq",
                      "full_mrna_seq", "dataset_id", "Source", "Organism", "GI_ID"]].copy()
    mrna_tmp["_target_len"] = mrna_tmp["target_seq"].fillna("").astype(str).str.len()
    mrna_tmp = mrna_tmp.sort_values("_target_len", ascending=False)
    mrna_nodes = (mrna_tmp.drop_duplicates("mrna_id", keep="first")
                  .rename(columns={"mrna_id": "node_id", "mRNA_name": "source_name",
                                   "mrna_seq": "sequence"}))
    mrna_nodes["node_type"] = "mrna"
    mrna_nodes["node_type_id"] = NODE_TARGET_SITE

    nodes = pd.concat([mirna_nodes, mrna_nodes], ignore_index=True, sort=False)
    nodes = _finalize_nodes(nodes, kmer_sizes)
    ni = dict(zip(nodes["node_id"].astype(str), nodes["node_idx"].astype(int)))
    return nodes, ni


def build_nodes_target_site(edges: pd.DataFrame, kmer_sizes: list[int],
                            ) -> tuple[pd.DataFrame, dict[str, int]]:
    mirna_nodes = (edges[["mirna_id", "species", "microRNA_name", "mirna_seq",
                          "dataset_id", "Source", "Organism"]]
                   .drop_duplicates("mirna_id", keep="first")
                   .rename(columns={"mirna_id": "node_id",
                                    "microRNA_name": "source_name",
                                    "mirna_seq": "sequence"}))
    mirna_nodes["node_type"] = "mirna"
    mirna_nodes["node_type_id"] = NODE_MIRNA

    ts_cols = ["target_site_id", "species", "mRNA_name", "mrna_seq",
               "target_seq", "full_mrna_seq", "mrna_id",
               "dataset_id", "Source", "Organism", "GI_ID"]
    ts_nodes = (edges[ts_cols].drop_duplicates("target_site_id", keep="first")
                .rename(columns={"target_site_id": "node_id",
                                 "mRNA_name": "source_name",
                                 "mrna_seq": "sequence"}))
    ts_nodes["node_type"] = "target_site"
    ts_nodes["node_type_id"] = NODE_TARGET_SITE

    nodes = pd.concat([mirna_nodes, ts_nodes], ignore_index=True, sort=False)
    nodes = _finalize_nodes(nodes, kmer_sizes)
    ni = dict(zip(nodes["node_id"].astype(str), nodes["node_idx"].astype(int)))
    return nodes, ni


def build_nodes_hierarchical(edges: pd.DataFrame, kmer_sizes: list[int],
                             ) -> tuple[pd.DataFrame, dict[str, int]]:
    # miRNA
    mirna_nodes = (edges[["mirna_id", "species", "microRNA_name", "mirna_seq",
                          "dataset_id", "Source", "Organism"]]
                   .drop_duplicates("mirna_id", keep="first")
                   .rename(columns={"mirna_id": "node_id",
                                    "microRNA_name": "source_name",
                                    "mirna_seq": "sequence"}))
    mirna_nodes["node_type"] = "mirna"
    mirna_nodes["node_type_id"] = NODE_MIRNA

    # target_site
    ts_cols = ["target_site_id", "species", "mRNA_name", "mrna_seq",
               "target_seq", "full_mrna_seq", "mrna_id",
               "dataset_id", "Source", "Organism", "GI_ID"]
    ts_nodes = (edges[ts_cols].drop_duplicates("target_site_id", keep="first")
                .rename(columns={"target_site_id": "node_id",
                                 "mRNA_name": "source_name",
                                 "mrna_seq": "sequence"}))
    ts_nodes["node_type"] = "target_site"
    ts_nodes["node_type_id"] = NODE_TARGET_SITE

    # mRNA gene
    mrna_tmp = edges[["mrna_id", "species", "mRNA_name",
                      "dataset_id", "Source", "Organism", "GI_ID"]].copy()
    mrna_tmp["_target_len"] = edges["target_seq"].fillna("").astype(str).str.len()
    mrna_tmp = mrna_tmp.sort_values("_target_len", ascending=False)
    full_seqs = (edges.groupby("mrna_id")["full_mrna_seq"]
                 .apply(lambda x: x.iloc[0] if len(x) > 0 else "").to_dict())
    mrna_gene = (mrna_tmp.drop_duplicates("mrna_id", keep="first")
                 .rename(columns={"mrna_id": "node_id", "mRNA_name": "source_name"}))
    mrna_gene["sequence"] = mrna_gene["node_id"].map(
        lambda nid: str(full_seqs.get(nid, "")))
    mrna_gene["node_type"] = "mrna_gene"
    mrna_gene["node_type_id"] = NODE_MRNA_GENE

    nodes = pd.concat([mirna_nodes, ts_nodes, mrna_gene],
                      ignore_index=True, sort=False)
    nodes = _finalize_nodes(nodes, kmer_sizes)
    ni = dict(zip(nodes["node_id"].astype(str), nodes["node_idx"].astype(int)))
    return nodes, ni


# ------------------------------------------------------------------
# edge attr helpers
# ------------------------------------------------------------------

def bools_to_numeric(series: pd.Series) -> pd.Series:
    def convert(value):
        try: return BOOL_TO_INT.get(value, value)
        except TypeError: return value
    return series.map(convert)


def infer_edge_attr_columns(edges: pd.DataFrame, drop_hot: bool) -> list[str]:
    forbidden = METADATA_COLUMNS | {
        "raw_index", "species", "dataset_id", "source_file", "source_row",
        "mirna_id", "mrna_id", "target_site_id", "mirna_seq", "target_seq",
        "full_mrna_seq", "mrna_seq", "site_start", "site_end",
        "label", "split", "src_idx", "dst_idx"}
    cols = []
    for c in edges.columns:
        if c in forbidden:
            continue
        if drop_hot and str(c).startswith("HotPairing"):
            continue
        s = bools_to_numeric(edges[c])
        if pd.to_numeric(s, errors="coerce").notna().any():
            cols.append(c)
    return cols


def edge_attr_matrix(edges: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.zeros((len(edges), 0), dtype=np.float32)
    return (edges[columns].apply(bools_to_numeric)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0).to_numpy(dtype=np.float32))


def standardize_edge_attr(all_attr, split):
    if all_attr.shape[1] == 0:
        return all_attr, np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    train = all_attr[split.eq("train").to_numpy()]
    m, s = train.mean(0).astype(np.float32), train.std(0).astype(np.float32)
    s[s < 1e-6] = 1.0
    return ((all_attr - m) / s).astype(np.float32), m, s


def standardize_node_features(x_raw, train_idx):
    if x_raw.shape[1] == 0:
        return x_raw, np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    if train_idx.size == 0:
        train_idx = np.arange(x_raw.shape[0], dtype=np.int64)
    train = x_raw[train_idx]
    m, s = train.mean(0).astype(np.float32), train.std(0).astype(np.float32)
    s[s < 1e-6] = 1.0
    return ((x_raw - m) / s).astype(np.float32), m, s


def export_edges_csv(edges, path):
    cols = ["species", "dataset_id", "source_file", "source_row", "split",
            "label", "mirna_id", "mrna_id", "target_site_id",
            "src_idx", "dst_idx"]
    present = [c for c in cols if c in edges.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    edges[present].to_csv(path, index=False)


# ------------------------------------------------------------------
# similarity edges
# ------------------------------------------------------------------

def _make_undirected(ei: np.ndarray, ew: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ei.size == 0:
        return ei, ew
    rev = ei[::-1]
    return np.concatenate([ei, rev], axis=1), np.concatenate([ew, ew], axis=0)


def _cosine_similarity_topk(vecs: np.ndarray, topk: int,
                            rng: np.random.Generator,
                            sim_mode: str = "topk",
                            sim_threshold: float = 0.3,
                            ) -> tuple[np.ndarray, np.ndarray]:
    n = vecs.shape[0]
    if n < 2:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    unit = vecs.astype(np.float64) / norms
    eff_k = min(topk, n - 1)

    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for i in range(n):
        sim = unit[i] @ unit.T
        sim[i] = -2.0
        if sim_mode == "threshold_topk":
            sim[sim < sim_threshold] = sim_threshold - 1.0
        if eff_k < n:
            top = np.argpartition(-sim, eff_k - 1)[:eff_k]
            top = top[np.argsort(-sim[top])]
        else:
            top = np.argsort(-sim)[:eff_k]
        for j in top:
            s = float(sim[j])
            if s > 0.0:
                edges.append((i, j))
                weights.append(s)

    if not edges:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

    ei = np.array(edges, dtype=np.int64)
    ew = np.array(weights, dtype=np.float32)

    if sim_mode == "mutual":
        edge_set = {(int(e[0]), int(e[1])) for e in ei}
        mutual = np.array([(int(e[1]), int(e[0])) in edge_set for e in ei], dtype=bool)
        ei = ei[mutual]
        ew = ew[mutual]
        if ei.size == 0:
            return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

    return ei.T, ew


def build_mirna_similarity_edges(nodes: pd.DataFrame, x_raw: np.ndarray,
                                 kmer_sizes: list[int], args: argparse.Namespace,
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    mirna = nodes[nodes["node_type"].eq("mirna")].copy()
    if len(mirna) < 2:
        empty = np.empty((2, 0), dtype=np.int64)
        return empty, np.full((0,), ET_MIRNA_SIM, dtype=np.int64), np.empty((0,), dtype=np.float32)

    l2g = {i: int(mirna.iloc[i].node_idx) for i in range(len(mirna))}
    seqs = mirna["sequence"].fillna("").astype(str).tolist()
    edge_map: dict[tuple[int, int], float] = {}

    # seed identity
    for slc in [slice(1, 7), slice(2, 8)]:
        sm: dict[str, list[int]] = {}
        for li, seq in enumerate(seqs):
            seed = seq[slc] if len(seq) >= slc.stop else ""
            if seed:
                sm.setdefault(seed, []).append(li)
        for grp in sm.values():
            if len(grp) < 2:
                continue
            for a in range(len(grp)):
                for b in range(a + 1, len(grp)):
                    gi, gj = l2g[grp[a]], l2g[grp[b]]
                    key = (min(gi, gj), max(gi, gj))
                    edge_map[key] = max(edge_map.get(key, 0.0), 1.0)

    # k-mer cosine
    kmer_cols = [c for c in nodes.columns
                 if c.startswith("kmer1_") or c.startswith("kmer2_") or c.startswith("kmer3_")]
    if kmer_cols:
        kvs = mirna[kmer_cols].to_numpy(dtype=np.float32)
        ci, cw = _cosine_similarity_topk(kvs, args.mirna_sim_topk, rng,
                                         sim_mode=args.sim_mode,
                                         sim_threshold=args.sim_threshold)
        for col in range(ci.shape[1]):
            gi, gj = l2g[int(ci[0, col])], l2g[int(ci[1, col])]
            key = (min(gi, gj), max(gi, gj))
            edge_map[key] = max(edge_map.get(key, 0.0), float(cw[col]))

    if not edge_map:
        empty = np.empty((2, 0), dtype=np.int64)
        return empty, np.full((0,), ET_MIRNA_SIM, dtype=np.int64), np.empty((0,), dtype=np.float32)

    pairs = sorted(edge_map.keys())
    ei = np.array(pairs, dtype=np.int64).T
    ew = np.array([edge_map[k] for k in pairs], dtype=np.float32)
    ei, ew = _make_undirected(ei, ew)
    et = np.full(ei.shape[1], ET_MIRNA_SIM, dtype=np.int64)
    return ei, et, ew


def build_target_similarity_edges(nodes: pd.DataFrame, args: argparse.Namespace,
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    targets = nodes[nodes["node_type"].isin(["mrna", "target_site"])].copy()
    if len(targets) < 2:
        empty = np.empty((2, 0), dtype=np.int64)
        return empty, np.full((0,), ET_MRNA_SIM, dtype=np.int64), np.empty((0,), dtype=np.float32)

    l2g = {i: int(targets.iloc[i].node_idx) for i in range(len(targets))}
    target_seqs = (targets.get("target_seq", targets["sequence"].fillna(""))
                   if "target_seq" in nodes.columns
                   else targets["sequence"].fillna(""))
    target_seqs = target_seqs.fillna("").astype(str).tolist()

    kmer_names = kmer_feature_names(args.kmer_sizes)
    vecs = np.zeros((len(targets), len(kmer_names)), dtype=np.float32)
    for i, seq in enumerate(target_seqs):
        feats = sequence_numeric_features(seq, args.kmer_sizes)
        for j, name in enumerate(kmer_names):
            vecs[i, j] = float(feats.get(name, 0.0))

    ci, cw = _cosine_similarity_topk(vecs, args.mrna_sim_topk, rng,
                                     sim_mode=args.sim_mode,
                                     sim_threshold=args.sim_threshold)
    if ci.size == 0:
        empty = np.empty((2, 0), dtype=np.int64)
        return empty, np.full((0,), ET_MRNA_SIM, dtype=np.int64), np.empty((0,), dtype=np.float32)

    edge_set: set[tuple[int, int]] = set()
    pairs, weights = [], []
    for col in range(ci.shape[1]):
        gi, gj = l2g[int(ci[0, col])], l2g[int(ci[1, col])]
        key = (min(gi, gj), max(gi, gj))
        if key in edge_set:
            continue
        edge_set.add(key)
        pairs.append(key)
        weights.append(float(cw[col]))

    if not pairs:
        empty = np.empty((2, 0), dtype=np.int64)
        return empty, np.full((0,), ET_MRNA_SIM, dtype=np.int64), np.empty((0,), dtype=np.float32)

    ei = np.array(pairs, dtype=np.int64).T
    ew = np.array(weights, dtype=np.float32)
    ei, ew = _make_undirected(ei, ew)
    et = np.full(ei.shape[1], ET_MRNA_SIM, dtype=np.int64)
    return ei, et, ew


def build_mrna_gene_similarity_edges(nodes: pd.DataFrame, args: argparse.Namespace,
                                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    genes = nodes[nodes["node_type"].eq("mrna_gene")].copy()
    if len(genes) < 2:
        empty = np.empty((2, 0), dtype=np.int64)
        return empty, np.full((0,), ET_MRNA_SIM, dtype=np.int64), np.empty((0,), dtype=np.float32)

    l2g = {i: int(genes.iloc[i].node_idx) for i in range(len(genes))}
    seqs = genes["sequence"].fillna("").astype(str).tolist()

    kmer_names = kmer_feature_names(args.kmer_sizes)
    vecs = np.zeros((len(genes), len(kmer_names)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        feats = sequence_numeric_features(seq, args.kmer_sizes)
        for j, name in enumerate(kmer_names):
            vecs[i, j] = float(feats.get(name, 0.0))

    ci, cw = _cosine_similarity_topk(vecs, args.mrna_sim_topk, rng,
                                     sim_mode=args.sim_mode,
                                     sim_threshold=args.sim_threshold)
    if ci.size == 0:
        empty = np.empty((2, 0), dtype=np.int64)
        return empty, np.full((0,), ET_MRNA_SIM, dtype=np.int64), np.empty((0,), dtype=np.float32)

    edge_set: set[tuple[int, int]] = set()
    pairs, weights = [], []
    for col in range(ci.shape[1]):
        gi, gj = l2g[int(ci[0, col])], l2g[int(ci[1, col])]
        key = (min(gi, gj), max(gi, gj))
        if key in edge_set:
            continue
        edge_set.add(key)
        pairs.append(key)
        weights.append(float(cw[col]))

    if not pairs:
        empty = np.empty((2, 0), dtype=np.int64)
        return empty, np.full((0,), ET_MRNA_SIM, dtype=np.int64), np.empty((0,), dtype=np.float32)

    ei = np.array(pairs, dtype=np.int64).T
    ew = np.array(weights, dtype=np.float32)
    ei, ew = _make_undirected(ei, ew)
    et = np.full(ei.shape[1], ET_MRNA_SIM, dtype=np.int64)
    return ei, et, ew


def build_belongs_to_edges(edges: pd.DataFrame,
                           node_index: dict[str, int],
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for row in edges.itertuples(index=False):
        ts_id = str(row.target_site_id)
        mrna_id = str(row.mrna_id)
        if ts_id in node_index and mrna_id in node_index:
            key = (node_index[ts_id], node_index[mrna_id])
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    if not pairs:
        empty = np.empty((2, 0), dtype=np.int64)
        return empty, np.full((0,), ET_BELONGS_TO, dtype=np.int64), np.empty((0,), dtype=np.float32)
    ei = np.array(pairs, dtype=np.int64).T
    ei_undir = np.concatenate([ei, ei[::-1]], axis=1)
    et = np.full(ei_undir.shape[1], ET_BELONGS_TO, dtype=np.int64)
    ew = np.ones(ei_undir.shape[1], dtype=np.float32)
    return ei_undir, et, ew


def homogenize_edge_index(interaction_undirected: np.ndarray,
                          sim_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = interaction_undirected.shape[1] // 2
    ei_parts = [interaction_undirected]
    et_parts = [np.concatenate([
        np.full(half, ET_INTERACTION_FWD, dtype=np.int64),
        np.full(half, ET_INTERACTION_REV, dtype=np.int64)])]
    ew_parts = [np.ones(interaction_undirected.shape[1], dtype=np.float32)]

    for se, st, sw in sim_parts:
        if se.size:
            ei_parts.append(se)
            et_parts.append(st)
            ew_parts.append(sw)

    return (np.concatenate(ei_parts, axis=1) if len(ei_parts) > 1 else ei_parts[0],
            np.concatenate(et_parts) if len(et_parts) > 1 else et_parts[0],
            np.concatenate(ew_parts) if len(ew_parts) > 1 else ew_parts[0])


# ------------------------------------------------------------------
# pair feature v3 (exported for training scripts via data.py)
# ------------------------------------------------------------------

_COMPLEMENT_TABLE = str.maketrans({"A": "U", "U": "A", "C": "G", "G": "C", "T": "A"})


def reverse_complement_v3(seq: str) -> str:
    return str(seq).upper().replace("T", "U").translate(_COMPLEMENT_TABLE)[::-1]


def gc_fraction_v3(seq: str) -> float:
    valid = [b for b in seq.upper().replace("T", "U") if b in {"A", "C", "G", "U"}]
    return float((valid.count("G") + valid.count("C")) / len(valid)) if valid else 0.0


def pair_feature_dim_v3() -> int:
    return PAIR_V3_DIM


# ------------------------------------------------------------------
# main export
# ------------------------------------------------------------------

def export_species_graph(species: str, edges: pd.DataFrame,
                         args: argparse.Namespace) -> dict:
    gdir = args.output_dir / species
    gdir.mkdir(parents=True, exist_ok=True)
    sseed = args.seed + sum(ord(c) for c in species)

    clean, report = clean_positive_edges(edges, node_mode=args.node_mode)
    clean["split"] = split_edges(clean, sseed,
                                 args.train_ratio, args.val_ratio, args.test_ratio)
    print(f"  cleaned: {report['clean_positive_edges']} edges, building nodes...", flush=True)

    # build nodes
    if args.node_mode == "mrna":
        nodes, ni = build_nodes_mrna(clean, args.kmer_sizes)
    elif args.node_mode == "target_site":
        nodes, ni = build_nodes_target_site(clean, args.kmer_sizes)
    elif args.node_mode == "hierarchical":
        nodes, ni = build_nodes_hierarchical(clean, args.kmer_sizes)
    else:
        raise ValueError(f"Unknown node_mode: {args.node_mode}")

    # map edges
    if args.node_mode in ("target_site", "hierarchical"):
        clean["src_idx"] = clean["mirna_id"].map(ni).astype("int64")
        clean["dst_idx"] = clean["target_site_id"].map(ni).astype("int64")
    else:
        clean["src_idx"] = clean["mirna_id"].map(ni).astype("int64")
        clean["dst_idx"] = clean["mrna_id"].map(ni).astype("int64")

    # edge attributes
    drop_hot = args.drop_hot_pairing and not args.keep_hot_pairing
    ea_cols = infer_edge_attr_columns(clean, drop_hot=drop_hot)
    ea_raw = edge_attr_matrix(clean, ea_cols)
    ea_std, ea_mean, ea_scale = standardize_edge_attr(ea_raw, clean["split"])

    # save CSVs
    nodes.to_csv(gdir / "nodes.csv", index=False)
    export_edges_csv(clean, gdir / "positive_edges.csv")
    for sp in ["train", "val", "test"]:
        export_edges_csv(clean[clean["split"].eq(sp)], gdir / f"{sp}_pos_edges.csv")

    # node feature matrix
    nf_cols = (["seq_log_length", "seq_gc"]
               + kmer_feature_names(args.kmer_sizes)
               + list(seed_features("").keys()))
    x_raw = nodes[nf_cols].to_numpy(dtype=np.float32)
    nt = nodes["node_type_id"].to_numpy(dtype=np.int64)
    sid = nodes["species_id"].to_numpy(dtype=np.int64)

    # interaction edges
    train_e = clean[clean["split"].eq("train")]
    ei = train_e[["src_idx", "dst_idx"]].to_numpy(dtype=np.int64).T
    ei_undir = np.concatenate([ei, ei[::-1]], axis=1) if ei.size else ei
    all_pos = clean[["src_idx", "dst_idx"]].to_numpy(dtype=np.int64).T

    # similarity edges
    sim_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    empty = np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float32)

    mir_sim = empty
    if args.mirna_sim_edges:
        print(f"  building mirna sim edges (topk={args.mirna_sim_topk})...", flush=True)
        mir_sim = build_mirna_similarity_edges(nodes, x_raw, args.kmer_sizes, args)
        if mir_sim[0].size:
            sim_parts.append(mir_sim)

    mrn_sim = empty
    if args.mrna_sim_edges:
        print(f"  building target sim edges (topk={args.mrna_sim_topk})...", flush=True)
        mrn_sim = build_target_similarity_edges(nodes, args)
        if mrn_sim[0].size:
            sim_parts.append(mrn_sim)

    aug_ei, aug_et, aug_ew = homogenize_edge_index(ei_undir, sim_parts)

    # hierarchical extras
    bt_edge_index = np.empty((2, 0), dtype=np.int64)
    bt_edge_type = np.empty((0,), dtype=np.int64)
    bt_edge_weight = np.empty((0,), dtype=np.float32)
    gene_sim_exp = empty

    if args.node_mode == "hierarchical":
        bt_edge_index, bt_edge_type, bt_edge_weight = build_belongs_to_edges(clean, ni)
        if bt_edge_index.size:
            aug_ei = np.concatenate([aug_ei, bt_edge_index], axis=1)
            aug_et = np.concatenate([aug_et, bt_edge_type])
            aug_ew = np.concatenate([aug_ew, bt_edge_weight])
        if args.mrna_sim_edges:
            gene_sim = build_mrna_gene_similarity_edges(nodes, args)
            if gene_sim[0].size:
                aug_ei = np.concatenate([aug_ei, gene_sim[0]], axis=1)
                aug_et = np.concatenate([aug_et, gene_sim[1]])
                aug_ew = np.concatenate([aug_ew, gene_sim[2]])
            gene_sim_exp = gene_sim

    # standardize node features
    train_idx = (np.unique(ei.reshape(-1)).astype(np.int64)
                 if ei.size else np.arange(len(nodes), dtype=np.int64))
    x, xm, xs = standardize_node_features(x_raw, train_idx)

    # assemble output
    arrays: dict = {
        "x": x, "x_raw": x_raw,
        "node_type": nt, "species_id": sid,
        "edge_index_train_pos": ei,
        "edge_index_train_pos_undirected": ei_undir,
        "all_positive_edge_index": all_pos,
        "node_ids": nodes["node_id"].astype(str).to_numpy(dtype=str),
        "node_sequences": nodes["sequence"].fillna("").astype(str).tolist(),
        "node_feature_names": np.asarray(nf_cols, dtype=str),
        "node_feature_mean": xm, "node_feature_scale": xs,
        "edge_attr_names": np.asarray(ea_cols, dtype=str),
        "edge_attr_mean": ea_mean, "edge_attr_scale": ea_scale,
        "augmented_edge_index": aug_ei,
        "augmented_edge_type": aug_et,
        "augmented_edge_weight": aug_ew,
        "similarity_edge_index_mirna": mir_sim[0],
        "similarity_edge_type_mirna": mir_sim[1],
        "similarity_edge_weight_mirna": mir_sim[2],
        "similarity_edge_index_mrna": mrn_sim[0],
        "similarity_edge_type_mrna": mrn_sim[1],
        "similarity_edge_weight_mrna": mrn_sim[2],
        "belongs_to_edge_index": bt_edge_index,
        "belongs_to_edge_type": bt_edge_type,
        "belongs_to_edge_weight": bt_edge_weight,
        "similarity_edge_index_mrna_gene": gene_sim_exp[0],
        "similarity_edge_type_mrna_gene": gene_sim_exp[1],
        "similarity_edge_weight_mrna_gene": gene_sim_exp[2],
    }
    for sp in ["train", "val", "test"]:
        m = clean["split"].eq(sp).to_numpy()
        sd = clean[m]
        arrays[f"{sp}_pos_edge_index"] = sd[["src_idx", "dst_idx"]].to_numpy(dtype=np.int64).T
        arrays[f"{sp}_pos_edge_attr_raw"] = ea_raw[m]
        arrays[f"{sp}_pos_edge_attr"] = ea_std[m]
        arrays[f"{sp}_pos_label"] = np.ones((int(m.sum()),), dtype=np.int64)

    # save npz
    np.savez_compressed(gdir / "graph_inputs.npz",
                        **{k: v for k, v in arrays.items() if k != "node_sequences"})

    # save pt
    torch_out = ""
    try:
        import torch
        tp = gdir / "graph_inputs.pt"
        tpt = gdir / "graph_inputs.pt.tmp"
        torch.save({k: v for k, v in arrays.items()}, tpt)
        tpt.replace(tp)
        torch_out = str(tp)
    except ModuleNotFoundError:
        pass

    # metadata
    num_mirna = int((nt == NODE_MIRNA).sum())
    num_ts = int((nt == NODE_TARGET_SITE).sum())
    num_gene = int((nt == NODE_MRNA_GENE).sum()) if args.node_mode == "hierarchical" else 0

    meta = {
        "version": "final_embedding",
        "species": species,
        "node_mode": args.node_mode,
        "sim_mode": args.sim_mode,
        "split_ratios": {"train": args.train_ratio, "val": args.val_ratio,
                         "test": args.test_ratio},
        "cleaning": report,
        "num_nodes": int(len(nodes)),
        "num_mirna_nodes": num_mirna,
        "num_target_site_nodes": num_ts,
        "num_mrna_gene_nodes": num_gene,
        "num_positive_edges": int(len(clean)),
        "split_counts": {s: int(clean["split"].eq(s).sum())
                         for s in ["train", "val", "test"]},
        "num_node_features": int(x.shape[1]),
        "num_edge_attr": int(ea_raw.shape[1]),
        "num_mirna_sim_edges": int(mir_sim[0].shape[1]),
        "num_mrna_sim_edges": int(mrn_sim[0].shape[1]),
        "num_belongs_to_edges": int(bt_edge_index.shape[1]),
        "num_mrna_gene_sim_edges": int(gene_sim_exp[0].shape[1]),
        "mirna_sim_topk": args.mirna_sim_topk if args.mirna_sim_edges else 0,
        "mrna_sim_topk": args.mrna_sim_topk if args.mrna_sim_edges else 0,
        "sim_threshold": args.sim_threshold if args.sim_mode == "threshold_topk" else None,
        "pair_feature_dim": PAIR_V3_DIM,
    }
    # split diagnostic
    diag = _split_diagnostic(clean, node_mode=args.node_mode)
    meta["split_diagnostic"] = diag
    print(f"  split_diag: (mirna_id,ts_id) overlap train/val={diag['pair_overlap_train_val']} "
          f"train/test={diag['pair_overlap_train_test']} val/test={diag['pair_overlap_val_test']}", flush=True)

    (gdir / "metadata.json").write_text(json.dumps(json_safe(meta), indent=2), encoding="utf-8")
    return meta


def _split_diagnostic(clean: pd.DataFrame, node_mode: str) -> dict:
    """Compute train/val/test overlap statistics."""
    pair_cols = (["mirna_id", "target_site_id"] if node_mode in ("target_site", "hierarchical")
                 else ["mirna_id", "mrna_id"])
    train_pairs = set(zip(clean.loc[clean["split"] == "train", pair_cols[0]],
                          clean.loc[clean["split"] == "train", pair_cols[1]]))
    val_pairs = set(zip(clean.loc[clean["split"] == "val", pair_cols[0]],
                        clean.loc[clean["split"] == "val", pair_cols[1]]))
    test_pairs = set(zip(clean.loc[clean["split"] == "test", pair_cols[0]],
                         clean.loc[clean["split"] == "test", pair_cols[1]]))

    # mRNA-level overlap (for target_site mode)
    mrna_pairs_train = set(zip(clean.loc[clean["split"] == "train", "mirna_id"],
                               clean.loc[clean["split"] == "train", "mrna_id"]))
    mrna_pairs_test = set(zip(clean.loc[clean["split"] == "test", "mirna_id"],
                              clean.loc[clean["split"] == "test", "mrna_id"]))
    mrna_overlap = mrna_pairs_train & mrna_pairs_test

    train_mask = clean["split"] == "train"
    test_mask = clean["split"] == "test"
    shared_mirna = len(set(clean.loc[train_mask, "mirna_id"])
                       & set(clean.loc[test_mask, "mirna_id"]))
    shared_target_site = len(set(clean.loc[train_mask, "target_site_id"])
                             & set(clean.loc[test_mask, "target_site_id"]))
    shared_mrna = len(set(clean.loc[train_mask, "mrna_id"])
                      & set(clean.loc[test_mask, "mrna_id"]))

    return {
        "pair_key": pair_cols,
        "num_train_pairs": len(train_pairs),
        "num_val_pairs": len(val_pairs),
        "num_test_pairs": len(test_pairs),
        "pair_overlap_train_val": len(train_pairs & val_pairs),
        "pair_overlap_train_test": len(train_pairs & test_pairs),
        "pair_overlap_val_test": len(val_pairs & test_pairs),
        "mrna_level_train_test_overlap": len(mrna_overlap),
        "mrna_level_overlap_ratio": round(len(mrna_overlap) / max(len(mrna_pairs_test), 1), 4),
        "shared_mirna_train_test": shared_mirna,
        "shared_target_site_train_test": shared_target_site,
        "shared_mrna_train_test": shared_mrna,
        "num_test_target_sites": clean.loc[test_mask, "target_site_id"].nunique(),
        "num_test_mrna": clean.loc[test_mask, "mrna_id"].nunique(),
    }


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    files = discover_pos_files(args.input_dir)
    raw_by: dict[str, list[pd.DataFrame]] = {s: [] for s in SPECIES_ORDER}
    for rf in files:
        if rf.species not in raw_by:
            raw_by[rf.species] = []
        raw_by[rf.species].append(read_positive_file(rf, args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"version": "final_embedding", "species": {}}

    for sp in SPECIES_ORDER:
        frames = raw_by.get(sp, [])
        if not frames:
            continue
        print(f"[{sp}] processing {len(frames)} CSV files...", flush=True)
        meta = export_species_graph(sp, pd.concat(frames, ignore_index=True, sort=False), args)
        summary["species"][sp] = meta

        extra = ""
        if args.node_mode == "hierarchical":
            extra = (f" mrna_gene={meta['num_mrna_gene_nodes']}"
                     f" belongs_to={meta['num_belongs_to_edges']}"
                     f" gene_sim={meta['num_mrna_gene_sim_edges']}")
        print(f"{sp}: mode={args.node_mode} sim={args.sim_mode} "
              f"nodes={meta['num_nodes']} (mirna={meta['num_mirna_nodes']} "
              f"ts={meta['num_target_site_nodes']}){extra} "
              f"pos={meta['num_positive_edges']} "
              f"train/val/test={meta['split_counts']} "
              f"mirna_sim={meta['num_mirna_sim_edges']} "
              f"mrna_sim={meta['num_mrna_sim_edges']}",
              flush=True)

    (args.output_dir / "preprocess_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"\nSaved -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
