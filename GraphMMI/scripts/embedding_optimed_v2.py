#!/usr/bin/env python3
"""Embedding-optimized graph preprocessing v2.

Key improvements over preprocess_graph_data_optimed.py:

1. Target-site-aware node representation (--node-mode):
   - mrna: Original mode, one node per mRNA (uses target_seq for features,
     same as --mrna-sequence-source target in the optimed script).
   - target_site: Replace mRNA nodes with per-target_site nodes. Each unique
     (mRNA_name, target_seq) pair gets its own node, preserving local
     binding-site information.  Minimum-implementation that avoids
     multi-site averaging.
   - hierarchical: Three-level graph — miRNA (type 0), target_site (type 1),
     mRNA (type 2).  Interaction edges go miRNA→target_site; belongs_to
     edges go target_site→mRNA.  Similarity edges exist at all three levels.

2. Cleaner similarity edges (--sim-mode):
   - topk: Current asymmetric top-k per node.
   - mutual: Only keep edges where A is in B's top-k AND B is in A's top-k.
   - threshold_topk: Require cosine > threshold, then retain top-k.

3. Differentiated edge-dropout metadata:
   Stores edge-type annotations so the training script can apply
   type-specific dropout (interaction 0.05–0.10, miRNA-sim 0.10–0.20,
   mRNA/target-site-sim 0.10–0.30).

4. Pair-feature v3 definitions:
   New biology-informed pair features exported for use by training scripts.

Usage:
  # Mode A — same as optimed, but with mutual top-k sim edges
  python scripts/embedding_optimed_v2.py --node-mode mrna --sim-mode mutual

  # Mode B — target-site nodes, minimum implementation
  python scripts/embedding_optimed_v2.py --node-mode target_site --sim-mode mutual

  # Mode C — full hierarchical graph
  python scripts/embedding_optimed_v2.py --node-mode hierarchical --sim-mode mutual
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

# Edge-type constants (stored as edge_type tensor)
ET_INTERACTION_FWD = 0   # miRNA → mRNA / miRNA → target_site
ET_INTERACTION_REV = 1   # reverse direction for undirected message passing
ET_MIRNA_SIM = 2         # miRNA–miRNA similarity
ET_MRNA_SIM = 3          # mRNA–mRNA / target_site–target_site similarity
ET_BELONGS_TO = 4        # target_site → mRNA (hierarchical mode only)

# Node-type constants
NODE_MIRNA = 0
NODE_MRNA = 1            # mRNA (mode A) or target_site (mode B)
NODE_TARGET_SITE = 1     # alias, same value
NODE_MRNA_GENE = 2       # mRNA gene-level node (mode C only)

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Embedding-optimized GraphMMI graph preprocessing v2")
    p.add_argument("--input-dir", type=Path, default=ROOT / "data/external")
    p.add_argument("--output-dir", type=Path,
                   default=ROOT / "data/processed/graph/embedding_optimed_v2")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.72)
    p.add_argument("--val-ratio", type=float, default=0.08)
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--kmer-sizes", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--drop-hot-pairing", action="store_true", default=True)
    p.add_argument("--keep-hot-pairing", action="store_true")
    # ---- node mode ----
    p.add_argument("--node-mode", choices=["mrna", "target_site", "hierarchical"],
                   default="target_site",
                   help="mrna: one node per mRNA (original). "
                        "target_site: one node per unique (mRNA, target_seq). "
                        "hierarchical: miRNA + target_site + mRNA gene nodes.")
    p.add_argument("--mrna-sequence-source", choices=["target", "full"],
                   default="target",
                   help="mRNA node sequence for mode=mrna. "
                        "Ignored in target_site/hierarchical modes (always target_seq).")
    # ---- similarity edges ----
    p.add_argument("--mirna-sim-edges", action="store_true", default=True,
                   help="Build miRNA-miRNA similarity edges.")
    p.add_argument("--mrna-sim-edges", action="store_true", default=True,
                   help="Build mRNA-mRNA / target_site-target_site similarity edges.")
    p.add_argument("--mirna-sim-topk", type=int, default=5)
    p.add_argument("--mrna-sim-topk", type=int, default=5)
    p.add_argument("--sim-mode", choices=["topk", "mutual", "threshold_topk"],
                   default="mutual",
                   help="topk: asymmetric top-k. "
                        "mutual: keep only reciprocal top-k edges. "
                        "threshold_topk: require cosine > threshold, then top-k.")
    p.add_argument("--sim-threshold", type=float, default=0.5,
                   help="Minimum cosine similarity for --sim-mode threshold_topk.")
    return p.parse_args()


# ------------------------------------------------------------------
# helpers (from optimed)
# ------------------------------------------------------------------

def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(i) for i in value]
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
    return str(value).strip()


def normalize_sequence(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", "", str(value).upper().replace("T", "U"))


def short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


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
        files.append(RawPosFile(
            path=path, species=m.group("species").lower(),
            dataset_id=m.group("dataset_id").lower()))
    if not files:
        raise FileNotFoundError(f"No *_pos.csv in {input_dir}")
    return files


@dataclass(frozen=True)
class RawPosFile:
    path: Path
    species: str
    dataset_id: str


# ------------------------------------------------------------------
# reading
# ------------------------------------------------------------------

def read_positive_file(raw_file: RawPosFile, args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(raw_file.path, index_col=0, low_memory=False)
    df = df.reset_index(drop=False).rename(columns={"index": "raw_index"})
    df["species"] = raw_file.species
    df["dataset_id"] = raw_file.dataset_id
    df["source_file"] = raw_file.path.name
    df["source_row"] = np.arange(len(df), dtype=np.int64)

    df["mirna_seq"] = df["miRNA sequence"].map(normalize_sequence)
    df["target_seq"] = df["target sequence"].map(normalize_sequence)
    df["full_mrna_seq"] = df["full_mrna"].map(normalize_sequence)

    # mRNA sequence for node embedding
    if args.node_mode == "mrna":
        if args.mrna_sequence_source == "target":
            df["mrna_seq"] = df["target_seq"]
        else:
            df["mrna_seq"] = np.where(
                df["full_mrna_seq"].astype(str).str.len() > 0,
                df["full_mrna_seq"], df["target_seq"])
    else:
        # target_site / hierarchical: always use target_seq
        df["mrna_seq"] = df["target_seq"]

    df["mirna_id"] = raw_file.species + "|" + df["microRNA_name"].map(clean_text)
    df["mrna_id"] = raw_file.species + "|" + df["mRNA_name"].map(clean_text)
    # target_site_id: unique per (species, mRNA_name, target_seq)
    df["target_site_id"] = (raw_file.species + "|"
                            + df["mRNA_name"].map(clean_text)
                            + "|" + df["target_seq"].map(short_hash))
    df["label"] = 1
    return df


# ------------------------------------------------------------------
# cleaning / split — mode-aware for target_site/hierarchical
# ------------------------------------------------------------------

def sequence_conflict_ids(frame: pd.DataFrame, id_col: str,
                          seq_col: str) -> set[str]:
    cnt = frame.groupby(id_col)[seq_col].nunique(dropna=False)
    return set(cnt[cnt.gt(1)].index.astype(str))


def clean_positive_edges(raw_edges: pd.DataFrame,
                         node_mode: str = "mrna") -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean edges with mode-aware dedup and conflict checking.

    In target_site / hierarchical modes:
      - Dedup key is (mirna_id, target_site_id) — preserves multiple
        target sites per mRNA instead of collapsing them to one.
      - Sequence conflicts are checked per target_site_id, not per
        mrna_id, because one mRNA naturally has multiple target_seq.
    """
    report = {"raw_rows": int(len(raw_edges)),
              "dropped_missing_core_fields": 0,
              "dropped_pair_sequence_conflicts": 0,
              "dropped_node_sequence_conflicts": 0,
              "duplicate_pair_rows_removed": 0}
    core = (raw_edges["mirna_id"].astype(str).ne("")
            & raw_edges["mrna_id"].astype(str).ne("")
            & raw_edges["mirna_seq"].astype(str).ne("")
            & raw_edges["mrna_seq"].astype(str).ne(""))
    report["dropped_missing_core_fields"] = int((~core).sum())
    edges = raw_edges[core].copy()

    # ---- pair-level sequence conflict check ----
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

    # ---- dedup — mode-aware key ----
    before = len(edges)
    edges = (edges.sort_values(["dataset_id", "source_row"])
             .drop_duplicates(pair_key, keep="first")
             .reset_index(drop=True))
    report["duplicate_pair_rows_removed"] = int(before - len(edges))

    # ---- node-level sequence conflict check ----
    mc = sequence_conflict_ids(edges, "mirna_id", "mirna_seq")
    if node_mode in ("target_site", "hierarchical"):
        # target_site_id → target_seq must be unique (one node per site)
        ts_conflicts = sequence_conflict_ids(edges, "target_site_id", "target_seq")
        conflict_mask = edges["mirna_id"].isin(mc) | edges["target_site_id"].isin(ts_conflicts)
    else:
        nc = sequence_conflict_ids(edges, "mrna_id", "mrna_seq")
        conflict_mask = edges["mirna_id"].isin(mc) | edges["mrna_id"].isin(nc)
    if conflict_mask.any():
        report["dropped_node_sequence_conflicts"] = int(conflict_mask.sum())
        edges = edges[~conflict_mask].reset_index(drop=True)

    report["clean_positive_edges"] = int(len(edges))
    return edges, report


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
# node building  — with Mode A / B / C support
# ------------------------------------------------------------------

def build_nodes_mode_mrna(edges: pd.DataFrame, kmer_sizes: list[int],
                          ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mode A: miRNA nodes + mRNA nodes (one per mRNA, longest target_seq)."""
    mirna_cols = ["mirna_id", "species", "microRNA_name", "mirna_seq",
                  "dataset_id", "Source", "Organism"]
    mrna_cols = ["mrna_id", "species", "mRNA_name", "mrna_seq", "target_seq",
                 "full_mrna_seq", "dataset_id", "Source", "Organism", "GI_ID"]

    mirna_nodes = (edges.reindex(columns=mirna_cols)
                   .drop_duplicates("mirna_id", keep="first")
                   .rename(columns={"mirna_id": "node_id",
                                    "microRNA_name": "source_name",
                                    "mirna_seq": "sequence"}))
    mirna_nodes["node_type"] = "mirna"
    mirna_nodes["node_type_id"] = NODE_MIRNA

    mrna_tmp = edges.reindex(columns=mrna_cols).copy()
    mrna_tmp["_target_len"] = mrna_tmp["target_seq"].fillna("").astype(str).str.len()
    mrna_tmp = mrna_tmp.sort_values("_target_len", ascending=False)
    mrna_nodes = (mrna_tmp.drop_duplicates("mrna_id", keep="first")
                  .rename(columns={"mrna_id": "node_id",
                                   "mRNA_name": "source_name",
                                   "mrna_seq": "sequence"}))
    mrna_nodes["node_type"] = "mrna"
    mrna_nodes["node_type_id"] = NODE_MRNA

    nodes = pd.concat([mirna_nodes, mrna_nodes], ignore_index=True, sort=False)
    nodes = _finalize_nodes(nodes, kmer_sizes)
    node_index = dict(zip(nodes["node_id"].astype(str),
                          nodes["node_idx"].astype(int)))
    return nodes, node_index


def build_nodes_mode_target_site(edges: pd.DataFrame, kmer_sizes: list[int],
                                 ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mode B: miRNA nodes + target_site nodes (one per unique target_seq per mRNA).

    Replaces mRNA nodes with target-site nodes so the model learns
    miRNA↔specific-binding-site relationships instead of averaging
    multiple sites into one mRNA embedding.
    """
    mirna_cols = ["mirna_id", "species", "microRNA_name", "mirna_seq",
                  "dataset_id", "Source", "Organism"]
    ts_cols = ["target_site_id", "species", "mRNA_name", "mrna_seq",
               "target_seq", "full_mrna_seq", "mrna_id",
               "dataset_id", "Source", "Organism", "GI_ID"]

    mirna_nodes = (edges.reindex(columns=mirna_cols)
                   .drop_duplicates("mirna_id", keep="first")
                   .rename(columns={"mirna_id": "node_id",
                                    "microRNA_name": "source_name",
                                    "mirna_seq": "sequence"}))
    mirna_nodes["node_type"] = "mirna"
    mirna_nodes["node_type_id"] = NODE_MIRNA

    # Each unique (mRNA_name, target_seq) → one target_site node
    ts_tmp = edges.reindex(columns=ts_cols).copy()
    ts_nodes = (ts_tmp.drop_duplicates("target_site_id", keep="first")
                .rename(columns={"target_site_id": "node_id",
                                 "mRNA_name": "source_name",
                                 "mrna_seq": "sequence"}))
    ts_nodes["node_type"] = "target_site"
    ts_nodes["node_type_id"] = NODE_TARGET_SITE

    nodes = pd.concat([mirna_nodes, ts_nodes], ignore_index=True, sort=False)
    nodes = _finalize_nodes(nodes, kmer_sizes)
    node_index = dict(zip(nodes["node_id"].astype(str),
                          nodes["node_idx"].astype(int)))
    return nodes, node_index


def build_nodes_mode_hierarchical(edges: pd.DataFrame, kmer_sizes: list[int],
                                  ) -> tuple[pd.DataFrame, dict[str, int]]:
    """Mode C: miRNA (type 0) + target_site (type 1) + mRNA gene (type 2).

    Three-level graph:
      miRNA ──interaction──> target_site ──belongs_to──> mRNA
    """
    mirna_cols = ["mirna_id", "species", "microRNA_name", "mirna_seq",
                  "dataset_id", "Source", "Organism"]
    ts_cols = ["target_site_id", "species", "mRNA_name", "mrna_seq",
               "target_seq", "full_mrna_seq", "mrna_id",
               "dataset_id", "Source", "Organism", "GI_ID"]
    mrna_cols = ["mrna_id", "species", "mRNA_name",
                 "dataset_id", "Source", "Organism", "GI_ID"]

    # miRNA
    mirna_nodes = (edges.reindex(columns=mirna_cols)
                   .drop_duplicates("mirna_id", keep="first")
                   .rename(columns={"mirna_id": "node_id",
                                    "microRNA_name": "source_name",
                                    "mirna_seq": "sequence"}))
    mirna_nodes["node_type"] = "mirna"
    mirna_nodes["node_type_id"] = NODE_MIRNA

    # target_site
    ts_tmp = edges.reindex(columns=ts_cols).copy()
    ts_nodes = (ts_tmp.drop_duplicates("target_site_id", keep="first")
                .rename(columns={"target_site_id": "node_id",
                                 "mRNA_name": "source_name",
                                 "mrna_seq": "sequence"}))
    ts_nodes["node_type"] = "target_site"
    ts_nodes["node_type_id"] = NODE_TARGET_SITE

    # mRNA gene — needs a placeholder sequence for feature extraction
    mrna_tmp = edges.reindex(columns=mrna_cols).copy()
    mrna_tmp["_target_len"] = edges["target_seq"].fillna("").astype(str).str.len()
    mrna_tmp = mrna_tmp.sort_values("_target_len", ascending=False)
    mrna_nodes_df = (mrna_tmp.drop_duplicates("mrna_id", keep="first")
                     .rename(columns={"mrna_id": "node_id",
                                      "mRNA_name": "source_name"}))
    # For mRNA gene nodes, use full_mrna_seq if available, else concat of target_seqs
    # We'll look up the full_mrna_seq from edges
    full_seqs = (edges.groupby("mrna_id")["full_mrna_seq"]
                 .apply(lambda x: x.iloc[0] if len(x) > 0 else "")
                 .to_dict())
    mrna_nodes_df["sequence"] = mrna_nodes_df["node_id"].map(
        lambda nid: str(full_seqs.get(nid, "")))
    mrna_nodes_df["node_type"] = "mrna_gene"
    mrna_nodes_df["node_type_id"] = NODE_MRNA_GENE

    nodes = pd.concat([mirna_nodes, ts_nodes, mrna_nodes_df],
                      ignore_index=True, sort=False)
    nodes = _finalize_nodes(nodes, kmer_sizes)
    node_index = dict(zip(nodes["node_id"].astype(str),
                          nodes["node_idx"].astype(int)))
    return nodes, node_index


def _finalize_nodes(nodes: pd.DataFrame, kmer_sizes: list[int]) -> pd.DataFrame:
    """Common post-processing for all node modes."""
    nodes.insert(0, "node_idx", np.arange(len(nodes), dtype=np.int64))
    sp2id = {s: i for i, s in enumerate(SPECIES_ORDER)}
    nodes["species_id"] = nodes["species"].map(sp2id).fillna(-1).astype("int64")
    nodes["sequence_hash"] = (nodes["sequence"].fillna("").astype(str)
                              .map(short_hash))
    nodes["mirna_len"] = np.where(nodes["node_type"].eq("mirna"),
                                  nodes["sequence"].astype(str).str.len(),
                                  np.nan)
    nodes["mrna_len"] = np.where(nodes["node_type"].isin(["mrna", "target_site", "mrna_gene"]),
                                 nodes["sequence"].astype(str).str.len(),
                                 np.nan)

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
    nodes["mrna_gc"] = np.where(nodes["node_type"].isin(["mrna", "target_site", "mrna_gene"]),
                                nodes["seq_gc"], np.nan)
    return nodes


# ------------------------------------------------------------------
# edge attr (unchanged from optimed)
# ------------------------------------------------------------------

def bools_to_numeric(series: pd.Series) -> pd.Series:
    def convert(value):
        try:
            return BOOL_TO_INT.get(value, value)
        except TypeError:
            return value
    return series.map(convert)


def infer_edge_attr_columns(edges: pd.DataFrame, drop_hot: bool) -> list[str]:
    forbidden = METADATA_COLUMNS | {
        "raw_index", "species", "dataset_id", "source_file", "source_row",
        "mirna_id", "mrna_id", "target_site_id", "mirna_seq", "target_seq",
        "full_mrna_seq", "mrna_seq", "label", "split", "src_idx", "dst_idx"}
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
    frame = edges.reindex(columns=columns)
    return (frame.apply(bools_to_numeric)
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


# ======================================================================
# similarity edges  — topk / mutual / threshold_topk
# ======================================================================

def _make_undirected(ei: np.ndarray, ew: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add reverse direction so similarity edges are truly undirected."""
    if ei.size == 0:
        return ei, ew
    rev = ei[::-1]
    return np.concatenate([ei, rev], axis=1), np.concatenate([ew, ew], axis=0)


def _cosine_similarity_topk(vecs: np.ndarray, topk: int,
                            rng: np.random.Generator,
                            sim_mode: str = "topk",
                            sim_threshold: float = 0.5,
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Cosine similarity top-k with optional mutual / threshold filtering.

    Returns (edge_index (2, E), edge_weight (E,)).
    """
    n = vecs.shape[0]
    if n < 2:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    unit = vecs.astype(np.float64) / norms
    eff_k = min(topk, n - 1)

    # --- compute all top-k edges first ---
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for i in range(n):
        sim = unit[i] @ unit.T
        sim[i] = -2.0  # exclude self
        if sim_mode == "threshold_topk":
            # mask out below-threshold before top-k
            mask_val = sim_threshold - 1.0
            sim[sim < sim_threshold] = mask_val
        top = (np.argpartition(-sim, eff_k)[:eff_k]
               if eff_k < n else np.argsort(-sim)[:eff_k])
        for j in top:
            s = float(sim[j])
            if s > 0.0:
                edges.append((i, j))
                weights.append(s)

    if not edges:
        return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

    ei = np.array(edges, dtype=np.int64)  # (E, 2)
    ew = np.array(weights, dtype=np.float32)

    # --- mutual filtering ---
    if sim_mode == "mutual":
        edge_set = {(int(e[0]), int(e[1])) for e in ei}
        mutual_mask = np.array([
            (int(e[1]), int(e[0])) in edge_set
            for e in ei], dtype=bool)
        ei = ei[mutual_mask]
        ew = ew[mutual_mask]
        if ei.size == 0:
            return np.empty((2, 0), dtype=np.int64), np.empty((0,), dtype=np.float32)

    return ei.T, ew  # (2, E), (E,)


def build_mirna_similarity_edges(nodes: pd.DataFrame, x_raw: np.ndarray,
                                 kmer_sizes: list[int],
                                 args: argparse.Namespace,
                                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """miRNA-miRNA similarity: seed identity + k-mer cosine top-k."""
    rng = np.random.default_rng(args.seed)
    mirna = nodes[nodes["node_type"].eq("mirna")].copy()
    if len(mirna) < 2:
        return (np.empty((2, 0), dtype=np.int64),
                np.full((0,), ET_MIRNA_SIM, dtype=np.int64),
                np.empty((0,), dtype=np.float32))
    l2g = {i: int(mirna.iloc[i].node_idx) for i in range(len(mirna))}
    seqs = mirna["sequence"].fillna("").astype(str).tolist()
    edge_map: dict[tuple[int, int], float] = {}

    # seed-identity edges (weight = 1.0)
    for _name, slc in [("seed_2_7", slice(1, 7)), ("seed_3_8", slice(2, 8))]:
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

    # k-mer cosine similarity edges
    kmer_cols = [c for c in nodes.columns
                 if c.startswith("kmer1_") or c.startswith("kmer2_")
                 or c.startswith("kmer3_")]
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
        return (np.empty((2, 0), dtype=np.int64),
                np.full((0,), ET_MIRNA_SIM, dtype=np.int64),
                np.empty((0,), dtype=np.float32))

    pairs = sorted(edge_map.keys())
    ei = np.array(pairs, dtype=np.int64).T
    ew = np.array([edge_map[k] for k in pairs], dtype=np.float32)
    ei, ew = _make_undirected(ei, ew)
    et = np.full(ei.shape[1], ET_MIRNA_SIM, dtype=np.int64)
    return ei, et, ew


def build_target_similarity_edges(nodes: pd.DataFrame, args: argparse.Namespace,
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build similarity edges for mRNA / target_site nodes based on target_seq k-mer."""
    rng = np.random.default_rng(args.seed)
    # Works for both "mrna" and "target_site" node types
    targets = nodes[nodes["node_type"].isin(["mrna", "target_site"])].copy()
    if len(targets) < 2:
        return (np.empty((2, 0), dtype=np.int64),
                np.full((0,), ET_MRNA_SIM, dtype=np.int64),
                np.empty((0,), dtype=np.float32))
    l2g = {i: int(targets.iloc[i].node_idx) for i in range(len(targets))}

    # Use target_seq from nodes.csv if it exists, else fall back to sequence column
    if "target_seq" in nodes.columns:
        target_seqs = targets.get("target_seq", targets["sequence"].fillna(""))
    else:
        target_seqs = targets["sequence"].fillna("")
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
        return (np.empty((2, 0), dtype=np.int64),
                np.full((0,), ET_MRNA_SIM, dtype=np.int64),
                np.empty((0,), dtype=np.float32))

    # Deduplicate
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
        return (np.empty((2, 0), dtype=np.int64),
                np.full((0,), ET_MRNA_SIM, dtype=np.int64),
                np.empty((0,), dtype=np.float32))

    ei = np.array(pairs, dtype=np.int64).T
    ew = np.array(weights, dtype=np.float32)
    ei, ew = _make_undirected(ei, ew)
    et = np.full(ei.shape[1], ET_MRNA_SIM, dtype=np.int64)
    return ei, et, ew


def build_mrna_gene_similarity_edges(nodes: pd.DataFrame,
                                     args: argparse.Namespace,
                                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mRNA-gene-level similarity (hierarchical mode only)."""
    rng = np.random.default_rng(args.seed)
    genes = nodes[nodes["node_type"].eq("mrna_gene")].copy()
    if len(genes) < 2:
        return (np.empty((2, 0), dtype=np.int64),
                np.full((0,), ET_MRNA_SIM, dtype=np.int64),
                np.empty((0,), dtype=np.float32))
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
        return (np.empty((2, 0), dtype=np.int64),
                np.full((0,), ET_MRNA_SIM, dtype=np.int64),
                np.empty((0,), dtype=np.float32))

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
        return (np.empty((2, 0), dtype=np.int64),
                np.full((0,), ET_MRNA_SIM, dtype=np.int64),
                np.empty((0,), dtype=np.float32))

    ei = np.array(pairs, dtype=np.int64).T
    ew = np.array(weights, dtype=np.float32)
    ei, ew = _make_undirected(ei, ew)
    et = np.full(ei.shape[1], ET_MRNA_SIM, dtype=np.int64)
    return ei, et, ew


def build_belongs_to_edges(edges: pd.DataFrame,
                           node_index: dict[str, int],
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build target_site → mRNA belongs_to edges (hierarchical mode)."""
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
        return (np.empty((2, 0), dtype=np.int64),
                np.full((0,), ET_BELONGS_TO, dtype=np.int64),
                np.empty((0,), dtype=np.float32))
    ei = np.array(pairs, dtype=np.int64).T
    # Undirected for message passing (target_site ↔ mRNA)
    ei_undir = np.concatenate([ei, ei[::-1]], axis=1)
    et = np.full(ei_undir.shape[1], ET_BELONGS_TO, dtype=np.int64)
    ew = np.ones(ei_undir.shape[1], dtype=np.float32)
    return ei_undir, et, ew


def homogenize_edge_index(interaction_undirected: np.ndarray,
                          sim_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate interaction + similarity edges into one homogeneous graph.

    interaction_undirected: (2, 2*E_interaction) — forward + reverse
    sim_parts: list of (ei, et, ew) for each similarity type

    Returns (edge_index, edge_type, edge_weight).
    """
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


# ======================================================================
# pair features v3 — richer biology-informed features
# ======================================================================

_COMPLEMENT_TABLE = str.maketrans({"A": "U", "U": "A", "C": "G", "G": "C", "T": "A"})

# Pair feature names (v1 baseline + v2 additions + v3 additions)
PAIR_V3_EXTRA_NAMES = [
    # v2 extras
    "pair_longest_complement",
    "pair_total_complement",
    "pair_complement_density",
    "pair_mono_cosine",
    "pair_seed_2_7_au",
    "pair_seed_3_8_au",
    "pair_mirna_log_len_v2",
    "pair_mrna_log_len_v2",
    "pair_log_len_ratio_v2",
    "pair_seed_2_7_first_pos",
    "pair_seed_3_8_first_pos",
    # v3 extras — detailed binding-site match (12 dims)
    "pair_gu_wobble_count",
    "pair_total_mismatch_count",
    "pair_seed_27_mismatch",
    "pair_seed_38_mismatch",
    "pair_seed_exact_any",
    "pair_target_local_gc",
    "pair_kmer_overlap_3mer",
    "pair_seed_2_8_sliding_match",
    "pair_target_seq_len",
    "pair_target_seq_log_len",
    "pair_seed_27_gu_wobble",
    "pair_seed_38_gu_wobble",
]

# v1 (17) + v2 (11) + v3 (12) = 40 dims
PAIR_V3_DIM = 17 + 11 + 12


def pair_feature_dim_v3() -> int:
    return PAIR_V3_DIM


def _reverse_complement(seq: str) -> str:
    return str(seq).upper().replace("T", "U").translate(_COMPLEMENT_TABLE)[::-1]


def _gc_fraction(seq: str) -> float:
    valid = [b for b in str(seq).upper().replace("T", "U") if b in {"A", "C", "G", "U"}]
    if not valid:
        return 0.0
    return float((valid.count("G") + valid.count("C")) / len(valid))


def _normalized_substring_count(pattern: str, text: str) -> float:
    if not pattern or not text or len(text) < len(pattern):
        return 0.0
    count = 0
    start = 0
    while True:
        idx = text.find(pattern, start)
        if idx < 0:
            break
        count += 1
        start = idx + 1
    return float(count / max(len(text) - len(pattern) + 1, 1))


def _longest_complement_v3(rc: str, mrna: str) -> int:
    """Longest contiguous exact complement between miRNA rc and mRNA."""
    if not rc or not mrna:
        return 0
    mrna = mrna[:2000]
    for w in range(len(rc), 0, -1):
        for i in range(len(rc) - w + 1):
            if rc[i:i + w] in mrna:
                return w
    return 0


def _total_complement_v3(rc: str, mrna: str) -> int:
    """Count miRNA positions whose complement base exists in mRNA."""
    comp = {"A": "U", "U": "A", "C": "G", "G": "C"}
    mrna_set = set(mrna[:2000])
    return sum(1 for b in rc if b in comp and comp[b] in mrna_set)


def _mono_cosine(seq1: str, seq2: str) -> float:
    bases = ["A", "C", "G", "U"]
    v1 = np.array([seq1.count(b) / max(len(seq1), 1) for b in bases], dtype=np.float64)
    v2 = np.array([seq2.count(b) / max(len(seq2), 1) for b in bases], dtype=np.float64)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def _first_seed_position(seed_rc: str, mrna: str) -> float:
    if not seed_rc or not mrna:
        return -1.0
    idx = mrna.find(seed_rc)
    if idx < 0:
        return -1.0
    return float(idx / max(len(mrna) - len(seed_rc), 1))


def _count_gu_wobble(mirna_seq: str, mrna_seq: str) -> int:
    """Count G-U wobble pairs when aligning miRNA (rc) to mRNA."""
    rc = _reverse_complement(mirna_seq)
    min_len = min(len(rc), len(mrna_seq))
    wobble = 0
    for i in range(min_len):
        b_rc = rc[i]
        b_mrna = mrna_seq[i]
        if (b_rc == "G" and b_mrna == "U") or (b_rc == "U" and b_mrna == "G"):
            wobble += 1
    return wobble


def _count_mismatch(mirna_seq: str, mrna_seq: str) -> int:
    """Count mismatches (non-Watson-Crick, non-GU) in aligned region.

    rc = reverse_complement(miRNA).  For perfect WC pairing, rc[i] == mrna[i]
    (the reverse complement *is* the expected mRNA binding sequence).
    GU wobble = (rc G, mRNA U) or (rc U, mRNA G).
    """
    rc = _reverse_complement(mirna_seq)
    min_len = min(len(rc), len(mrna_seq))
    mism = 0
    for i in range(min_len):
        b_rc = rc[i]
        b_mrna = mrna_seq[i]
        is_wc = (b_rc == b_mrna)
        is_gu = (b_rc == "G" and b_mrna == "U") or (b_rc == "U" and b_mrna == "G")
        if not is_wc and not is_gu:
            mism += 1
    return mism


def _seed_mismatch(mirna_seq: str, mrna_seq: str, seed_slice: slice) -> int:
    """Count mismatches in a seed region, aligned at its best-match position.

    Finds where the seed reverse-complement first matches in the mRNA,
    then counts mismatches at that offset.  If no match is found,
    falls back to position 0 (seed aligned to mRNA start).
    """
    rc = _reverse_complement(mirna_seq)
    seed_rc = rc[seed_slice] if len(rc) >= seed_slice.stop else ""
    if not seed_rc:
        return 0
    # Find best alignment offset for this seed in mRNA
    offset = mrna_seq.find(seed_rc)
    if offset < 0:
        offset = 0
    mism = 0
    for i, b in enumerate(seed_rc):
        mrna_pos = offset + i
        if mrna_pos >= len(mrna_seq):
            mism += 1
            continue
        b_mrna = mrna_seq[mrna_pos]
        is_wc = (b == b_mrna)
        is_gu = (b == "G" and b_mrna == "U") or (b == "U" and b_mrna == "G")
        if not is_wc and not is_gu:
            mism += 1
    return mism


def _local_gc(seq: str, window: int = 20) -> float:
    """Maximum local GC content in sliding windows."""
    if len(seq) < window:
        return _gc_fraction(seq)
    max_gc = 0.0
    for i in range(len(seq) - window + 1):
        gc = _gc_fraction(seq[i:i + window])
        if gc > max_gc:
            max_gc = gc
    return max_gc


def _kmer_overlap_3mer(mirna_seq: str, mrna_seq: str) -> float:
    """Jaccard overlap of 3-mer sets between miRNA rc and mRNA."""
    rc = _reverse_complement(mirna_seq)
    if len(rc) < 3 or len(mrna_seq) < 3:
        return 0.0

    def kmer_set(seq, k=3):
        return {seq[i:i + k] for i in range(len(seq) - k + 1)
                if all(b in BASES for b in seq[i:i + k])}

    rc_set = kmer_set(rc)
    mrna_set = kmer_set(mrna_seq)
    union = len(rc_set | mrna_set)
    if union == 0:
        return 0.0
    return float(len(rc_set & mrna_set) / union)


def _seed_2_8_sliding_match(mirna_seq: str, mrna_seq: str,
                            window: int = 7) -> float:
    """Max fraction of seed_2_8 rc bases matching in sliding windows of mRNA.

    Slides a window of length `window` across the mRNA and computes the
    fraction of seed_2_8 reverse-complement bases that match exactly.
    """
    rc = _reverse_complement(mirna_seq)
    seed_rc = rc[1:8] if len(rc) >= 8 else rc  # seed 2-8 region of rc
    if len(seed_rc) < window:
        return 0.0
    max_match = 0.0
    for start in range(max(len(mrna_seq) - window + 1, 1)):
        window_mrna = mrna_seq[start:start + window]
        matches = 0
        for i in range(min(window, len(seed_rc))):
            if i < len(window_mrna) and seed_rc[i] == window_mrna[i]:
                matches += 1
        match_frac = matches / window
        if match_frac > max_match:
            max_match = match_frac
    return max_match


def pair_feature_row_v3(src: int, dst: int, sequences: np.ndarray) -> np.ndarray:
    """Compute full v3 pair features for a miRNA–target_site/mRNA pair.

    Returns np.ndarray of shape (PAIR_V3_DIM,), dtype float32.
    Includes all v1 (17) + v2 (11) + v3 (12) features = 40 dims.
    """
    mirna_seq = str(sequences[int(src)]).upper().replace("T", "U")
    mrna_seq = str(sequences[int(dst)]).upper().replace("T", "U")
    rc = _reverse_complement(mirna_seq)
    mirna_len = max(len(mirna_seq), 1)
    mrna_len = max(len(mrna_seq), 1)

    # ---- v1 features (17 dims) ----
    mirna_log_len = float(np.log1p(mirna_len))
    mrna_log_len = float(np.log1p(mrna_len))
    mirna_gc = _gc_fraction(mirna_seq)
    mrna_gc = _gc_fraction(mrna_seq)

    seed_2_7 = _reverse_complement(mirna_seq[1:7])
    seed_3_8 = _reverse_complement(mirna_seq[2:8])
    seed_2_8 = _reverse_complement(mirna_seq[1:8])
    seed_patterns = [seed_2_7, seed_3_8, seed_2_8]
    counts = [_normalized_substring_count(seed, mrna_seq) for seed in seed_patterns]
    exact = [1.0 if count > 0.0 else 0.0 for count in counts]
    seed_gc_vals = [_gc_fraction(s) for s in [mirna_seq[1:7], mirna_seq[2:8], mirna_seq[1:8]]]

    v1 = np.asarray([
        mirna_log_len,
        mrna_log_len,
        mirna_log_len / max(mrna_log_len, 1e-6),
        abs(mirna_log_len - mrna_log_len),
        mirna_gc,
        mrna_gc,
        abs(mirna_gc - mrna_gc),
        mirna_gc * mrna_gc,
        *exact,
        *counts,
        *seed_gc_vals,
    ], dtype=np.float32)

    # ---- v2 extras (11 dims) ----
    longest = float(_longest_complement_v3(rc, mrna_seq))
    total = float(_total_complement_v3(rc, mrna_seq))
    density = total / float(mirna_len)
    mono_cos = _mono_cosine(mirna_seq, mrna_seq)
    au_27 = float(seed_2_7.count("A") + seed_2_7.count("U")) / max(len(seed_2_7), 1)
    au_38 = float(seed_3_8.count("A") + seed_3_8.count("U")) / max(len(seed_3_8), 1)
    pos_27 = _first_seed_position(seed_2_7, mrna_seq)
    pos_38 = _first_seed_position(seed_3_8, mrna_seq)

    v2 = np.asarray([
        longest, total, density, mono_cos,
        au_27, au_38,
        np.log1p(mirna_len), np.log1p(mrna_len),
        np.log1p(mirna_len) / max(np.log1p(mrna_len), 1e-6),
        pos_27, pos_38,
    ], dtype=np.float32)

    # ---- v3 extras (12 dims) ----
    gu_wobble = float(_count_gu_wobble(mirna_seq, mrna_seq))
    total_mismatch = float(_count_mismatch(mirna_seq, mrna_seq))
    seed_27_mismatch = float(_seed_mismatch(mirna_seq, mrna_seq, slice(1, 7)))
    seed_38_mismatch = float(_seed_mismatch(mirna_seq, mrna_seq, slice(2, 8)))
    seed_exact_any = float(max(exact))  # any seed has exact match (0 or 1)
    target_local_gc = float(_local_gc(mrna_seq))
    kmer_overlap = float(_kmer_overlap_3mer(mirna_seq, mrna_seq))
    seed_28_sliding = float(_seed_2_8_sliding_match(mirna_seq, mrna_seq))
    target_seq_len = float(mrna_len)
    target_seq_log_len = float(np.log1p(mrna_len))
    seed_27_gu = float(_count_gu_wobble(mirna_seq[1:7], mrna_seq[:6]))
    seed_38_gu = float(_count_gu_wobble(mirna_seq[2:8], mrna_seq[:6]))

    v3 = np.asarray([
        gu_wobble, total_mismatch,
        seed_27_mismatch, seed_38_mismatch,
        seed_exact_any,
        target_local_gc, kmer_overlap, seed_28_sliding,
        target_seq_len, target_seq_log_len,
        seed_27_gu, seed_38_gu,
    ], dtype=np.float32)

    return np.concatenate([v1, v2, v3])


# ======================================================================
# main export
# ======================================================================

def export_species_graph(species: str, edges: pd.DataFrame,
                         args: argparse.Namespace) -> dict:
    gdir = args.output_dir / species
    gdir.mkdir(parents=True, exist_ok=True)
    sseed = args.seed + sum(ord(c) for c in species)
    clean, report = clean_positive_edges(edges, node_mode=args.node_mode)
    clean["split"] = split_edges(clean, sseed,
                                 args.train_ratio, args.val_ratio, args.test_ratio)

    # ---- Build nodes ----
    if args.node_mode == "mrna":
        nodes, ni = build_nodes_mode_mrna(clean, args.kmer_sizes)
    elif args.node_mode == "target_site":
        nodes, ni = build_nodes_mode_target_site(clean, args.kmer_sizes)
    elif args.node_mode == "hierarchical":
        nodes, ni = build_nodes_mode_hierarchical(clean, args.kmer_sizes)
    else:
        raise ValueError(f"Unknown node_mode: {args.node_mode}")

    # ---- Map edge endpoints ----
    # For target_site and hierarchical modes, map to target_site_id
    if args.node_mode in ("target_site", "hierarchical"):
        clean["src_idx"] = clean["mirna_id"].map(ni).astype("int64")
        clean["dst_idx"] = clean["target_site_id"].map(ni).astype("int64")
    else:
        clean["src_idx"] = clean["mirna_id"].map(ni).astype("int64")
        clean["dst_idx"] = clean["mrna_id"].map(ni).astype("int64")

    # ---- Edge attributes ----
    drop_hot = args.drop_hot_pairing and not args.keep_hot_pairing
    ea_cols = infer_edge_attr_columns(clean, drop_hot=drop_hot)
    ea_raw = edge_attr_matrix(clean, ea_cols)
    ea_std, ea_mean, ea_scale = standardize_edge_attr(ea_raw, clean["split"])

    # ---- Save CSVs ----
    nodes_path = gdir / "nodes.csv"
    pos_path = gdir / "positive_edges.csv"
    nodes.to_csv(nodes_path, index=False)
    export_edges_csv(clean, pos_path)
    for sp in ["train", "val", "test"]:
        export_edges_csv(clean[clean["split"].eq(sp)],
                         gdir / f"{sp}_pos_edges.csv")

    # ---- Node feature matrix ----
    nf_cols = (["seq_log_length", "seq_gc"]
               + kmer_feature_names(args.kmer_sizes)
               + list(seed_features("").keys()))
    x_raw = nodes[nf_cols].to_numpy(dtype=np.float32)
    nt = nodes["node_type_id"].to_numpy(dtype=np.int64)
    sid = nodes["species_id"].to_numpy(dtype=np.int64)

    # ---- Interaction edges ----
    train_e = clean[clean["split"].eq("train")]
    ei = train_e[["src_idx", "dst_idx"]].to_numpy(dtype=np.int64).T
    ei_undir = np.concatenate([ei, ei[::-1]], axis=1) if ei.size else ei
    all_pos = clean[["src_idx", "dst_idx"]].to_numpy(dtype=np.int64).T

    # ---- Similarity edges ----
    sim_parts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    if args.mirna_sim_edges:
        mir_sim = build_mirna_similarity_edges(nodes, x_raw, args.kmer_sizes, args)
        if mir_sim[0].size:
            sim_parts.append(mir_sim)
    else:
        mir_sim = (np.empty((2, 0), dtype=np.int64),
                   np.empty((0,), dtype=np.int64),
                   np.empty((0,), dtype=np.float32))

    if args.mrna_sim_edges:
        mrn_sim = build_target_similarity_edges(nodes, args)
        if mrn_sim[0].size:
            sim_parts.append(mrn_sim)
    else:
        mrn_sim = (np.empty((2, 0), dtype=np.int64),
                   np.empty((0,), dtype=np.int64),
                   np.empty((0,), dtype=np.float32))

    # Build augmented graph
    aug_ei, aug_et, aug_ew = homogenize_edge_index(ei_undir, sim_parts)

    # ---- Belongs-to edges (hierarchical only) ----
    if args.node_mode == "hierarchical":
        bt_edge_index, bt_edge_type, bt_edge_weight = build_belongs_to_edges(clean, ni)
        if bt_edge_index.size:
            aug_ei = np.concatenate([aug_ei, bt_edge_index], axis=1)
            aug_et = np.concatenate([aug_et, bt_edge_type])
            aug_ew = np.concatenate([aug_ew, bt_edge_weight])
        # mRNA gene-level similarity
        if args.mrna_sim_edges:
            gene_sim = build_mrna_gene_similarity_edges(nodes, args)
            if gene_sim[0].size:
                aug_ei = np.concatenate([aug_ei, gene_sim[0]], axis=1)
                aug_et = np.concatenate([aug_et, gene_sim[1]])
                aug_ew = np.concatenate([aug_ew, gene_sim[2]])
        gene_sim_export = (gene_sim if args.mrna_sim_edges
                           else (np.empty((2, 0), dtype=np.int64),
                                 np.empty((0,), dtype=np.int64),
                                 np.empty((0,), dtype=np.float32)))
    else:
        bt_edge_index = np.empty((2, 0), dtype=np.int64)
        bt_edge_type = np.empty((0,), dtype=np.int64)
        bt_edge_weight = np.empty((0,), dtype=np.float32)
        gene_sim_export = (np.empty((2, 0), dtype=np.int64),
                           np.empty((0,), dtype=np.int64),
                           np.empty((0,), dtype=np.float32))

    # ---- Standardize node features ----
    train_idx = (np.unique(ei.reshape(-1)).astype(np.int64)
                 if ei.size else np.arange(len(nodes), dtype=np.int64))
    x, xm, xs = standardize_node_features(x_raw, train_idx)

    # ---- Assemble arrays dict ----
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
        # homogenized graph
        "augmented_edge_index": aug_ei,
        "augmented_edge_type": aug_et,
        "augmented_edge_weight": aug_ew,
        # per-type similarity edges (for ablation / dynamic filtering)
        "similarity_edge_index_mirna": mir_sim[0],
        "similarity_edge_type_mirna": mir_sim[1],
        "similarity_edge_weight_mirna": mir_sim[2],
        "similarity_edge_index_mrna": mrn_sim[0],
        "similarity_edge_type_mrna": mrn_sim[1],
        "similarity_edge_weight_mrna": mrn_sim[2],
        # hierarchical extras
        "belongs_to_edge_index": bt_edge_index,
        "belongs_to_edge_type": bt_edge_type,
        "belongs_to_edge_weight": bt_edge_weight,
        "similarity_edge_index_mrna_gene": gene_sim_export[0],
        "similarity_edge_type_mrna_gene": gene_sim_export[1],
        "similarity_edge_weight_mrna_gene": gene_sim_export[2],
    }
    for sp in ["train", "val", "test"]:
        m = clean["split"].eq(sp).to_numpy()
        sd = clean[m]
        arrays[f"{sp}_pos_edge_index"] = (sd[["src_idx", "dst_idx"]]
                                          .to_numpy(dtype=np.int64).T)
        arrays[f"{sp}_pos_edge_attr_raw"] = ea_raw[m]
        arrays[f"{sp}_pos_edge_attr"] = ea_std[m]
        arrays[f"{sp}_pos_label"] = np.ones((int(m.sum()),), dtype=np.int64)

    # ---- Save ----
    # npz (without node_sequences, which is a list of str)
    npz_a = {k: v for k, v in arrays.items() if k != "node_sequences"}
    np.savez_compressed(gdir / "graph_inputs.npz", **npz_a)

    # pt
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

    # ---- Count node types ----
    num_mirna = int((nt == NODE_MIRNA).sum())
    num_mrna_or_ts = int((nt == NODE_MRNA).sum())
    num_gene = int((nt == NODE_MRNA_GENE).sum()) if args.node_mode == "hierarchical" else 0

    # ---- Metadata ----
    meta = {
        "version": "embedding_optimed_v2",
        "species": species,
        "node_mode": args.node_mode,
        "sim_mode": args.sim_mode,
        "split_ratios": {"train": args.train_ratio, "val": args.val_ratio,
                         "test": args.test_ratio},
        "cleaning": report,
        "num_nodes": int(len(nodes)),
        "num_mirna_nodes": num_mirna,
        "num_mrna_or_target_site_nodes": num_mrna_or_ts,
        "num_mrna_gene_nodes": num_gene,
        "num_positive_edges": int(len(clean)),
        "split_counts": {s: int(clean["split"].eq(s).sum())
                         for s in ["train", "val", "test"]},
        "num_node_features": int(x.shape[1]),
        "num_edge_attr": int(ea_raw.shape[1]),
        "node_feature_columns": nf_cols,
        "mrna_sequence_source": (args.mrna_sequence_source
                                 if args.node_mode == "mrna" else "target_seq"),
        "node_feature_normalization": {
            "method": "z-score fit on train-graph nodes only",
            "num_train_nodes_for_fit": int(train_idx.size)},
        "edge_attr_columns": ea_cols,
        "num_mirna_sim_edges": int(mir_sim[0].shape[1]),
        "num_mrna_sim_edges": int(mrn_sim[0].shape[1]),
        "num_belongs_to_edges": int(bt_edge_index.shape[1]),
        "num_mrna_gene_sim_edges": int(gene_sim_export[0].shape[1]),
        "sim_mode": args.sim_mode,
        "mirna_sim_topk": args.mirna_sim_topk if args.mirna_sim_edges else 0,
        "mrna_sim_topk": args.mrna_sim_topk if args.mrna_sim_edges else 0,
        "sim_threshold": args.sim_threshold if args.sim_mode == "threshold_topk" else None,
        "homogeneous_edge_types": {
            "0": "miRNA→target (forward)",
            "1": "miRNA→target (reverse)",
            "2": "miRNA–miRNA similarity",
            "3": "target–target similarity",
            "4": "target_site→mRNA belongs_to",
        },
        "pair_feature_version": "v3",
        "pair_feature_dim": PAIR_V3_DIM,
        "pair_feature_extra_names": PAIR_V3_EXTRA_NAMES,
        "negative_sampling": ("Dynamic; excludes all known positives; "
                              "val/test edges not in message-passing graph."),
        "paths": {
            "nodes": str(nodes_path),
            "positive_edges": str(pos_path),
            "graph_inputs_npz": str(gdir / "graph_inputs.npz"),
            "graph_inputs_pt": torch_out,
        },
    }
    (gdir / "metadata.json").write_text(
        json.dumps(json_safe(meta), indent=2), encoding="utf-8")
    return meta


# ======================================================================
# main
# ======================================================================

def main() -> None:
    args = parse_args()
    files = discover_pos_files(args.input_dir)
    raw_by: dict[str, list[pd.DataFrame]] = {s: [] for s in SPECIES_ORDER}
    for rf in files:
        if rf.species not in raw_by:
            raw_by[rf.species] = []
        raw_by[rf.species].append(read_positive_file(rf, args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": "embedding_optimed_v2",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "seed": args.seed,
        "kmer_sizes": args.kmer_sizes,
        "node_mode": args.node_mode,
        "sim_mode": args.sim_mode,
        "mrna_sequence_source": args.mrna_sequence_source,
        "species": {},
    }
    for sp in SPECIES_ORDER:
        frames = raw_by.get(sp, [])
        if not frames:
            continue
        meta = export_species_graph(
            sp, pd.concat(frames, ignore_index=True, sort=False), args)
        summary["species"][sp] = meta
        extra = ""
        if args.node_mode == "hierarchical":
            extra = (f" mrna_gene={meta['num_mrna_gene_nodes']}"
                     f" belongs_to={meta['num_belongs_to_edges']}"
                     f" gene_sim={meta['num_mrna_gene_sim_edges']}")
        print(f"{sp}: mode={args.node_mode} sim={args.sim_mode} "
              f"nodes={meta['num_nodes']} "
              f"(mirna={meta['num_mirna_nodes']} "
              f"target/mrna={meta['num_mrna_or_target_site_nodes']}){extra} "
              f"pos={meta['num_positive_edges']} "
              f"train/val/test={meta['split_counts']} "
              f"mirna_sim={meta['num_mirna_sim_edges']} "
              f"mrna_sim={meta['num_mrna_sim_edges']}",
              flush=True)
    (args.output_dir / "preprocess_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"Saved -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
