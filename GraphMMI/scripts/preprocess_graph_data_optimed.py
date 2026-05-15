#!/usr/bin/env python3
"""Optimized graph preprocessing with three fixes over the original:

1.  Similarity edges are BIDIRECTIONAL (both i→j and j→i).
2.  mRNA node sequence defaults to *target_seq* (not full_mrna), controlled
    by --mrna-sequence-source {target,full}.
3.  When the same mRNA appears in multiple positive pairs with different
    target_seq values, the LONGEST target_seq is kept for node embedding.
"""
from __future__ import annotations

import argparse, hashlib, json, math, re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILE_RE = re.compile(r"^(?P<dataset_id>(?P<species>[A-Za-z]+)\d+)_pos\.csv$")
SPECIES_ORDER = ["human", "cow", "mouse", "worm"]
BASES = ["A", "C", "G", "U"]
BOOL_TO_INT = {True: 1, False: 0, "True": 1, "False": 0}

REQUIRED_COLUMNS = ["microRNA_name", "miRNA sequence", "mRNA_name", "target sequence", "full_mrna"]
METADATA_COLUMNS = {"Source", "Organism", "GI_ID", "microRNA_name", "miRNA sequence",
                    "target sequence", "number of reads", "mRNA_name", "full_mrna"}


@dataclass(frozen=True)
class RawPosFile:
    path: Path; species: str; dataset_id: str


# ==================================================================
# CLI
# ==================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optimized GraphMMI graph preprocessing")
    p.add_argument("--input-dir", type=Path, default=ROOT / "data/external")
    p.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/graph/optimized")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.72)
    p.add_argument("--val-ratio", type=float, default=0.08)
    p.add_argument("--test-ratio", type=float, default=0.20)
    p.add_argument("--kmer-sizes", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--drop-hot-pairing", action="store_true", default=True)
    p.add_argument("--keep-hot-pairing", action="store_true")
    p.add_argument("--mirna-sim-edges", action="store_true")
    p.add_argument("--mrna-sim-edges", action="store_true")
    p.add_argument("--mirna-sim-topk", type=int, default=5)
    p.add_argument("--mrna-sim-topk", type=int, default=5)
    p.add_argument("--mrna-sequence-source", choices=["target", "full"], default="target",
                   help="Which sequence to use for mRNA node embedding. 'target' avoids "
                        "diluting binding-site signal with full-transcript noise.")
    return p.parse_args()


# ==================================================================
# helpers (unchanged from original)
# ==================================================================

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
    if pd.isna(value): return ""
    return str(value).strip()

def normalize_sequence(value: object) -> str:
    if pd.isna(value): return ""
    return re.sub(r"\s+", "", str(value).upper().replace("T", "U"))

def short_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]

def discover_pos_files(input_dir: Path) -> list[RawPosFile]:
    files = []
    for path in sorted(input_dir.glob("*_pos.csv")):
        m = FILE_RE.match(path.name)
        if not m: continue
        cols = pd.read_csv(path, nrows=0).columns.tolist()
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing: raise ValueError(f"{path} missing columns: {missing}")
        files.append(RawPosFile(path=path, species=m.group("species").lower(), dataset_id=m.group("dataset_id").lower()))
    if not files: raise FileNotFoundError(f"No *_pos.csv in {input_dir}")
    return files


# ==================================================================
# reading  (FIX 2 — mRNA sequence source)
# ==================================================================

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

    # FIX 2: controlled by --mrna-sequence-source
    if args.mrna_sequence_source == "target":
        df["mrna_seq"] = df["target_seq"]
    else:  # full
        df["mrna_seq"] = np.where(
            df["full_mrna_seq"].astype(str).str.len() > 0,
            df["full_mrna_seq"], df["target_seq"])

    df["mirna_id"] = raw_file.species + "|" + df["microRNA_name"].map(clean_text)
    df["mrna_id"] = raw_file.species + "|" + df["mRNA_name"].map(clean_text)
    df["label"] = 1
    return df


# ==================================================================
# cleaning / split (unchanged)
# ==================================================================

def sequence_conflict_ids(frame: pd.DataFrame, id_col: str, seq_col: str) -> set[str]:
    cnt = frame.groupby(id_col)[seq_col].nunique(dropna=False)
    return set(cnt[cnt.gt(1)].index.astype(str))

def clean_positive_edges(raw_edges: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    report = {"raw_rows": int(len(raw_edges)), "dropped_missing_core_fields": 0,
              "dropped_pair_sequence_conflicts": 0, "dropped_node_sequence_conflicts": 0,
              "duplicate_pair_rows_removed": 0}
    core = (raw_edges["mirna_id"].astype(str).ne("") & raw_edges["mrna_id"].astype(str).ne("")
            & raw_edges["mirna_seq"].astype(str).ne("") & raw_edges["mrna_seq"].astype(str).ne(""))
    report["dropped_missing_core_fields"] = int((~core).sum())
    edges = raw_edges[core].copy()

    pair_g = edges.groupby(["mirna_id", "mrna_id"], dropna=False)
    conflicts = pair_g.filter(lambda g: g["mirna_seq"].nunique(dropna=False) > 1
                                        or g["mrna_seq"].nunique(dropna=False) > 1)
    if not conflicts.empty:
        ck = set(zip(conflicts["mirna_id"], conflicts["mrna_id"]))
        report["dropped_pair_sequence_conflicts"] = int(edges[["mirna_id","mrna_id"]].apply(tuple, axis=1).isin(ck).sum())
        edges = edges[~edges[["mirna_id","mrna_id"]].apply(tuple, axis=1).isin(ck)].copy()

    before = len(edges)
    edges = edges.sort_values(["dataset_id","source_row"]).drop_duplicates(["mirna_id","mrna_id"], keep="first").reset_index(drop=True)
    report["duplicate_pair_rows_removed"] = int(before - len(edges))

    mc = sequence_conflict_ids(edges, "mirna_id", "mirna_seq")
    nc = sequence_conflict_ids(edges, "mrna_id", "mrna_seq")
    if mc or nc:
        cm = edges["mirna_id"].isin(mc) | edges["mrna_id"].isin(nc)
        report["dropped_node_sequence_conflicts"] = int(cm.sum())
        edges = edges[~cm].reset_index(drop=True)

    report["clean_positive_edges"] = int(len(edges))
    return edges, report

def split_edges(edges: pd.DataFrame, seed: int, train_ratio: float, val_ratio: float, test_ratio: float) -> pd.Series:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Split ratios must sum to 1.0")
    rng = np.random.default_rng(seed); indices = edges.index.to_numpy(); rng.shuffle(indices)
    n_total = len(indices); n_val = int(round(n_total * val_ratio)); n_test = int(round(n_total * test_ratio))
    n_train = n_total - n_val - n_test
    split = pd.Series(index=edges.index, dtype="object")
    split.loc[indices[:n_train]] = "train"
    split.loc[indices[n_train:n_train+n_val]] = "val"
    split.loc[indices[n_train+n_val:]] = "test"
    return split


# ==================================================================
# sequence features (unchanged)
# ==================================================================

def kmers(k: int) -> list[str]:
    words = [""]
    for _ in range(k): words = [p + b for p in words for b in BASES]
    return words

def kmer_feature_names(kmer_sizes: Iterable[int]) -> list[str]:
    names = []
    for k in kmer_sizes: names.extend([f"kmer{k}_{w}" for w in kmers(k)])
    return names

def sequence_numeric_features(seq: str, kmer_sizes: Iterable[int]) -> dict[str, float]:
    valid = [b for b in seq if b in BASES]; length = len(valid)
    feats = {"seq_length": float(length), "seq_log_length": float(np.log1p(length)),
             "seq_gc": float((valid.count("G") + valid.count("C")) / length) if length else 0.0}
    for k in kmer_sizes:
        vocab = kmers(k); cnt = {w: 0.0 for w in vocab}; total = 0
        for idx in range(max(len(seq) - k + 1, 0)):
            word = seq[idx:idx+k]
            if all(b in BASES for b in word): cnt[word] += 1.0; total += 1
        for w in vocab: feats[f"kmer{k}_{w}"] = cnt[w] / float(total) if total else 0.0
    return feats

def seed_features(seq: str) -> dict[str, float]:
    regions = {"seed_2_7": seq[1:7], "seed_3_8": seq[2:8]}; feats = {}
    for name, region in regions.items():
        valid = [b for b in region if b in BASES]; length = len(valid)
        feats[f"{name}_len"] = float(length)
        feats[f"{name}_gc"] = float((valid.count("G")+valid.count("C"))/length) if length else 0.0
        for base in BASES: feats[f"{name}_{base}_freq"] = float(valid.count(base)/length) if length else 0.0
    return feats


# ==================================================================
# node building  (FIX 3 — longest target_seq per mRNA)
# ==================================================================

def build_nodes(edges: pd.DataFrame, kmer_sizes: list[int]) -> tuple[pd.DataFrame, dict[str, int]]:
    mirna_cols = ["mirna_id", "species", "microRNA_name", "mirna_seq", "dataset_id", "Source", "Organism"]
    mrna_cols = ["mrna_id", "species", "mRNA_name", "mrna_seq", "target_seq",
                 "full_mrna_seq", "dataset_id", "Source", "Organism", "GI_ID"]

    mirna_nodes = (edges.reindex(columns=mirna_cols).drop_duplicates("mirna_id", keep="first")
                   .rename(columns={"mirna_id": "node_id", "microRNA_name": "source_name", "mirna_seq": "sequence"}))
    mirna_nodes["node_type"] = "mirna"; mirna_nodes["node_type_id"] = 0

    # FIX 3: keep longest target_seq per mRNA for better embedding
    mrna_tmp = edges.reindex(columns=mrna_cols).copy()
    mrna_tmp["_target_len"] = mrna_tmp["target_seq"].fillna("").astype(str).str.len()
    mrna_tmp = mrna_tmp.sort_values("_target_len", ascending=False)
    mrna_nodes = (mrna_tmp.drop_duplicates("mrna_id", keep="first")
                  .rename(columns={"mrna_id": "node_id", "mRNA_name": "source_name", "mrna_seq": "sequence"}))
    mrna_nodes["node_type"] = "mrna"; mrna_nodes["node_type_id"] = 1

    nodes = pd.concat([mirna_nodes, mrna_nodes], ignore_index=True, sort=False)
    nodes.insert(0, "node_idx", np.arange(len(nodes), dtype=np.int64))
    sp2id = {s: i for i, s in enumerate(SPECIES_ORDER)}
    nodes["species_id"] = nodes["species"].map(sp2id).fillna(-1).astype("int64")
    nodes["sequence_hash"] = nodes["sequence"].fillna("").astype(str).map(short_hash)
    nodes["mirna_len"] = np.where(nodes["node_type"].eq("mirna"), nodes["sequence"].astype(str).str.len(), np.nan)
    nodes["mrna_len"] = np.where(nodes["node_type"].eq("mrna"), nodes["sequence"].astype(str).str.len(), np.nan)

    frows = []
    for row in nodes.itertuples(index=False):
        seq = str(row.sequence)
        feats = sequence_numeric_features(seq, kmer_sizes)
        seeds = seed_features(seq) if row.node_type == "mirna" else {k: 0.0 for k in seed_features("")}
        feats.update(seeds); frows.append(feats)
    fdf = pd.DataFrame(frows).fillna(0.0).astype("float32")
    nodes = pd.concat([nodes.reset_index(drop=True), fdf.reset_index(drop=True)], axis=1)
    nodes["mirna_gc"] = np.where(nodes["node_type"].eq("mirna"), nodes["seq_gc"], np.nan)
    nodes["mrna_gc"] = np.where(nodes["node_type"].eq("mrna"), nodes["seq_gc"], np.nan)
    node_index = dict(zip(nodes["node_id"].astype(str), nodes["node_idx"].astype(int)))
    return nodes, node_index


# ==================================================================
# edge attr (unchanged)
# ==================================================================

def bools_to_numeric(series: pd.Series) -> pd.Series:
    def convert(value):
        try: return BOOL_TO_INT.get(value, value)
        except TypeError: return value
    return series.map(convert)

def infer_edge_attr_columns(edges: pd.DataFrame, drop_hot: bool) -> list[str]:
    forbidden = METADATA_COLUMNS | {"raw_index","species","dataset_id","source_file","source_row",
                                     "mirna_id","mrna_id","mirna_seq","target_seq","full_mrna_seq",
                                     "mrna_seq","label","split","src_idx","dst_idx"}
    cols = []
    for c in edges.columns:
        if c in forbidden: continue
        if drop_hot and str(c).startswith("HotPairing"): continue
        s = bools_to_numeric(edges[c])
        if pd.to_numeric(s, errors="coerce").notna().any(): cols.append(c)
    return cols

def edge_attr_matrix(edges: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns: return np.zeros((len(edges), 0), dtype=np.float32)
    frame = edges.reindex(columns=columns)
    return frame.apply(bools_to_numeric).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)

def standardize_edge_attr(all_attr, split):
    if all_attr.shape[1] == 0: return all_attr, np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    train = all_attr[split.eq("train").to_numpy()]; m, s = train.mean(0).astype(np.float32), train.std(0).astype(np.float32)
    s[s < 1e-6] = 1.0
    return ((all_attr - m) / s).astype(np.float32), m, s

def standardize_node_features(x_raw, train_idx):
    if x_raw.shape[1] == 0: return x_raw, np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)
    if train_idx.size == 0: train_idx = np.arange(x_raw.shape[0], dtype=np.int64)
    train = x_raw[train_idx]; m, s = train.mean(0).astype(np.float32), train.std(0).astype(np.float32)
    s[s < 1e-6] = 1.0
    return ((x_raw - m) / s).astype(np.float32), m, s

def export_edges_csv(edges, path):
    cols = ["species","dataset_id","source_file","source_row","split","label","mirna_id","mrna_id","src_idx","dst_idx"]
    present = [c for c in cols if c in edges.columns]
    path.parent.mkdir(parents=True, exist_ok=True); edges[present].to_csv(path, index=False)


# ==================================================================
# similarity edges  (FIX 1 — bidirectional)
# ==================================================================

def _make_undirected(ei: np.ndarray, ew: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add reverse direction so similarity edges are truly undirected."""
    if ei.size == 0: return ei, ew
    rev = ei[::-1]
    return np.concatenate([ei, rev], axis=1), np.concatenate([ew, ew], axis=0)

def _cosine_similarity_topk(vecs: np.ndarray, topk: int, rng: np.random.Generator):
    n = vecs.shape[0]
    if n < 2: return np.empty((2,0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True); norms = np.where(norms < 1e-12, 1.0, norms)
    unit = vecs.astype(np.float64) / norms; edges, weights = [], []
    eff_k = min(topk, n - 1)
    for i in range(n):
        sim = unit[i] @ unit.T; sim[i] = -2.0
        top = np.argpartition(-sim, eff_k)[:eff_k] if eff_k < n else np.argsort(-sim)[:eff_k]
        for j in top:
            s = float(sim[j])
            if s > 0.0: edges.append((i, j)); weights.append(s)
    if not edges: return np.empty((2,0), dtype=np.int64), np.empty((0,), dtype=np.float32)
    return np.asarray(edges, dtype=np.int64).T, np.asarray(weights, dtype=np.float32)

def build_mirna_similarity_edges(nodes, x_raw, kmer_sizes, args):
    rng = np.random.default_rng(args.seed)
    mirna = nodes[nodes["node_type"].eq("mirna")].copy()
    if len(mirna) < 2:
        return np.empty((2,0), dtype=np.int64), np.full((0,),2,dtype=np.int64), np.empty((0,),dtype=np.float32)
    l2g = {i: int(mirna.iloc[i].node_idx) for i in range(len(mirna))}
    seqs = mirna["sequence"].fillna("").astype(str).tolist()
    edge_set: dict[tuple[int,int], float] = {}

    for _name, slc in [("seed_2_7", slice(1,7)), ("seed_3_8", slice(2,8))]:
        sm: dict[str, list[int]] = {}
        for li, seq in enumerate(seqs):
            seed = seq[slc] if len(seq) >= slc.stop else ""
            if seed: sm.setdefault(seed, []).append(li)
        for grp in sm.values():
            if len(grp) < 2: continue
            for a in range(len(grp)):
                for b in range(a+1, len(grp)):
                    gi, gj = l2g[grp[a]], l2g[grp[b]]; key = (min(gi,gj), max(gi,gj))
                    edge_set[key] = max(edge_set.get(key, 0.0), 1.0)

    kmer_cols = [c for c in nodes.columns if c.startswith("kmer1_") or c.startswith("kmer2_") or c.startswith("kmer3_")]
    if kmer_cols:
        kvs = mirna[kmer_cols].to_numpy(dtype=np.float32)
        ci, cw = _cosine_similarity_topk(kvs, args.mirna_sim_topk, rng)
        for col in range(ci.shape[1]):
            gi, gj = l2g[int(ci[0,col])], l2g[int(ci[1,col])]; key = (min(gi,gj), max(gi,gj))
            edge_set[key] = max(edge_set.get(key, 0.0), float(cw[col]))

    if not edge_set:
        return np.empty((2,0), dtype=np.int64), np.full((0,),2,dtype=np.int64), np.empty((0,),dtype=np.float32)

    pairs = sorted(edge_set.keys())
    ei = np.array(pairs, dtype=np.int64).T; ew = np.array([edge_set[k] for k in pairs], dtype=np.float32)
    # FIX 1 — make bidirectional
    ei, ew = _make_undirected(ei, ew)
    et = np.full(ei.shape[1], 2, dtype=np.int64)
    return ei, et, ew

def build_mrna_similarity_edges(nodes, args):
    rng = np.random.default_rng(args.seed)
    mrna = nodes[nodes["node_type"].eq("mrna")].copy()
    if len(mrna) < 2:
        return np.empty((2,0), dtype=np.int64), np.full((0,),3,dtype=np.int64), np.empty((0,),dtype=np.float32)
    l2g = {i: int(mrna.iloc[i].node_idx) for i in range(len(mrna))}
    target_seqs = mrna.get("target_seq", pd.Series([""]*len(mrna), index=mrna.index))
    target_seqs = target_seqs.fillna("").astype(str).tolist()

    kmer_names = kmer_feature_names(args.kmer_sizes)
    vecs = np.zeros((len(mrna), len(kmer_names)), dtype=np.float32)
    for i, seq in enumerate(target_seqs):
        feats = sequence_numeric_features(seq, args.kmer_sizes)
        for j, name in enumerate(kmer_names): vecs[i,j] = float(feats.get(name, 0.0))

    ci, cw = _cosine_similarity_topk(vecs, args.mrna_sim_topk, rng)
    if ci.size == 0:
        return np.empty((2,0), dtype=np.int64), np.full((0,),3,dtype=np.int64), np.empty((0,),dtype=np.float32)

    edge_set: set[tuple[int,int]] = set(); pairs, weights = [], []
    for col in range(ci.shape[1]):
        gi, gj = l2g[int(ci[0,col])], l2g[int(ci[1,col])]; key = (min(gi,gj), max(gi,gj))
        if key in edge_set: continue
        edge_set.add(key); pairs.append(key); weights.append(float(cw[col]))

    if not pairs:
        return np.empty((2,0), dtype=np.int64), np.full((0,),3,dtype=np.int64), np.empty((0,),dtype=np.float32)

    ei = np.array(pairs, dtype=np.int64).T; ew = np.array(weights, dtype=np.float32)
    # FIX 1 — make bidirectional
    ei, ew = _make_undirected(ei, ew)
    et = np.full(ei.shape[1], 3, dtype=np.int64)
    return ei, et, ew

def homogenize_edge_index(edge_index_undirected, mirna_sim, mrna_sim):
    half = edge_index_undirected.shape[1] // 2
    ei = [edge_index_undirected]
    et = [np.concatenate([np.zeros(half, dtype=np.int64), np.ones(half, dtype=np.int64)])]
    ew = [np.ones(edge_index_undirected.shape[1], dtype=np.float32)]
    for se, st, sw in (mirna_sim, mrna_sim):
        if se.size: ei.append(se); et.append(st); ew.append(sw)
    return (np.concatenate(ei, axis=1) if len(ei)>1 else ei[0],
            np.concatenate(et) if len(et)>1 else et[0],
            np.concatenate(ew) if len(ew)>1 else ew[0])


# ==================================================================
# main export
# ==================================================================

def export_species_graph(species: str, edges: pd.DataFrame, args: argparse.Namespace) -> dict:
    gdir = args.output_dir / species; gdir.mkdir(parents=True, exist_ok=True)
    sseed = args.seed + sum(ord(c) for c in species)
    clean, report = clean_positive_edges(edges)
    clean["split"] = split_edges(clean, sseed, args.train_ratio, args.val_ratio, args.test_ratio)
    nodes, ni = build_nodes(clean, args.kmer_sizes)
    clean["src_idx"] = clean["mirna_id"].map(ni).astype("int64")
    clean["dst_idx"] = clean["mrna_id"].map(ni).astype("int64")

    drop_hot = args.drop_hot_pairing and not args.keep_hot_pairing
    ea_cols = infer_edge_attr_columns(clean, drop_hot=drop_hot)
    ea_raw = edge_attr_matrix(clean, ea_cols)
    ea_std, ea_mean, ea_scale = standardize_edge_attr(ea_raw, clean["split"])

    nodes_path = gdir / "nodes.csv"; pos_path = gdir / "positive_edges.csv"
    nodes.to_csv(nodes_path, index=False); export_edges_csv(clean, pos_path)
    for sp in ["train","val","test"]:
        export_edges_csv(clean[clean["split"].eq(sp)], gdir / f"{sp}_pos_edges.csv")

    nf_cols = ["seq_log_length","seq_gc"] + kmer_feature_names(args.kmer_sizes) + list(seed_features("").keys())
    x_raw = nodes[nf_cols].to_numpy(dtype=np.float32)
    nt = nodes["node_type_id"].to_numpy(dtype=np.int64)
    sid = nodes["species_id"].to_numpy(dtype=np.int64)
    train_e = clean[clean["split"].eq("train")]
    ei = train_e[["src_idx","dst_idx"]].to_numpy(dtype=np.int64).T
    ei_undir = np.concatenate([ei, ei[::-1]], axis=1) if ei.size else ei
    all_pos = clean[["src_idx","dst_idx"]].to_numpy(dtype=np.int64).T

    # similarity
    empty = np.empty((2,0), dtype=np.int64), np.empty((0,),dtype=np.int64), np.empty((0,),dtype=np.float32)
    mir_sim = empty; mrn_sim = empty
    if args.mirna_sim_edges: mir_sim = build_mirna_similarity_edges(nodes, x_raw, args.kmer_sizes, args)
    if args.mrna_sim_edges: mrn_sim = build_mrna_similarity_edges(nodes, args)
    aug_ei, aug_et, aug_ew = homogenize_edge_index(ei_undir, mir_sim, mrn_sim)

    train_idx = np.unique(ei.reshape(-1)).astype(np.int64) if ei.size else np.arange(len(nodes), dtype=np.int64)
    x, xm, xs = standardize_node_features(x_raw, train_idx)

    arrays: dict = {
        "x":x, "x_raw":x_raw, "node_type":nt, "species_id":sid,
        "edge_index_train_pos":ei, "edge_index_train_pos_undirected":ei_undir,
        "all_positive_edge_index":all_pos,
        "node_ids": nodes["node_id"].astype(str).to_numpy(dtype=str),
        "node_sequences": nodes["sequence"].fillna("").astype(str).tolist(),
        "node_feature_names": np.asarray(nf_cols, dtype=str),
        "node_feature_mean":xm, "node_feature_scale":xs,
        "edge_attr_names": np.asarray(ea_cols, dtype=str),
        "edge_attr_mean":ea_mean, "edge_attr_scale":ea_scale,
        "augmented_edge_index":aug_ei, "augmented_edge_type":aug_et, "augmented_edge_weight":aug_ew,
        "similarity_edge_index_mirna":mir_sim[0], "similarity_edge_type_mirna":mir_sim[1], "similarity_edge_weight_mirna":mir_sim[2],
        "similarity_edge_index_mrna":mrn_sim[0], "similarity_edge_type_mrna":mrn_sim[1], "similarity_edge_weight_mrna":mrn_sim[2],
    }
    for sp in ["train","val","test"]:
        m = clean["split"].eq(sp).to_numpy(); sd = clean[m]
        arrays[f"{sp}_pos_edge_index"] = sd[["src_idx","dst_idx"]].to_numpy(dtype=np.int64).T
        arrays[f"{sp}_pos_edge_attr_raw"] = ea_raw[m]
        arrays[f"{sp}_pos_edge_attr"] = ea_std[m]
        arrays[f"{sp}_pos_label"] = np.ones((int(m.sum()),), dtype=np.int64)

    # npz
    npz_a = {k:v for k,v in arrays.items() if k!="node_sequences"}
    np.savez_compressed(gdir / "graph_inputs.npz", **npz_a)

    # pt
    torch_out = ""
    try:
        import torch
        tp = gdir / "graph_inputs.pt"; tpt = gdir / "graph_inputs.pt.tmp"
        torch.save({k:v for k,v in arrays.items()}, tpt); tpt.replace(tp); torch_out = str(tp)
    except ModuleNotFoundError: pass

    meta = {
        "species":species,
        "split_ratios":{"train":args.train_ratio,"val":args.val_ratio,"test":args.test_ratio},
        "cleaning":report,
        "num_nodes":int(len(nodes)), "num_mirna_nodes":int(nt[nt==0].size), "num_mrna_nodes":int(nt[nt==1].size),
        "num_positive_edges":int(len(clean)),
        "split_counts":{s:int(clean["split"].eq(s).sum()) for s in ["train","val","test"]},
        "num_node_features":int(x.shape[1]), "num_edge_attr":int(ea_raw.shape[1]),
        "node_feature_columns":nf_cols,
        "mrna_sequence_source": args.mrna_sequence_source,
        "node_feature_normalization":{"method":"z-score fit on train-graph nodes only",
                                       "num_train_nodes_for_fit":int(train_idx.size)},
        "edge_attr_columns":ea_cols,
        "num_mirna_sim_edges":int(mir_sim[0].shape[1]), "num_mrna_sim_edges":int(mrn_sim[0].shape[1]),
        "mirna_sim_bidirectional":args.mirna_sim_edges,
        "mrna_sim_bidirectional":args.mrna_sim_edges,
        "homogeneous_edge_types":{"0":"miRNA->mRNA","1":"mRNA->miRNA","2":"miRNA-miRNA sim","3":"mRNA-mRNA sim"},
        "mirna_sim_topk":args.mirna_sim_topk if args.mirna_sim_edges else 0,
        "mrna_sim_topk":args.mrna_sim_topk if args.mrna_sim_edges else 0,
        "negative_sampling":"Dynamic; excludes all known positives; val/test edges not in message-passing graph.",
        "paths":{"nodes":str(nodes_path),"positive_edges":str(pos_path),"graph_inputs_npz":str(gdir/"graph_inputs.npz"),"graph_inputs_pt":torch_out},
    }
    (gdir / "metadata.json").write_text(json.dumps(json_safe(meta), indent=2), encoding="utf-8")
    return meta


# ==================================================================
# main
# ==================================================================

def main() -> None:
    args = parse_args()
    files = discover_pos_files(args.input_dir)
    raw_by: dict[str, list[pd.DataFrame]] = {s:[] for s in SPECIES_ORDER}
    for rf in files:
        if rf.species not in raw_by: raw_by[rf.species] = []
        raw_by[rf.species].append(read_positive_file(rf, args))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"input_dir":args.input_dir,"output_dir":args.output_dir,"seed":args.seed,
               "kmer_sizes":args.kmer_sizes,"mrna_sequence_source":args.mrna_sequence_source,
               "species":{}}
    for sp in SPECIES_ORDER:
        frames = raw_by.get(sp, [])
        if not frames: continue
        meta = export_species_graph(sp, pd.concat(frames, ignore_index=True, sort=False), args)
        summary["species"][sp] = meta
        print(f"{sp}: nodes={meta['num_nodes']} pos={meta['num_positive_edges']} "
              f"train/val/test={meta['split_counts']} "
              f"mirna_sim={meta['num_mirna_sim_edges']} mrna_sim={meta['num_mrna_sim_edges']}",
              flush=True)
    (args.output_dir / "preprocess_summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(f"Saved -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
