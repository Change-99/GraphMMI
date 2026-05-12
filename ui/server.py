from __future__ import annotations

import csv
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


ROOT = Path(__file__).resolve().parent
GNN_ROOT = ROOT.parent / "gnn"
DATA_ROOT = GNN_ROOT / "data" / "processed" / "graphsage_mrna"
TRANSFER_ROOT = GNN_ROOT / "runs" / "species_transfer_matrix"
DYNAMIC_ROOT = GNN_ROOT / "runs" / "graphsage_dynamic_neg"

if str(GNN_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(GNN_ROOT / "src"))

from mti_graphsage import GraphSAGENodePairPredictor  # noqa: E402


SPECIES = ["human", "cow", "mouse", "worm"]
PLOT_MAP = {
    "score_distribution": "validation_score_distribution.png",
    "pr_curve": "validation_pr_curve.png",
    "calibration_curve": "calibration_curve.png",
}


app = FastAPI(title="miRNA-mRNA UI API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def latest_dir(root: Path) -> Path:
    candidates = [path for path in root.iterdir() if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run directories in {root}")
    return sorted(candidates)[-1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_run_dir(run_id: str | None) -> Path:
    if run_id:
        candidate = Path(run_id)
        if candidate.exists():
            return candidate
        candidate = TRANSFER_ROOT / run_id
        if candidate.exists():
            return candidate
    return latest_dir(TRANSFER_ROOT)


def resolve_model_run_dir(run_id: str | None) -> Path:
    if run_id:
        candidate = Path(run_id)
        if candidate.exists():
            return candidate
        candidate = DYNAMIC_ROOT / run_id
        if candidate.exists():
            return candidate
    if DYNAMIC_ROOT.exists():
        return latest_dir(DYNAMIC_ROOT)
    raise FileNotFoundError(f"No model runs in {DYNAMIC_ROOT}")


@lru_cache(maxsize=32)
def load_metadata(split: str, species: str) -> dict[str, Any]:
    path = DATA_ROOT / split / species / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


@lru_cache(maxsize=32)
def load_nodes_df(split: str, species: str) -> pd.DataFrame:
    path = DATA_ROOT / split / species / "nodes.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


@lru_cache(maxsize=32)
def load_positive_edges_df(split: str, species: str) -> pd.DataFrame:
    path = DATA_ROOT / split / species / "positive_edges.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


@lru_cache(maxsize=32)
def load_candidate_edges_df(split: str, species: str) -> pd.DataFrame:
    path = DATA_ROOT / split / species / "candidate_edges.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


@lru_cache(maxsize=32)
def load_graph_npz(split: str, species: str) -> dict[str, np.ndarray]:
    path = DATA_ROOT / split / species / "graphsage_inputs.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def species_summary(split: str, species: str) -> dict[str, Any]:
    meta = load_metadata(split, species)
    return {
        "species": species,
        "split": split,
        "num_nodes": meta["num_nodes"],
        "num_mirna_nodes": meta["num_mirna_nodes"],
        "num_mrna_nodes": meta["num_mrna_nodes"],
        "positive_edges": meta["deduplicated_positive_edges"],
        "node_feature_dim": meta["num_node_features"],
        "edge_feature_dim": meta["num_edge_features"],
        "split_counts": meta["split_counts"],
    }


def search_nodes(split: str, species: str, kind: str, query_text: str, limit: int) -> list[dict[str, Any]]:
    df = load_nodes_df(split, species)
    subset = df.copy()
    if kind in {"mirna", "mrna"}:
        subset = subset[subset["node_type"].eq(kind)]
    if query_text:
        q = query_text.lower()
        mask = subset["node_id"].astype(str).str.lower().str.contains(q, na=False) | subset["source_name"].astype(
            str
        ).str.lower().str.contains(q, na=False)
        subset = subset[mask]
    subset = subset.head(limit)
    results: list[dict[str, Any]] = []
    for row in subset.itertuples(index=False):
        results.append(
            {
                "id": str(row.node_id),
                "idx": int(row.node_idx),
                "name": str(row.source_name),
                "type": str(row.node_type),
                "sequence_hash": str(row.sequence_hash),
                "sequence_length": int(getattr(row, "seq_length", 0)) if hasattr(row, "seq_length") else None,
            }
        )
    return results


def graph_payload(split: str, species: str, limit_mirna: int, limit_edges: int) -> dict[str, Any]:
    nodes_df = load_nodes_df(split, species)
    edges_df = load_positive_edges_df(split, species)
    mirna_nodes = nodes_df[nodes_df["node_type"].eq("mirna")].head(limit_mirna)
    mirna_ids = set(mirna_nodes["node_id"].astype(str))
    edges = edges_df[edges_df["mirna_id"].astype(str).isin(mirna_ids)].head(limit_edges)
    node_ids = set(edges["mirna_id"].astype(str)).union(set(edges["mrna_id"].astype(str)))
    nodes = nodes_df[nodes_df["node_id"].astype(str).isin(node_ids)].copy()
    node_payload = [
        {
            "id": str(row.node_id),
            "idx": int(row.node_idx),
            "name": str(row.source_name),
            "type": str(row.node_type),
            "sequence_hash": str(row.sequence_hash),
        }
        for row in nodes.itertuples(index=False)
    ]
    edge_payload = [
        {
            "source": int(row.src_idx),
            "target": int(row.dst_idx),
            "source_id": str(row.mirna_id),
            "target_id": str(row.mrna_id),
            "split": str(row.split),
            "label": int(row.label),
            "edge_source": str(row.edge_source),
        }
        for row in edges.itertuples(index=False)
    ]
    return {"species": species, "nodes": node_payload, "edges": edge_payload}


def read_matrix_csv(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    species = df.columns[1:].tolist()
    matrix = [[json_safe(value) for value in row[1:]] for row in df.itertuples(index=False, name=None)]
    rows = []
    for row in df.itertuples(index=False, name=None):
        rows.append({"source": row[0], "values": list(row[1:])})
    return {"species": species, "matrix": matrix, "rows": rows}


def read_transfer_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "transfer_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def transfer_payload(run_id: str | None) -> dict[str, Any]:
    run_dir = resolve_run_dir(run_id)
    rows = read_transfer_rows(run_dir)
    species = SPECIES
    matrices = {}
    for metric in ["auc", "ap", "accuracy", "f1"]:
        matrices[metric] = read_matrix_csv(run_dir / f"transfer_{metric}_matrix.csv")["matrix"]
    return {
        "run_id": run_dir.name,
        "run_path": str(run_dir),
        "species": species,
        "rows": rows,
        "matrices": matrices,
    }


def diagnostics_summary_payload(run_id: str | None) -> dict[str, Any]:
    run_dir = resolve_run_dir(run_id)
    path = run_dir / "diagnostics" / "diagnostic_summary.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    return {
        "run_id": run_dir.name,
        "rows": df.to_dict(orient="records"),
    }


def transfer_run_list() -> list[dict[str, Any]]:
    runs = []
    if not TRANSFER_ROOT.exists():
        return runs
    for path in sorted(TRANSFER_ROOT.iterdir()):
        if not path.is_dir():
            continue
        metrics = path / "transfer_metrics.csv"
        if metrics.exists():
            runs.append(
                {
                    "run_id": path.name,
                    "path": str(path),
                    "split": "cold_mirna" if "cold" in path.name else "random",
                    "created_at": path.name.replace("_", " "),
                    "has_diagnostics": (path / "diagnostics" / "diagnostic_summary.csv").exists(),
                    "has_thresholds": (path / "transfer_best_threshold_matrix.csv").exists(),
                }
            )
    return runs


def dynamic_model_run_list() -> list[dict[str, Any]]:
    runs = []
    if not DYNAMIC_ROOT.exists():
        return runs
    for path in sorted(DYNAMIC_ROOT.iterdir()):
        if not path.is_dir():
            continue
        if (path / "config.json").exists() and (path / "models").exists():
            runs.append({"run_id": path.name, "path": str(path)})
    return runs


def species_run_summary() -> dict[str, Any]:
    split = "cold_mirna"
    species_summaries = {species: species_summary(split, species) for species in SPECIES}
    totals = {
        "mirna_nodes": sum(item["num_mirna_nodes"] for item in species_summaries.values()),
        "mrna_nodes": sum(item["num_mrna_nodes"] for item in species_summaries.values()),
        "nodes": sum(item["num_nodes"] for item in species_summaries.values()),
        "positive_edges": sum(item["positive_edges"] for item in species_summaries.values()),
        "node_feature_dim": 22,
        "edge_feature_dim": 0,
    }
    transfer = transfer_payload(None)
    rows = transfer["rows"]
    mean_auc = float(np.mean([row["test_auc"] for row in rows]))
    mean_ap = float(np.mean([row["test_ap"] for row in rows]))
    mean_f1 = float(np.mean([row["test_f1"] for row in rows]))
    mean_acc = float(np.mean([row["test_accuracy"] for row in rows]))
    return {
        "split": split,
        "species": SPECIES,
        "totals": totals,
        "overview_metrics": [
            {"label": "AUC", "value": mean_auc},
            {"label": "AP", "value": mean_ap},
            {"label": "F1", "value": mean_f1},
            {"label": "ACC", "value": mean_acc},
        ],
        "best_model": {
            "name": "GraphSAGE dynamic negative",
            "primary_metric": "AUC",
            "mean_auc": mean_auc,
            "mean_ap": mean_ap,
            "mean_f1": mean_f1,
        },
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/runs")
def api_runs() -> dict[str, Any]:
    return {
        "transfer_runs": transfer_run_list(),
        "model_runs": dynamic_model_run_list(),
        "default_transfer_run_id": latest_dir(TRANSFER_ROOT).name if TRANSFER_ROOT.exists() else None,
    }


@app.get("/api/dashboard/summary")
def api_dashboard_summary() -> dict[str, Any]:
    return species_run_summary()


@app.get("/api/species")
def api_species() -> dict[str, Any]:
    return {"species": SPECIES}


@app.get("/api/species/{species}/summary")
def api_species_summary(species: str, split: str = Query("cold_mirna")) -> dict[str, Any]:
    if species not in SPECIES:
        raise HTTPException(404, f"Unknown species: {species}")
    return species_summary(split, species)


@app.get("/api/species/{species}/graph")
def api_species_graph(
    species: str,
    split: str = Query("cold_mirna"),
    limit_mirna: int = 20,
    limit_edges: int = 80,
) -> dict[str, Any]:
    if species not in SPECIES:
        raise HTTPException(404, f"Unknown species: {species}")
    return graph_payload(split, species, limit_mirna, limit_edges)


@app.get("/api/species/{species}/nodes/search")
def api_species_nodes_search(
    species: str,
    split: str = Query("cold_mirna"),
    kind: str = Query("all"),
    q: str = Query(""),
    limit: int = Query(20, ge=1, le=100),
) -> dict[str, Any]:
    if species not in SPECIES:
        raise HTTPException(404, f"Unknown species: {species}")
    return {
        "species": species,
        "split": split,
        "kind": kind,
        "query": q,
        "items": search_nodes(split, species, kind, q, limit),
    }


@app.get("/api/results/transfer")
def api_results_transfer(run_id: str | None = None) -> dict[str, Any]:
    return transfer_payload(run_id)


@app.get("/api/results/transfer/heatmap/{metric}")
def api_results_transfer_heatmap(metric: str, run_id: str | None = None):
    if metric not in {"auc", "ap", "accuracy", "f1"}:
        raise HTTPException(404, f"Unknown metric: {metric}")
    run_dir = resolve_run_dir(run_id)
    path = run_dir / f"transfer_{metric}_heatmap.png"
    if not path.exists():
        raise HTTPException(404, str(path))
    return FileResponse(path)


@app.get("/api/results/thresholds")
def api_results_thresholds(run_id: str | None = None) -> dict[str, Any]:
    run_dir = resolve_run_dir(run_id)
    path = run_dir / "transfer_best_threshold_matrix.csv"
    if not path.exists():
        raise HTTPException(404, str(path))
    return read_matrix_csv(path)


@app.get("/api/diagnostics/summary")
def api_diagnostics_summary(run_id: str | None = None) -> dict[str, Any]:
    return diagnostics_summary_payload(run_id)


@app.get("/api/diagnostics/{source}/{target}/plot/{plot_type}")
def api_diagnostics_plot(source: str, target: str, plot_type: str, run_id: str | None = None):
    if plot_type not in PLOT_MAP:
        raise HTTPException(404, f"Unknown plot type: {plot_type}")
    run_dir = resolve_run_dir(run_id)
    path = run_dir / "diagnostics" / f"{source}_to_{target}" / PLOT_MAP[plot_type]
    if not path.exists():
        raise HTTPException(404, str(path))
    return FileResponse(path)


def load_transfer_model(run_id: str, source: str, device: torch.device):
    run_dir = resolve_run_dir(run_id)
    config = read_json(run_dir / "config.json")
    model = GraphSAGENodePairPredictor(
        node_feature_dim=22,
        hidden_channels=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
        dropout=float(config["dropout"]),
        decoder_hidden_channels=int(config["decoder_hidden_dim"]),
    ).to(device)
    ckpt = torch.load(run_dir / "models" / source / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, config, run_dir


@lru_cache(maxsize=32)
def load_species_graph(split: str, species: str):
    data = load_graph_npz(split, species)
    x = torch.from_numpy(data["x"]).float()
    edge_index = torch.from_numpy(data["edge_index_undirected"]).long()
    node_ids = data["node_ids"].astype(str)
    return {"x": x, "edge_index": edge_index, "node_ids": node_ids}


def resolve_node_id(split: str, species: str, query_text: str, kind: str) -> dict[str, Any]:
    df = load_nodes_df(split, species)
    subset = df.copy()
    if kind in {"mirna", "mrna"}:
        subset = subset[subset["node_type"].eq(kind)]
    q = query_text.strip().lower()
    if not q:
        raise HTTPException(400, "query is required")
    exact = subset[
        subset["node_id"].astype(str).str.lower().eq(q) | subset["source_name"].astype(str).str.lower().eq(q)
    ]
    if exact.empty:
        exact = subset[
            subset["node_id"].astype(str).str.lower().str.contains(q, na=False)
            | subset["source_name"].astype(str).str.lower().str.contains(q, na=False)
        ]
    if exact.empty:
        return {"found": False, "query": query_text, "message": "Node not found in current graph"}
    row = exact.iloc[0]
    return {
        "found": True,
        "query": query_text,
        "node_id": str(row["node_id"]),
        "name": str(row["source_name"]),
        "idx": int(row["node_idx"]),
        "type": str(row["node_type"]),
    }


def get_calibration_params(run_dir: Path, source: str, target: str) -> dict[str, float]:
    path = run_dir / "diagnostics" / "diagnostic_summary.json"
    if not path.exists():
        return {"temperature": 1.0, "bias": 0.0}
    data = read_json(path)
    for item in data:
        if item["source"] == source and item["target"] == target:
            return {
                "temperature": float(item["platt_scaling"]["temperature"]),
                "bias": float(item["platt_scaling"]["bias"]),
                "threshold": float(item["raw"]["best_threshold_from_val"]),
            }
    return {"temperature": 1.0, "bias": 0.0, "threshold": 0.5}


def apply_calibration(logit: float, temperature: float, bias: float) -> float:
    return float(1.0 / (1.0 + math.exp(-((logit / temperature) + bias))))


@app.post("/api/predict/pair")
def api_predict_pair(payload: dict[str, Any]) -> dict[str, Any]:
    species = str(payload.get("species", "human"))
    source = str(payload.get("source_model", species))
    run_id = str(payload.get("run_id") or latest_dir(TRANSFER_ROOT).name)
    split = str(payload.get("split", "cold_mirna"))
    mirna_query = str(payload.get("mirna", "")).strip()
    mrna_query = str(payload.get("mrna", "")).strip()
    device = torch.device("cpu")

    if species not in SPECIES:
        raise HTTPException(400, f"Unknown species: {species}")
    mirna = resolve_node_id(split, species, mirna_query, "mirna")
    mrna = resolve_node_id(split, species, mrna_query, "mrna")
    if not mirna["found"] or not mrna["found"]:
        return {
            "species": species,
            "source_model": source,
            "target_species": species,
            "mirna": mirna,
            "mrna": mrna,
            "found": False,
            "message": "Node not found in current graph",
        }

    model, config, run_dir = load_transfer_model(run_id, source, device)
    graph = load_species_graph(split, species)
    x = graph["x"]
    edge_index = graph["edge_index"]
    with torch.no_grad():
        z = model.encode(x, edge_index)
        edge_label_index = torch.tensor([[mirna["idx"]], [mrna["idx"]]], dtype=torch.long)
        logit = float(model.decode(z, edge_label_index).item())
        raw_score = float(torch.sigmoid(torch.tensor(logit)).item())
    calib = get_calibration_params(run_dir, source, species)
    calibrated_score = apply_calibration(logit, calib.get("temperature", 1.0), calib.get("bias", 0.0))
    threshold = calib.get("threshold", 0.5)
    label = "Positive interaction" if calibrated_score >= threshold else "Negative interaction"
    confidence = "High" if calibrated_score >= 0.82 else "Medium" if calibrated_score >= 0.68 else "Low"
    return {
        "species": species,
        "source_model": source,
        "target_species": species,
        "mirna": mirna,
        "mrna": mrna,
        "raw_logit": logit,
        "raw_score": raw_score,
        "calibrated_score": calibrated_score,
        "threshold": threshold,
        "label": label,
        "confidence": confidence,
        "calibration": calib,
    }


@app.get("/api/predict/topk")
def api_predict_topk(
    species: str = Query("human"),
    source_model: str | None = None,
    run_id: str | None = None,
    mirna: str = Query(""),
    split: str = Query("cold_mirna"),
    k: int = Query(20, ge=1, le=100),
):
    if species not in SPECIES:
        raise HTTPException(400, f"Unknown species: {species}")
    source = source_model or species
    run_id = run_id or latest_dir(TRANSFER_ROOT).name
    device = torch.device("cpu")
    model, _, run_dir = load_transfer_model(run_id, source, device)
    mirna_node = resolve_node_id(split, species, mirna, "mirna")
    if not mirna_node["found"]:
        raise HTTPException(404, "miRNA not found")
    graph = load_species_graph(split, species)
    nodes_df = load_nodes_df(split, species)
    mrna_nodes = nodes_df[nodes_df["node_type"].eq("mrna")].copy()
    with torch.no_grad():
        z = model.encode(graph["x"], graph["edge_index"])
        src_idx = torch.tensor([mirna_node["idx"]] * len(mrna_nodes), dtype=torch.long)
        dst_idx = torch.tensor(mrna_nodes["node_idx"].to_numpy(), dtype=torch.long)
        logits = model.decode(z, torch.stack([src_idx, dst_idx], dim=0))
        probs = torch.sigmoid(logits).detach().cpu().numpy()
    calib = get_calibration_params(run_dir, source, species)
    calibrated = np.array([apply_calibration(float(logit), calib.get("temperature", 1.0), calib.get("bias", 0.0)) for logit in logits.detach().cpu().numpy()])
    mrna_nodes["score"] = probs
    mrna_nodes["calibrated_score"] = calibrated
    mrna_nodes = mrna_nodes.sort_values("score", ascending=False).head(k)
    items = []
    for rank, row in enumerate(mrna_nodes.itertuples(index=False), start=1):
        items.append(
            {
                "rank": rank,
                "mirna": mirna_node["name"],
                "mrna": str(row.source_name),
                "mrna_id": str(row.node_id),
                "score": float(row.score),
                "calibrated_score": float(row.calibrated_score),
            }
        )
    return {"species": species, "source_model": source, "mirna": mirna_node, "items": items}


@app.get("/api/network")
def api_network(
    species: str = Query("human"),
    mirna: str = Query(""),
    source_model: str | None = None,
    run_id: str | None = None,
    split: str = Query("cold_mirna"),
    threshold: float = Query(0.5),
    limit: int = Query(40, ge=1, le=100),
):
    topk = api_predict_topk(species=species, source_model=source_model, run_id=run_id, mirna=mirna, split=split, k=limit)
    nodes_df = load_nodes_df(split, species)
    pos_edges = load_positive_edges_df(split, species)
    center = topk["mirna"]
    selected_mrna_ids = {
        item["mrna_id"] for item in topk["items"] if float(item.get("calibrated_score", item["score"])) >= threshold
    }
    edges = pos_edges[pos_edges["mrna_id"].astype(str).isin(selected_mrna_ids) & pos_edges["mirna_id"].astype(str).eq(center["node_id"])]
    if edges.empty:
        edges = pos_edges[pos_edges["mirna_id"].astype(str).eq(center["node_id"])].head(limit)
    targets = nodes_df[nodes_df["node_id"].astype(str).isin(set(edges["mrna_id"].astype(str)))].copy()
    nodes = [
        {"id": str(center["node_id"]), "label": str(center["name"]), "type": "mirna"},
        *[
            {"id": str(row.node_id), "label": str(row.source_name), "type": str(row.node_type)}
            for row in targets.itertuples(index=False)
        ],
    ]
    return {
        "center": {"id": center["node_id"], "label": center["name"], "type": "mirna"},
        "nodes": nodes,
        "edges": [
            {
                "source": str(row.mirna_id),
                "target": str(row.mrna_id),
                "score": None,
                "type": "known",
            }
            for row in edges.itertuples(index=False)
        ],
        "threshold": threshold,
    }


app.mount("/", StaticFiles(directory=ROOT, html=True), name="ui")
