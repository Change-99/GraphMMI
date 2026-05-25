#!/usr/bin/env python3
"""Local UI server for the GraphMMI prototype.

It serves the static UI and exposes a small local-only API that can launch
training scripts with whitelisted form parameters.
"""

from __future__ import annotations

import json
import os
import errno
import csv
import hashlib
import shlex
import subprocess
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


UI_DIR = Path(__file__).resolve().parent
ROOT = UI_DIR.parent
RUN_ROOT = ROOT / "runs" / "ui_experiments"
LOG_ROOT = UI_DIR / "jobs"
PREDICTION_ROOT = ROOT / "runs" / "ui_predictions"
PREDICTION_DATA_ROOT = ROOT / "data" / "processed" / "graph" / "final_target_site"
MAX_LOG_CHARS = 24000

SPECIES = {"human", "cow", "mouse", "worm"}
MODELS = {"GraphSAGE": "graphsage", "GATv2": "gatv2"}
SETTINGS = {"source_only": "strict_zero_shot", "transfer": "finetune"}
SIM_FLAGS = {
    "no-sim": [],
    "miRNA-only": ["--mirna-sim-edges"],
    "target-only": ["--mrna-sim-edges"],
    "both-sim": ["--mirna-sim-edges", "--mrna-sim-edges"],
}
NEGATIVE_STRATEGIES = {"endpoint_corrupt", "degree_aware", "sequence_aware", "random", "uniform"}
RUN_RESULT_FILENAMES = {"transfer_metrics.csv", "decoder_ablation.csv", "metrics_long.csv"}
HEATMAP_ROOTS = {"runs": ROOT / "runs", "final_exp": ROOT / "final_exp"}
DEMO_MODELS = [
    {
        "id": "graphsage_bothsim_l4",
        "label": "graphsage_bothsim_l4",
        "model_type": "graphsage",
        "source": "demo",
        "mode": "demo_scoring",
    },
    {
        "id": "gatv2_bothsim_l3",
        "label": "gatv2_bothsim_l3",
        "model_type": "gatv2",
        "source": "demo",
        "mode": "demo_scoring",
    },
    {
        "id": "xgb_baseline",
        "label": "xgb_baseline",
        "model_type": "xgb",
        "source": "demo",
        "mode": "demo_scoring",
    },
]

JOBS: dict[str, dict[str, object]] = {}


def _fmt(value: str | float) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _format_int(value: object) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def dataset_summary_payload() -> dict:
    rows = []
    totals = {
        "samples": 0,
        "nodes": 0,
        "mirna": 0,
        "targets": 0,
        "positive_edges": 0,
        "mirna_sim_edges": 0,
        "target_sim_edges": 0,
    }
    for name in ["human", "cow", "mouse", "worm"]:
        root = PREDICTION_DATA_ROOT / name
        meta_path = root / "metadata.json"
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
        split = meta.get("split_counts", {})
        cleaning = meta.get("cleaning", {})
        files = [
            {"name": "metadata.json", "path": str(meta_path), "exists": meta_path.exists()},
            {"name": "nodes.csv", "path": str(root / "nodes.csv"), "exists": (root / "nodes.csv").exists()},
            {"name": "positive_edges.csv", "path": str(root / "positive_edges.csv"), "exists": (root / "positive_edges.csv").exists()},
            {"name": "graph_inputs.pt", "path": str(root / "graph_inputs.pt"), "exists": (root / "graph_inputs.pt").exists()},
            {"name": "graph_inputs.npz", "path": str(root / "graph_inputs.npz"), "exists": (root / "graph_inputs.npz").exists()},
        ]
        row = {
            "species": name,
            "samples": int(cleaning.get("clean_positive_edges", meta.get("num_positive_edges", 0))),
            "raw_rows": int(cleaning.get("raw_rows", 0)),
            "duplicate_removed": int(cleaning.get("duplicate_pair_rows_removed", 0)),
            "node_conflicts_removed": int(cleaning.get("dropped_node_sequence_conflicts", 0)),
            "nodes": int(meta.get("num_nodes", 0)),
            "mirna": int(meta.get("num_mirna_nodes", 0)),
            "targets": int(meta.get("num_target_site_nodes", 0)),
            "positive_edges": int(meta.get("num_positive_edges", 0)),
            "mirna_sim_edges": int(meta.get("num_mirna_sim_edges", 0)),
            "target_sim_edges": int(meta.get("num_mrna_sim_edges", 0)),
            "node_features": int(meta.get("num_node_features", 0)),
            "edge_attr": int(meta.get("num_edge_attr", 0)),
            "pair_feature_dim": int(meta.get("pair_feature_dim", 0)),
            "node_mode": meta.get("node_mode", ""),
            "sim_mode": meta.get("sim_mode", ""),
            "mirna_sim_topk": meta.get("mirna_sim_topk", 0),
            "target_sim_topk": meta.get("mrna_sim_topk", 0),
            "split": {
                "train": int(split.get("train", 0)),
                "val": int(split.get("val", 0)),
                "test": int(split.get("test", 0)),
            },
            "diagnostic": meta.get("split_diagnostic", {}),
            "files": files,
            "status": "ready" if all(item["exists"] for item in files[:4]) else "incomplete",
            "root": str(root),
        }
        rows.append(row)
        for key in totals:
            totals[key] += int(row[key])
    return {
        "rows": rows,
        "totals": totals,
        "cards": [
            {"label": "物种数", "value": len(rows), "hint": "human / cow / mouse / worm"},
            {"label": "正样本总数", "value": _format_int(totals["samples"]), "hint": "clean positive edges"},
            {"label": "图节点总数", "value": _format_int(totals["nodes"]), "hint": "miRNA + target site"},
            {"label": "相似边总数", "value": _format_int(totals["mirna_sim_edges"] + totals["target_sim_edges"]), "hint": "miRNA + target site top-k"},
        ],
        "source": str(PREDICTION_DATA_ROOT),
    }


def transfer_summary_payload() -> dict:
    path = ROOT / "final_exp" / "exp2" / "summary_metrics.csv"
    rows = _read_csv(path)
    return {
        "columns": ["encoder", "setting", "AUPRavg", "AUPRdiag", "AUPRcross", "AUCavg", "F1avg", "MCCavg"],
        "rows": [
            [
                row["encoder"],
                row["setting"],
                _fmt(row["aupr_mean"]),
                _fmt(row["aupr_diag_mean"]),
                _fmt(row["aupr_cross_species_mean"]),
                _fmt(row["auc_mean"]),
                _fmt(row["f1_mean"]),
                _fmt(row["mcc_mean"]),
            ]
            for row in rows
        ],
        "source": str(path),
    }


def edge_ablation_payload() -> dict:
    base = ROOT / "final_exp" / "exp3" / "A_edges" / "result"
    configs = [
        ("no-sim", "×", "×"),
        ("miRNA-only", "√", "×"),
        ("target-only", "×", "√"),
        ("both-sim", "√", "√"),
    ]
    rows = []
    sources = []
    for name, mirna_edge, target_edge in configs:
        path = base / name / "transfer_metrics.csv"
        metrics = _read_csv(path)
        auprs = [float(row["aupr"]) for row in metrics]
        cross = [float(row["aupr"]) for row in metrics if row["source"] != row["target"]]
        aucs = [float(row["auc"]) for row in metrics]
        rows.append([name, mirna_edge, target_edge, _fmt(_mean(auprs)), _fmt(_mean(cross)), _fmt(_mean(aucs))])
        sources.append(str(path))
    return {
        "columns": ["实验设置", "miRNA相似边", "target site相似边", "AUPRavg", "AUPRcross", "AUCavg"],
        "rows": rows,
        "source": "; ".join(sources),
    }


def decoder_ablation_payload() -> dict:
    path = ROOT / "final_exp" / "exp3" / "C_decoder" / "result" / "20260519-114534" / "decoder_ablation.csv"
    designs = {
        "baseline": "concat + MLP",
        "residual": "MLP 中加入残差连接",
        "gated": "pair feature 门控融合",
        "bilinear": "双线性匹配项",
        "separated": "节点特征与 pair feature 分别编码后融合",
    }
    rows = _read_csv(path)
    return {
        "columns": ["decoder 结构", "主要设计", "test AUC", "test AUPR", "最佳 epoch"],
        "rows": [
            [
                row["decoder"],
                designs.get(row["decoder"], ""),
                _fmt(row["test_auc"]),
                _fmt(row["test_aupr"]),
                row["best_epoch"],
            ]
            for row in rows
        ],
        "source": str(path),
    }


def result_payload(kind: str) -> dict:
    if kind == "transfer":
        return transfer_summary_payload()
    if kind == "edges":
        return edge_ablation_payload()
    if kind == "decoder":
        return decoder_ablation_payload()
    raise ValueError(f"Unknown result table: {kind}")


def overview_summary_payload() -> dict:
    dataset = dataset_summary_payload()
    transfer = transfer_summary_payload()
    edges = edge_ablation_payload()
    decoder = decoder_ablation_payload()
    run_files = runs_result_files_payload().get("files", [])
    heatmap_files = heatmap_files_payload().get("files", [])
    models = predict_models_payload().get("models", [])

    best_decoder = max(decoder["rows"], key=lambda row: float(row[3]))
    best_transfer = max(transfer["rows"], key=lambda row: float(row[4]))
    best_edge = max(edges["rows"], key=lambda row: float(row[4]))
    totals = dataset["totals"]
    cards = [
        {"label": "数据物种数", "value": len(dataset["rows"]), "hint": "human / cow / mouse / worm"},
        {"label": "正样本总数", "value": _format_int(totals["samples"]), "hint": "clean positive edges"},
        {"label": "最高 AUPR", "value": _fmt(best_decoder[3]), "hint": f"decoder={best_decoder[0]}, epoch={best_decoder[4]}"},
        {"label": "最佳跨物种 AUPR", "value": _fmt(best_edge[4]), "hint": f"{best_edge[0]} edge ablation"},
    ]
    modules = [
        {"name": "数据管理", "status": "ready", "detail": f"{_format_int(totals['nodes'])} nodes / {_format_int(totals['samples'])} positives", "page": "datasets"},
        {"name": "模型训练", "status": "local", "detail": "GraphSAGE、GATv2 训练脚本可调度", "page": "training"},
        {"name": "实验结果", "status": "ready", "detail": f"{len(run_files)} 个 runs 结果 CSV 可浏览", "page": "results"},
        {"name": "热力图", "status": "ready", "detail": f"{len(heatmap_files)} 个 transfer_metrics.csv 可选", "page": "visualization"},
        {"name": "相互作用预测", "status": "demo", "detail": f"{len(models)} 个模型/权重条目已注册", "page": "prediction"},
    ]
    return {
        "cards": cards,
        "modules": modules,
        "best": [
            {"label": "Decoder 最优", "value": _fmt(best_decoder[3]), "detail": f"{best_decoder[0]} AUPR, AUC={best_decoder[2]}"},
            {"label": "迁移汇总最优", "value": _fmt(best_transfer[4]), "detail": f"{best_transfer[0]} {best_transfer[1]} cross AUPR"},
            {"label": "相似边消融最优", "value": _fmt(best_edge[4]), "detail": f"{best_edge[0]} cross AUPR, AUC={best_edge[5]}"},
        ],
        "recent": [
            {
                "label": item["label"],
                "path": item["path"],
                "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(item["mtime"]))),
            }
            for item in run_files[:5]
        ],
        "source": str(ROOT),
    }


def runs_result_files_payload() -> dict:
    runs_root = ROOT / "runs"
    files = []
    if runs_root.exists():
        for path in runs_root.rglob("*.csv"):
            if path.name not in RUN_RESULT_FILENAMES:
                continue
            rel = path.relative_to(runs_root).as_posix()
            files.append(
                {
                    "id": rel,
                    "label": rel,
                    "path": str(path),
                    "mtime": path.stat().st_mtime,
                }
            )
    files.sort(key=lambda item: float(item["mtime"]), reverse=True)
    return {"files": files, "root": str(runs_root)}


def _safe_runs_csv(relative_path: str) -> Path:
    runs_root = (ROOT / "runs").resolve()
    path = (runs_root / relative_path).resolve()
    try:
        path.relative_to(runs_root)
    except ValueError as exc:
        raise ValueError("Result path must stay under GraphMMI/runs.") from exc
    if path.name not in RUN_RESULT_FILENAMES or path.suffix.lower() != ".csv":
        raise ValueError("Only known run result CSV files can be opened.")
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def run_result_payload(relative_path: str) -> dict:
    path = _safe_runs_csv(relative_path)
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []
        raw_rows = list(reader)
    max_rows = 300
    rows = [[_fmt(row.get(column, "")) for column in columns] for row in raw_rows[:max_rows]]
    return {
        "columns": columns,
        "rows": rows,
        "source": str(path),
        "total_rows": len(raw_rows),
        "truncated": len(raw_rows) > max_rows,
    }


def heatmap_files_payload() -> dict:
    files = []
    for prefix, root in HEATMAP_ROOTS.items():
        if not root.exists():
            continue
        for path in root.rglob("transfer_metrics.csv"):
            rel = path.relative_to(root).as_posix()
            rows = _read_csv(path)
            encoders = sorted({row.get("encoder", "") for row in rows if row.get("encoder")})
            settings = sorted({row.get("setting", "") for row in rows if row.get("setting")})
            species = sorted(
                {row.get("source", "") for row in rows if row.get("source")}
                | {row.get("target", "") for row in rows if row.get("target")}
            )
            files.append(
                {
                    "id": f"{prefix}/{rel}",
                    "label": f"{prefix}/{rel} ({len(species)}×{len(species)})",
                    "path": str(path),
                    "encoders": encoders,
                    "settings": settings,
                    "species_count": len(species),
                    "pair_count": len(rows),
                    "mtime": path.stat().st_mtime,
                }
            )
    files.sort(
        key=lambda item: (int(item["species_count"]), int(item["pair_count"]), float(item["mtime"])),
        reverse=True,
    )
    return {"files": files}


def _safe_heatmap_csv(identifier: str) -> Path:
    prefix, _, rel = identifier.partition("/")
    if prefix not in HEATMAP_ROOTS or not rel:
        raise ValueError("Heatmap file id must start with runs/ or final_exp/.")
    root = HEATMAP_ROOTS[prefix].resolve()
    path = (root / rel).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("Heatmap path must stay under its result root.") from exc
    if path.name != "transfer_metrics.csv":
        raise ValueError("Heatmap data must come from transfer_metrics.csv.")
    if not path.exists():
        raise FileNotFoundError(str(path))
    return path


def heatmap_payload(identifier: str, metric: str, encoder: str, setting: str) -> dict:
    metric = metric.lower()
    if metric not in {"aupr", "auc", "f1", "mcc", "acc"}:
        raise ValueError("Unsupported heatmap metric.")
    path = _safe_heatmap_csv(identifier)
    rows = _read_csv(path)
    if not rows:
        raise ValueError("Selected transfer_metrics.csv is empty.")
    encoders = sorted({row.get("encoder", "") for row in rows if row.get("encoder")})
    settings = sorted({row.get("setting", "") for row in rows if row.get("setting")})
    selected_encoder = encoder if encoder in encoders else (encoders[0] if encoders else "")
    selected_setting = setting if setting in settings else (settings[0] if settings else "")
    filtered = [
        row for row in rows
        if (not selected_encoder or row.get("encoder") == selected_encoder)
        and (not selected_setting or row.get("setting") == selected_setting)
    ]
    if not filtered:
        raise ValueError("No rows match the selected encoder/setting.")
    species_order = ["human", "cow", "mouse", "worm"]
    present = sorted({row["source"] for row in filtered} | {row["target"] for row in filtered})
    species = [name for name in species_order if name in present] + [name for name in present if name not in species_order]
    by_pair = {(row["source"], row["target"]): float(row[metric]) for row in filtered if row.get(metric)}
    matrix = [[by_pair.get((src, dst)) for dst in species] for src in species]
    return {
        "species": species,
        "matrix": matrix,
        "metric": metric,
        "encoder": selected_encoder,
        "setting": selected_setting,
        "encoders": encoders,
        "settings": settings,
        "source": str(path),
    }


def predict_models_payload() -> dict:
    models = list(DEMO_MODELS)
    for root in [ROOT / "runs", ROOT / "final_exp"]:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".pt", ".json", ".h5"}:
                continue
            if "models" not in path.parts:
                continue
            rel = path.relative_to(ROOT).as_posix()
            model_type = path.parent.name
            models.append(
                {
                    "id": rel,
                    "label": rel,
                    "model_type": model_type,
                    "source": str(path),
                    "mode": "registered_artifact",
                }
            )
    return {"models": models}


def _require_prediction_text(payload: dict, key: str, label: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"{label} cannot be empty.")
    return value


def _demo_probability(model_id: str, species: str, mirna: str, target: str, sequence: str) -> float:
    clean_sequence = "".join(base for base in sequence.upper() if base in "ACGUT")
    gc_count = clean_sequence.count("G") + clean_sequence.count("C")
    gc_ratio = gc_count / len(clean_sequence) if clean_sequence else 0.0
    seed = f"{model_id}|{species}|{mirna}|{target}|{clean_sequence}".encode("utf-8")
    digest = int(hashlib.sha256(seed).hexdigest()[:8], 16) / 0xFFFFFFFF
    score = 0.18 + 0.58 * digest + 0.22 * gc_ratio
    if model_id.startswith("graphsage"):
        score += 0.04
    elif model_id.startswith("gatv2"):
        score += 0.02
    return max(0.001, min(0.999, score))


def _prediction_result(payload: dict) -> dict:
    model_id = _require_prediction_text(payload, "model_id", "model_id")
    species = _require(payload.get("species"), SPECIES, "species")
    mirna = _require_prediction_text(payload, "mirna_name", "mirna_name")
    target = _require_prediction_text(payload, "target_site_id", "target_site_id")
    sequence = _require_prediction_text(payload, "target_sequence", "target_sequence")
    probability = _demo_probability(model_id, species, mirna, target, sequence)
    return {
        "model_id": model_id,
        "species": species,
        "mirna_name": mirna,
        "target_site_id": target,
        "target_sequence": sequence,
        "interaction_probability": round(probability, 6),
        "prediction_label": 1 if probability >= 0.5 else 0,
        "mode": "demo_scoring",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _append_prediction_rows(rows: list[dict]) -> Path:
    PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    path = PREDICTION_ROOT / "predictions.csv"
    columns = [
        "created_at",
        "model_id",
        "species",
        "mirna_name",
        "target_site_id",
        "target_sequence",
        "interaction_probability",
        "prediction_label",
        "mode",
    ]
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def predict_single_payload(payload: dict) -> dict:
    result = _prediction_result(payload)
    path = _append_prediction_rows([result])
    result["result_file"] = str(path)
    return result


def predict_batch_payload(payload: dict) -> dict:
    model_id = _require_prediction_text(payload, "model_id", "model_id")
    species = _require(payload.get("species"), SPECIES, "species")
    items = payload.get("items", [])
    if not isinstance(items, list) or not items:
        raise ValueError("items must be a non-empty list.")
    results = []
    for item in items[:500]:
        if not isinstance(item, dict):
            continue
        merged = {
            "model_id": model_id,
            "species": species,
            "mirna_name": item.get("mirna_name", item.get("mirna", "")),
            "target_site_id": item.get("target_site_id", item.get("target", "")),
            "target_sequence": item.get("target_sequence", item.get("sequence", "")),
        }
        results.append(_prediction_result(merged))
    if not results:
        raise ValueError("No valid batch rows were provided.")
    results.sort(key=lambda row: float(row["interaction_probability"]), reverse=True)
    path = _append_prediction_rows(results)
    return {
        "job_id": f"pred_{time.strftime('%Y%m%d_%H%M%S')}",
        "count": len(results),
        "results": results,
        "result_file": str(path),
        "mode": "demo_scoring",
    }


def prediction_history_payload(limit: int = 100) -> dict:
    path = PREDICTION_ROOT / "predictions.csv"
    if not path.exists():
        return {"columns": [], "rows": [], "source": str(path)}
    rows = _read_csv(path)
    rows = rows[-limit:][::-1]
    return {
        "columns": ["mirna_name", "target_site_id", "interaction_probability", "prediction_label"],
        "rows": [
            [
                row.get("mirna_name", ""),
                row.get("target_site_id", ""),
                _fmt(row.get("interaction_probability", "")),
                row.get("prediction_label", ""),
            ]
            for row in rows
        ],
        "source": str(path),
    }


def predict_options_payload(species: str) -> dict:
    species = _require(species, SPECIES, "species")
    path = PREDICTION_DATA_ROOT / species / "nodes.csv"
    rows = _read_csv(path)
    mirnas = []
    targets = []
    for row in rows:
        node_type = row.get("node_type", "")
        item = {
            "id": row.get("node_id", ""),
            "name": row.get("source_name", "") or row.get("node_id", ""),
            "sequence": row.get("target_seq", "") or row.get("sequence", ""),
        }
        if not item["id"]:
            continue
        if node_type == "mirna":
            mirnas.append(item)
        elif node_type == "target_site":
            targets.append(item)
    return {
        "species": species,
        "mirnas": mirnas,
        "targets": targets,
        "source": str(path),
    }


def _json_response(handler: SimpleHTTPRequestHandler, status: HTTPStatus, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _csv_response(handler: SimpleHTTPRequestHandler, path: Path) -> None:
    if not path.exists():
        _json_response(handler, HTTPStatus.NOT_FOUND, {"error": "prediction file not found"})
        return
    body = path.read_bytes()
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/csv; charset=utf-8")
    handler.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _bad_request(handler: SimpleHTTPRequestHandler, message: str) -> None:
    _json_response(handler, HTTPStatus.BAD_REQUEST, {"error": message})


def _require(value: object, allowed: set[str], field: str) -> str:
    text = str(value)
    if text not in allowed:
        raise ValueError(f"Invalid {field}: {text}")
    return text


def build_command(payload: dict) -> list[str]:
    model_label = str(payload.get("model", ""))
    if model_label not in MODELS:
        raise ValueError("Local launcher currently supports GraphSAGE and GATv2.")

    source = _require(payload.get("source"), SPECIES, "source")
    target = _require(payload.get("target"), SPECIES, "target")
    setting = _require(payload.get("setting"), set(SETTINGS), "setting")
    sim = _require(payload.get("sim"), set(SIM_FLAGS), "sim")
    negative = _require(payload.get("negative"), NEGATIVE_STRATEGIES, "negative")

    try:
        layers = int(payload.get("layers", 4))
    except (TypeError, ValueError) as exc:
        raise ValueError("layers must be an integer") from exc
    if layers < 1 or layers > 6:
        raise ValueError("layers must be between 1 and 6")

    encoder = MODELS[model_label]
    species_args = [source] if source == target else [source, target]
    hidden_args = ["--graphsage-hidden-dim", "128"] if encoder == "graphsage" else ["--gatv2-hidden-dim", "64"]

    command = [
        "python",
        "-u",
        str(ROOT / "scripts" / "train_gnn_transfer.py"),
        "--species",
        *species_args,
        "--encoders",
        encoder,
        "--settings",
        SETTINGS[setting],
        "--epochs",
        "40",
        "--patience",
        "8",
        "--finetune-epochs",
        "15",
        "--finetune-patience",
        "5",
        "--num-layers",
        str(layers),
        *hidden_args,
        "--processed-dir",
        str(ROOT / "data" / "processed" / "graph" / "final_target_site"),
        *SIM_FLAGS[sim],
        "--skip-preprocess",
        "--run-root",
        str(RUN_ROOT),
        "--no-heatmaps",
        "--neg-strategy",
        negative,
        "--eval-neg-strategy",
        "endpoint_corrupt",
    ]
    return command


def format_command(command: list[str]) -> str:
    parts = [shlex.quote(item) for item in command]
    lines = [" ".join(parts[:3]) + " \\"]
    index = 3
    while index < len(parts):
        group = [parts[index]]
        index += 1
        while index < len(parts) and not parts[index].startswith("--"):
            group.append(parts[index])
            index += 1
        suffix = " \\" if index < len(parts) else ""
        lines.append("  " + " ".join(group) + suffix)
    return "\n".join(lines)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/health":
            _json_response(self, HTTPStatus.OK, {"ok": True, "run_root": str(RUN_ROOT)})
            return
        if path == "/api/overview/summary":
            try:
                _json_response(self, HTTPStatus.OK, overview_summary_payload())
            except (FileNotFoundError, KeyError, ValueError) as exc:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        if path == "/api/datasets/summary":
            try:
                _json_response(self, HTTPStatus.OK, dataset_summary_payload())
            except (FileNotFoundError, KeyError, ValueError) as exc:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        if path == "/api/predict/models":
            _json_response(self, HTTPStatus.OK, predict_models_payload())
            return
        if path == "/api/predict/options":
            query = parse_qs(parsed.query)
            try:
                _json_response(self, HTTPStatus.OK, predict_options_payload(query.get("species", ["human"])[0]))
            except (FileNotFoundError, ValueError) as exc:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        if path == "/api/predict/history":
            query = parse_qs(parsed.query)
            try:
                limit = int(query.get("limit", ["100"])[0])
            except ValueError:
                limit = 100
            _json_response(self, HTTPStatus.OK, prediction_history_payload(max(1, min(limit, 500))))
            return
        if path.startswith("/api/predict/") and path.endswith("/download"):
            _csv_response(self, PREDICTION_ROOT / "predictions.csv")
            return
        if path == "/api/visualization/heatmaps":
            query = parse_qs(parsed.query)
            selected = query.get("file", [""])[0]
            try:
                payload = heatmap_payload(
                    selected,
                    query.get("metric", ["aupr"])[0],
                    query.get("encoder", [""])[0],
                    query.get("setting", [""])[0],
                ) if selected else heatmap_files_payload()
                _json_response(self, HTTPStatus.OK, payload)
            except (FileNotFoundError, KeyError, ValueError) as exc:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        if path == "/api/results/runs":
            query = parse_qs(parsed.query)
            selected = query.get("file", [""])[0]
            try:
                payload = run_result_payload(selected) if selected else runs_result_files_payload()
                _json_response(self, HTTPStatus.OK, payload)
            except (FileNotFoundError, ValueError) as exc:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        if path.startswith("/api/results/"):
            kind = path.split("/")[3] if len(path.split("/")) > 3 else ""
            try:
                _json_response(self, HTTPStatus.OK, result_payload(kind))
            except (FileNotFoundError, KeyError, ValueError) as exc:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": str(exc)})
            return
        if path.startswith("/api/experiments/") and path.endswith("/status"):
            job_id = path.split("/")[3]
            job = JOBS.get(job_id)
            if not job:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "job not found"})
                return
            process = job["process"]
            assert isinstance(process, subprocess.Popen)
            if process.poll() is None:
                status = "running"
            elif process.returncode == 0:
                status = "completed"
            else:
                status = "failed"
            payload = {key: value for key, value in job.items() if key != "process"}
            payload["status"] = status
            payload["returncode"] = process.returncode
            _json_response(self, HTTPStatus.OK, payload)
            return
        if path.startswith("/api/experiments/") and path.endswith("/log"):
            job_id = path.split("/")[3]
            job = JOBS.get(job_id)
            if not job:
                _json_response(self, HTTPStatus.NOT_FOUND, {"error": "job not found"})
                return
            log_path = Path(str(job["log_path"]))
            if not log_path.exists():
                _json_response(self, HTTPStatus.OK, {"experiment_id": job_id, "log": ""})
                return
            with log_path.open("rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - MAX_LOG_CHARS))
                text = f.read().decode("utf-8", errors="replace")
            if size > MAX_LOG_CHARS:
                text = "[showing last log lines]\n" + text
            _json_response(self, HTTPStatus.OK, {"experiment_id": job_id, "log": text})
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path in {"/api/predict/single", "/api/predict/batch"}:
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                result = predict_single_payload(payload) if path.endswith("/single") else predict_batch_payload(payload)
                _json_response(self, HTTPStatus.OK, result)
            except (json.JSONDecodeError, ValueError) as exc:
                _bad_request(self, str(exc))
            return

        if path != "/api/experiments":
            _json_response(self, HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            command = build_command(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            _bad_request(self, str(exc))
            return

        RUN_ROOT.mkdir(parents=True, exist_ok=True)
        LOG_ROOT.mkdir(parents=True, exist_ok=True)
        job_id = f"exp_{time.strftime('%Y%m%d_%H%M%S')}"
        log_path = LOG_ROOT / f"{job_id}.log"
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        log_file.close()
        JOBS[job_id] = {
            "experiment_id": job_id,
            "status": "running",
            "pid": process.pid,
            "command": command,
            "command_text": format_command(command),
            "log_path": str(log_path),
            "run_root": str(RUN_ROOT),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "process": process,
        }
        payload = {key: value for key, value in JOBS[job_id].items() if key != "process"}
        _json_response(self, HTTPStatus.ACCEPTED, payload)


def main() -> None:
    host = "127.0.0.1"
    start_port = int(os.environ.get("GRAPHMMI_UI_PORT", "8000"))
    server = None
    port = start_port
    for candidate in range(start_port, start_port + 20):
        try:
            server = ThreadingHTTPServer((host, candidate), Handler)
            port = candidate
            break
        except OSError as exc:
            if exc.errno not in {errno.EADDRINUSE, 98, 48}:
                raise
            print(f"Port {candidate} is busy, trying {candidate + 1}...")
    if server is None:
        raise RuntimeError(
            f"Could not bind a local UI server on ports {start_port}-{start_port + 19}. "
            "Set GRAPHMMI_UI_PORT=8080 or stop the process using the port."
        )
    print(f"GraphMMI UI server: http://{host}:{port}")
    print("Use this server, not python -m http.server, when you want training buttons to launch jobs.")
    server.serve_forever()


if __name__ == "__main__":
    main()
