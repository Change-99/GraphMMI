#!/usr/bin/env python3
"""Local UI server for the GraphMMI prototype.

It serves the static UI and exposes a small local-only API that can launch
training scripts with whitelisted form parameters.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


UI_DIR = Path(__file__).resolve().parent
ROOT = UI_DIR.parent
RUN_ROOT = ROOT / "runs" / "ui_experiments"
LOG_ROOT = UI_DIR / "jobs"

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

JOBS: dict[str, dict[str, object]] = {}


def _json_response(handler: SimpleHTTPRequestHandler, status: HTTPStatus, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
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
        path = urlparse(self.path).path
        if path == "/api/health":
            _json_response(self, HTTPStatus.OK, {"ok": True, "run_root": str(RUN_ROOT)})
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
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
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
    port = int(os.environ.get("GRAPHMMI_UI_PORT", "8000"))
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"GraphMMI UI server: http://{host}:{port}")
    print("Use this server, not python -m http.server, when you want training buttons to launch jobs.")
    server.serve_forever()


if __name__ == "__main__":
    main()
