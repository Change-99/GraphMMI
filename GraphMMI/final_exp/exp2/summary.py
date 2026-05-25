#!/usr/bin/env python3
"""Summarize Exp2 GraphSAGE and GATv2 transfer metrics."""

from __future__ import annotations

import csv
import math
from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
RESULT_DIR = EXP_DIR / "result"
OUTPUT_CSV = EXP_DIR / "summary_metrics.csv"

RUNS = {
    "graphsage": RESULT_DIR / "20260518-170917",
    "gatv2": RESULT_DIR / "20260518-172922",
}

SETTING_LABELS = {
    "strict_zero_shot": "source_only",
    "finetune": "transfer",
}

METRIC_COLUMNS = ("aupr", "auc", "f1", "mcc")
SUMMARY_COLUMNS = (
    "encoder",
    "setting",
    "aupr_mean",
    "aupr_diag_mean",
    "aupr_cross_species_mean",
    "auc_mean",
    "f1_mean",
    "mcc_mean",
)


def read_metrics(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(rows: list[dict[str, str]], column: str) -> float:
    values = [float(row[column]) for row in rows if row.get(column)]
    values = [value for value in values if not math.isnan(value)]
    if not values:
        raise ValueError(f"No valid values for column {column}")
    return sum(values) / len(values)


def summarize_setting(
    rows: list[dict[str, str]], encoder: str, raw_setting: str
) -> dict[str, float | str]:
    setting_rows = [row for row in rows if row["setting"] == raw_setting]
    if not setting_rows:
        raise ValueError(f"No rows for {encoder} setting={raw_setting}")

    diag_rows = [row for row in setting_rows if row["source"] == row["target"]]
    cross_rows = [row for row in setting_rows if row["source"] != row["target"]]
    if not diag_rows:
        raise ValueError(f"No diagonal rows for {encoder} setting={raw_setting}")
    if not cross_rows:
        raise ValueError(f"No cross-species rows for {encoder} setting={raw_setting}")

    return {
        "encoder": encoder,
        "setting": SETTING_LABELS[raw_setting],
        "aupr_mean": mean(setting_rows, "aupr"),
        "aupr_diag_mean": mean(diag_rows, "aupr"),
        "aupr_cross_species_mean": mean(cross_rows, "aupr"),
        "auc_mean": mean(setting_rows, "auc"),
        "f1_mean": mean(setting_rows, "f1"),
        "mcc_mean": mean(setting_rows, "mcc"),
    }


def format_row(row: dict[str, float | str]) -> dict[str, str]:
    formatted = {"encoder": str(row["encoder"]), "setting": str(row["setting"])}
    for column in SUMMARY_COLUMNS:
        if column not in formatted:
            formatted[column] = f"{float(row[column]):.4f}"
    return formatted


def print_table(rows: list[dict[str, float | str]]) -> None:
    formatted_rows = [format_row(row) for row in rows]
    widths = {
        column: max(len(column), *(len(row[column]) for row in formatted_rows))
        for column in SUMMARY_COLUMNS
    }

    header = "  ".join(column.rjust(widths[column]) for column in SUMMARY_COLUMNS)
    print(header)
    print("-" * len(header))
    for row in formatted_rows:
        print("  ".join(row[column].rjust(widths[column]) for column in SUMMARY_COLUMNS))


def write_csv(rows: list[dict[str, float | str]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(format_row(row) for row in rows)


def main() -> None:
    summary_rows: list[dict[str, float | str]] = []
    for encoder, run_dir in RUNS.items():
        rows = read_metrics(run_dir / "transfer_metrics.csv")
        for raw_setting in SETTING_LABELS:
            summary_rows.append(summarize_setting(rows, encoder, raw_setting))

    print_table(summary_rows)
    write_csv(summary_rows, OUTPUT_CSV)
    print(f"\nSaved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
