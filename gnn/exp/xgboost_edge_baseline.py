#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XGBoost baseline over precomputed edge_attr features."
    )
    parser.add_argument("--species", required=True, choices=["human", "mouse", "worm", "cow"])
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed/graphsage")
    parser.add_argument("--run-root", type=Path, default=ROOT / "exp/runs/xgboost_edge")
    parser.add_argument("--drop-missing-leak-cols", action="store_true")
    parser.add_argument("--missing-gap-threshold", type=float, default=0.01)
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.9)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def load_npz_features(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    data = np.load(data_path, allow_pickle=True)
    feature_names = data["edge_feature_names"].astype(str).tolist()
    return (
        data["train_edge_attr"].astype(np.float32),
        data["train_edge_label"].astype(np.int64),
        data["val_edge_attr"].astype(np.float32),
        data["val_edge_label"].astype(np.int64),
        data["test_edge_attr"].astype(np.float32),
        data["test_edge_label"].astype(np.int64),
        feature_names,
    )


def detect_missing_leak_columns(
    raw_path: Path,
    feature_names: list[str],
    missing_gap_threshold: float,
) -> tuple[list[str], pd.DataFrame]:
    raw = pd.read_csv(raw_path)
    labels = raw["label"].to_numpy(dtype=np.int64)
    rows: list[dict[str, Any]] = []
    dropped: list[str] = []

    for feature in feature_names:
        if feature not in raw.columns:
            continue
        values = raw[feature]
        pos = values[labels == 1]
        neg = values[labels == 0]
        pos_missing = float(pos.isna().mean())
        neg_missing = float(neg.isna().mean())
        gap = abs(pos_missing - neg_missing)
        drop = gap > missing_gap_threshold or pos.notna().sum() == 0 or neg.notna().sum() == 0
        if drop:
            dropped.append(feature)
        rows.append(
            {
                "feature": feature,
                "pos_missing_rate": pos_missing,
                "neg_missing_rate": neg_missing,
                "missing_gap": gap,
                "drop": drop,
            }
        )

    report = pd.DataFrame(rows).sort_values("missing_gap", ascending=False)
    return dropped, report


def metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = prob >= threshold
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "ap": float(average_precision_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


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


def main() -> None:
    args = parse_args()
    data_path = args.data_root / args.species / "graphsage_inputs.npz"
    raw_path = args.data_root / args.species / "edge_features_raw.csv"
    if not data_path.exists():
        raise FileNotFoundError(data_path)

    x_train, y_train, x_val, y_val, x_test, y_test, feature_names = load_npz_features(data_path)
    if x_train.shape[1] <= 0:
        raise ValueError(f"{data_path} has edge_attr_dim={x_train.shape[1]}; XGBoost edge baseline needs features.")

    dropped_features: list[str] = []
    missing_report = pd.DataFrame()
    if args.drop_missing_leak_cols:
        if not raw_path.exists():
            raise FileNotFoundError(f"Need raw feature file for missingness leak detection: {raw_path}")
        dropped_features, missing_report = detect_missing_leak_columns(
            raw_path, feature_names, args.missing_gap_threshold
        )
        keep_indices = [i for i, name in enumerate(feature_names) if name not in set(dropped_features)]
        x_train = x_train[:, keep_indices]
        x_val = x_val[:, keep_indices]
        x_test = x_test[:, keep_indices]
        feature_names = [feature_names[i] for i in keep_indices]

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "_drop_missing_leak" if args.drop_missing_leak_cols else ""
    run_dir = args.run_root / f"{args.species}{suffix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric=["logloss", "auc", "aucpr"],
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=args.reg_lambda,
        reg_alpha=args.reg_alpha,
        random_state=args.seed,
        n_jobs=args.n_jobs,
        tree_method="hist",
    )

    print(
        f"XGBoost edge baseline {args.species}: "
        f"train={x_train.shape} val={x_val.shape} test={x_test.shape} "
        f"drop_missing_leak_cols={args.drop_missing_leak_cols}"
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    train_prob = model.predict_proba(x_train)[:, 1]
    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]

    result = {
        "config": vars(args),
        "data_path": data_path,
        "raw_path": raw_path if raw_path.exists() else None,
        "num_features_used": len(feature_names),
        "num_features_dropped": len(dropped_features),
        "dropped_features": dropped_features,
        "train": metrics(y_train, train_prob, args.threshold),
        "val": metrics(y_val, val_prob, args.threshold),
        "test": metrics(y_test, test_prob, args.threshold),
    }

    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_gain": model.feature_importances_,
        }
    ).sort_values("importance_gain", ascending=False)
    importances.to_csv(run_dir / "feature_importance.csv", index=False)
    if not missing_report.empty:
        missing_report.to_csv(run_dir / "missingness_report.csv", index=False)

    pd.DataFrame(
        {
            "label": y_test,
            "probability": test_prob,
            "prediction": (test_prob >= args.threshold).astype(int),
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)

    (run_dir / "metrics.json").write_text(json.dumps(json_safe(result), indent=2), encoding="utf-8")
    print(json.dumps(json_safe(result), indent=2))
    print(f"Saved run: {run_dir}")


if __name__ == "__main__":
    main()
