#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MPLCONFIG_PATH = ROOT / ".mplconfig"
MPLCONFIG_PATH.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_PATH))

DATASETS = ["cow1", "worm1", "worm2", "human1", "human2", "human3", "mouse1", "mouse2"]
SPECIES_DATASETS = {
    "human": ["human1", "human2", "human3"],
    "cow": ["cow1"],
    "mouse": ["mouse1", "mouse2"],
    "worm": ["worm1", "worm2"],
}
SPECIES_ORDER = ["human", "cow", "mouse", "worm"]

IMPORTANT_FEATURES = [
    "miRNAPairingCount_Seed_GU",
    "miRNAMatchPosition_1",
    "miRNAPairingCount_Total_GU",
    "Energy_MEF_local_target",
    "MRNA_Target_G_comp",
    "MRNA_Target_GG_comp",
    "miRNAMatchPosition_4",
    "miRNAMatchPosition_5",
    "miRNAPairingCount_Seed_bulge_nt",
    "miRNAPairingCount_Seed_GC",
    "miRNAMatchPosition_2",
    "miRNAPairingCount_Seed_mismatch",
    "miRNAPairingCount_X3p_GC",
    "Seed_match_compact_interactions_all",
]
SEQUENCE_FEATURES = [
    "mRNA_start",
    "label",
    "mRNA_name",
    "target sequence",
    "microRNA_name",
    "miRNA sequence",
    "full_mrna",
]

FEATURES_TO_DROP = [
    "mRNA_start",
    "label",
    "mRNA_name",
    "target sequence",
    "microRNA_name",
    "miRNA sequence",
    "full_mrna",
    "canonic_seed",
    "duplex_RNAplex_equals",
    "non_canonic_seed",
    "site_start",
    "num_of_pairs",
    "mRNA_end",
    "constraint",
]

XGB_PARAMS = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "learning_rate": 0.1,
    "gamma": 0.5,
    "max_depth": 2,
    "min_child_weight": 1,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "reg_lambda": 1,
    "verbosity": 0,
    "eval_metric": ["error", "logloss"],
    "random_state": 42,
}


@dataclass
class TabularDataset:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_frame: pd.DataFrame
    test_frame: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline reproduction for miRNA-mRNA transfer learning. "
            "It keeps ANN/XGBoost tabular baselines isolated from the future GNN pipeline."
        )
    )
    parser.add_argument("--external-dir", type=Path, default=ROOT / "data/external")
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/baseline_tabular")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/baseline_transfer")
    parser.add_argument("--models", nargs="+", default=["ann", "xgb"], choices=["ann", "xgb"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--transfer-size", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--xgb-n-estimators", type=int, default=100)
    parser.add_argument("--xgb-n-jobs", type=int, default=-1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--max-transfer-val-rows",
        type=int,
        default=0,
        help=(
            "Optional speed knob. 0 keeps the baseline behavior and validates on all remaining "
            "target training rows after the transfer subset."
        ),
    )
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--only-important", action="store_true")
    parser.add_argument(
        "--keep-hot-pairing",
        action="store_true",
        help="By default this reproduces baseline preprocessing and drops HotPairing* columns.",
    )
    parser.add_argument(
        "--check-data",
        action="store_true",
        help="Run preprocessing and feature alignment only; do not import ANN/XGBoost dependencies.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def dataset_seed(seed: int, name: str) -> int:
    return seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(name))


def json_safe(value: Any) -> Any:
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


def stratify_train_test_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "microRNA_name" not in df.columns:
        raise ValueError("Expected column microRNA_name for baseline stratified split.")

    rng = np.random.default_rng(seed)
    counts = df["microRNA_name"].value_counts(dropna=False)
    unique_mask = df["microRNA_name"].map(counts).eq(1)
    unique_mirna = df[unique_mask]
    non_unique = df[~unique_mask]

    train_indices: list[int] = []
    test_indices: list[int] = []
    for _, group in non_unique.groupby("microRNA_name", dropna=False):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        if len(indices) <= 1:
            test_indices.extend(indices.tolist())
            continue
        n_test = int(round(len(indices) * test_size))
        n_test = min(max(n_test, 1), len(indices) - 1)
        test_indices.extend(indices[:n_test].tolist())
        train_indices.extend(indices[n_test:].tolist())

    train = df.loc[train_indices].copy()
    test = pd.concat([df.loc[test_indices], unique_mirna], ignore_index=False).copy()
    return train, test


def read_and_split_dataset(
    dataset_id: str,
    external_dir: Path,
    test_size: float,
    seed: int,
    remove_hot_pairing: bool,
    only_important: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pos_path = external_dir / f"{dataset_id}_pos.csv"
    neg_path = external_dir / f"{dataset_id}_neg.csv"
    if not pos_path.exists() or not neg_path.exists():
        raise FileNotFoundError(f"Missing pos/neg files for dataset {dataset_id} under {external_dir}")

    pos = pd.read_csv(pos_path, index_col=0, low_memory=False)
    neg = pd.read_csv(neg_path, index_col=0, low_memory=False)
    pos.insert(0, "label", 1)
    neg.insert(0, "label", 0)

    pos.drop(["Source", "Organism", "number of reads"], axis=1, inplace=True, errors="ignore")
    if remove_hot_pairing:
        pos = pos[[col for col in pos.columns if not str(col).startswith("HotPairing")]]

    common_columns = [col for col in pos.columns if col in neg.columns]
    pos = pos[common_columns]
    neg = neg[common_columns]

    if only_important:
        selected = [col for col in IMPORTANT_FEATURES + SEQUENCE_FEATURES if col in common_columns]
        pos = pos[selected]
        neg = neg[selected]

    split_seed = dataset_seed(seed, dataset_id)
    pos_train, pos_test = stratify_train_test_split(pos, test_size, split_seed)
    neg_train, neg_test = stratify_train_test_split(neg, test_size, split_seed)

    train = pd.concat([pos_train, neg_train], ignore_index=True)
    test = pd.concat([pos_test, neg_test], ignore_index=True)

    rng = np.random.default_rng(split_seed)
    train = train.iloc[rng.permutation(len(train))].reset_index(drop=True)
    test = test.iloc[rng.permutation(len(test))].reset_index(drop=True)

    summary = {
        "dataset": dataset_id,
        "raw_positive_rows": int(len(pos)),
        "raw_negative_rows": int(len(neg)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positive": int(train["label"].sum()),
        "test_positive": int(test["label"].sum()),
        "num_columns": int(len(common_columns)),
        "remove_hot_pairing": remove_hot_pairing,
        "only_important": only_important,
    }
    return train, test, summary


def preprocess_all(args: argparse.Namespace) -> dict[str, Any]:
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    for dataset_id in DATASETS:
        train, test, summary = read_and_split_dataset(
            dataset_id=dataset_id,
            external_dir=args.external_dir,
            test_size=args.test_size,
            seed=args.seed,
            remove_hot_pairing=not args.keep_hot_pairing,
            only_important=args.only_important,
        )
        train.to_csv(args.processed_dir / f"{dataset_id}_train.csv", index=False)
        test.to_csv(args.processed_dir / f"{dataset_id}_test.csv", index=False)
        summaries[dataset_id] = summary
    manifest = {
        "external_dir": args.external_dir,
        "processed_dir": args.processed_dir,
        "datasets": summaries,
        "note": "Baseline-style tabular preprocessing copied into GraphMMI without importing TransferLearningMTI.",
    }
    (args.processed_dir / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2), encoding="utf-8")
    return manifest


def read_species_frames(processed_dir: Path) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    species_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for species, datasets in SPECIES_DATASETS.items():
        train_parts = []
        test_parts = []
        for dataset_id in datasets:
            train_path = processed_dir / f"{dataset_id}_train.csv"
            test_path = processed_dir / f"{dataset_id}_test.csv"
            if not train_path.exists() or not test_path.exists():
                raise FileNotFoundError(
                    f"Missing processed train/test for {dataset_id}. Run without --skip-preprocess first."
                )
            train_parts.append(pd.read_csv(train_path, low_memory=False))
            test_parts.append(pd.read_csv(test_path, low_memory=False))
        species_frames[species] = (
            pd.concat(train_parts, ignore_index=True, sort=False),
            pd.concat(test_parts, ignore_index=True, sort=False),
        )
    return species_frames


def infer_common_feature_columns(species_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> list[str]:
    drop = set(FEATURES_TO_DROP)
    all_frames = [frame for pair in species_frames.values() for frame in pair]
    common = set(all_frames[0].columns)
    for frame in all_frames[1:]:
        common &= set(frame.columns)
    reference_columns = list(all_frames[0].columns)
    return [col for col in reference_columns if col in common and col not in drop and col != "label"]


def frame_to_arrays(frame: pd.DataFrame, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if "label" not in frame.columns:
        raise ValueError("Expected label column in processed frame.")
    features = frame.reindex(columns=feature_columns)
    features = features.replace({True: 1, False: 0, "True": 1, "False": 0})
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x = features.to_numpy(dtype=np.float32)
    y = pd.to_numeric(frame["label"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
    return x, y


def build_tabular_datasets(
    species_frames: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    feature_columns: list[str],
) -> dict[str, TabularDataset]:
    datasets: dict[str, TabularDataset] = {}
    for species, (train_frame, test_frame) in species_frames.items():
        x_train, y_train = frame_to_arrays(train_frame, feature_columns)
        x_test, y_test = frame_to_arrays(test_frame, feature_columns)
        datasets[species] = TabularDataset(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            train_frame=train_frame,
            test_frame=test_frame,
        )
    return datasets


def split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(y) == 0:
        raise ValueError("Cannot split an empty dataset.")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_val = int(round(len(y) * val_ratio))
    n_val = min(max(n_val, 1), len(y) - 1) if len(y) > 1 else 0
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    if len(val_idx) == 0:
        val_idx = train_idx
    return x[train_idx], x[val_idx], y[train_idx], y[val_idx]


def transfer_subset(
    dataset: TabularDataset,
    transfer_size: int,
    seed: int,
    max_val_rows: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(dataset.y_train))
    n_train = min(max(transfer_size, 1), len(order))
    train_idx = order[:n_train]
    val_idx = order[n_train:]
    if len(val_idx) == 0:
        val_idx = train_idx
    elif max_val_rows > 0 and len(val_idx) > max_val_rows:
        val_idx = val_idx[:max_val_rows]
    return dataset.x_train[train_idx], dataset.x_train[val_idx], dataset.y_train[train_idx], dataset.y_train[val_idx]


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int64)
    pos = labels == 1
    neg = labels == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty_like(scores, dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    sum_pos_ranks = ranks[pos].sum()
    return float((sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def classification_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    labels_bool = labels.astype(bool)
    pred = probs >= threshold
    tp = int(np.logical_and(pred, labels_bool).sum())
    tn = int(np.logical_and(~pred, ~labels_bool).sum())
    fp = int(np.logical_and(pred, ~labels_bool).sum())
    fn = int(np.logical_and(~pred, labels_bool).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": binary_auc(labels, probs),
    }


def evaluate_predictions(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    probs = np.nan_to_num(np.ravel(probs).astype(np.float64), nan=0.0, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, 0.0, 1.0)
    return classification_metrics(y_true, probs, threshold)


def require_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "TensorFlow/Keras is required for --models ann. "
            "Use the baseline conda environment from TransferLearningMTI/environment.yml."
        ) from exc
    return tf


def require_xgboost():
    try:
        import xgboost as xgb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "XGBoost is required for --models xgb. "
            "Use the baseline conda environment from TransferLearningMTI/environment.yml."
        ) from exc
    return xgb


def build_ann_model(tf: Any, input_dim: int):
    regularizers = tf.keras.regularizers
    layers = tf.keras.layers
    inputs = layers.Input(shape=(input_dim,), name="input")
    h = layers.Dense(
        300,
        activation="relu",
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.l2(1e-4),
        activity_regularizer=regularizers.l2(1e-5),
        name="dense_300",
    )(inputs)
    h = layers.Dropout(rate=0.6, name="dropout_1")(h)
    h = layers.Dense(
        200,
        activation="relu",
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
        bias_regularizer=regularizers.l2(1e-4),
        activity_regularizer=regularizers.l2(1e-5),
        name="dense_200",
    )(h)
    h = layers.Dropout(rate=0.6, name="dropout_2")(h)
    h = layers.Dense(100, activation="relu", name="dense_100")(h)
    h = layers.Dropout(rate=0.6, name="dropout_3")(h)
    h = layers.Dense(20, activation="relu", name="dense_20")(h)
    outputs = layers.Dense(1, activation="sigmoid", name="output")(h)
    model = tf.keras.Model(inputs, outputs, name="baseline_ann")
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    return model


def freeze_ann_for_transfer(model: Any) -> None:
    for layer in model.layers:
        layer.trainable = False
    model.get_layer("dense_20").trainable = True
    model.get_layer("output").trainable = True
    model.compile(optimizer=model.optimizer.__class__.from_config(model.optimizer.get_config()), loss="binary_crossentropy", metrics=["accuracy"])


def build_xgb_model(xgb: Any, n_estimators: int, n_jobs: int, seed: int):
    params = dict(XGB_PARAMS)
    params["n_estimators"] = n_estimators
    params["n_jobs"] = n_jobs
    params["random_state"] = seed
    return xgb.XGBClassifier(**params)


def train_ann_transfer(
    datasets: dict[str, TabularDataset],
    args: argparse.Namespace,
    run_dir: Path,
) -> list[dict[str, Any]]:
    tf = require_tensorflow()
    tf.random.set_seed(args.seed)
    input_dim = next(iter(datasets.values())).x_train.shape[1]
    source_models: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    model_dir = run_dir / "models/ann"
    model_dir.mkdir(parents=True, exist_ok=True)

    for source in SPECIES_ORDER:
        dataset = datasets[source]
        x_train, x_val, y_train, y_val = split_train_val(
            dataset.x_train, dataset.y_train, args.val_ratio, dataset_seed(args.seed, f"ann-{source}")
        )
        model = build_ann_model(tf, input_dim)
        model.fit(
            x_train,
            y_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_val, y_val),
            verbose=0,
        )
        model.save_weights(str(model_dir / f"{source}.weights.h5"))
        source_models[source] = model
        print(f"[ANN] trained source model: {source}", flush=True)

    for source in SPECIES_ORDER:
        source_model = source_models[source]
        for target in SPECIES_ORDER:
            target_dataset = datasets[target]
            source_probs = np.ravel(source_model.predict(target_dataset.x_test, verbose=0))
            source_metrics = evaluate_predictions(target_dataset.y_test, source_probs, args.threshold)
            rows.append(
                {
                    "model": "ann",
                    "protocol": "source_only",
                    "source": source,
                    "target": target,
                    "transfer_size": 0,
                    **source_metrics,
                }
            )

            if source == target:
                transfer_metrics = source_metrics
            else:
                transfer_model = build_ann_model(tf, input_dim)
                transfer_model.set_weights(source_model.get_weights())
                freeze_ann_for_transfer(transfer_model)
                x_ft, x_val, y_ft, y_val = transfer_subset(
                    target_dataset,
                    args.transfer_size,
                    dataset_seed(args.seed, f"ann-{source}-{target}"),
                    args.max_transfer_val_rows,
                )
                transfer_model.fit(
                    x_ft,
                    y_ft,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    validation_data=(x_val, y_val),
                    verbose=0,
                )
                probs = np.ravel(transfer_model.predict(target_dataset.x_test, verbose=0))
                transfer_metrics = evaluate_predictions(target_dataset.y_test, probs, args.threshold)
            print(f"[ANN] evaluated {source}->{target}", flush=True)
            rows.append(
                {
                    "model": "ann",
                    "protocol": "transfer",
                    "source": source,
                    "target": target,
                    "transfer_size": 0 if source == target else args.transfer_size,
                    **transfer_metrics,
                }
            )
    return rows


def train_xgb_transfer(
    datasets: dict[str, TabularDataset],
    args: argparse.Namespace,
    run_dir: Path,
) -> list[dict[str, Any]]:
    xgb = require_xgboost()
    source_models: dict[str, Any] = {}
    source_paths: dict[str, Path] = {}
    rows: list[dict[str, Any]] = []
    model_dir = run_dir / "models/xgb"
    model_dir.mkdir(parents=True, exist_ok=True)

    for source in SPECIES_ORDER:
        dataset = datasets[source]
        x_train, x_val, y_train, y_val = split_train_val(
            dataset.x_train, dataset.y_train, args.val_ratio, dataset_seed(args.seed, f"xgb-{source}")
        )
        model = build_xgb_model(xgb, args.xgb_n_estimators, args.xgb_n_jobs, args.seed)
        model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
        source_path = model_dir / f"{source}.json"
        model.save_model(str(source_path))
        source_models[source] = model
        source_paths[source] = source_path
        print(f"[XGB] trained source model: {source}", flush=True)

    for source in SPECIES_ORDER:
        source_model = source_models[source]
        for target in SPECIES_ORDER:
            target_dataset = datasets[target]
            source_probs = source_model.predict_proba(target_dataset.x_test)[:, 1]
            source_metrics = evaluate_predictions(target_dataset.y_test, source_probs, args.threshold)
            rows.append(
                {
                    "model": "xgb",
                    "protocol": "source_only",
                    "source": source,
                    "target": target,
                    "transfer_size": 0,
                    **source_metrics,
                }
            )

            if source == target:
                transfer_metrics = source_metrics
            else:
                transfer_model = build_xgb_model(xgb, args.xgb_n_estimators, args.xgb_n_jobs, args.seed)
                x_ft, x_val, y_ft, y_val = transfer_subset(
                    target_dataset,
                    args.transfer_size,
                    dataset_seed(args.seed, f"xgb-{source}-{target}"),
                    args.max_transfer_val_rows,
                )
                transfer_model.fit(
                    x_ft,
                    y_ft,
                    eval_set=[(x_val, y_val)],
                    verbose=False,
                    xgb_model=str(source_paths[source]),
                )
                probs = transfer_model.predict_proba(target_dataset.x_test)[:, 1]
                transfer_metrics = evaluate_predictions(target_dataset.y_test, probs, args.threshold)
            print(f"[XGB] evaluated {source}->{target}", flush=True)
            rows.append(
                {
                    "model": "xgb",
                    "protocol": "transfer",
                    "source": source,
                    "target": target,
                    "transfer_size": 0 if source == target else args.transfer_size,
                    **transfer_metrics,
                }
            )
    return rows


def write_long_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_matrix(rows: list[dict[str, Any]], model: str, protocol: str, metric: str) -> np.ndarray:
    values = {
        (row["source"], row["target"]): row[metric]
        for row in rows
        if row["model"] == model and row["protocol"] == protocol
    }
    matrix = np.zeros((len(SPECIES_ORDER), len(SPECIES_ORDER)), dtype=np.float64)
    for i, source in enumerate(SPECIES_ORDER):
        for j, target in enumerate(SPECIES_ORDER):
            matrix[i, j] = float(values.get((source, target), np.nan))
    return matrix


def write_matrix_csv(matrix: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source\\target", *SPECIES_ORDER])
        for source, values in zip(SPECIES_ORDER, matrix):
            writer.writerow([source, *[float(value) for value in values]])


def plot_heatmap(matrix: np.ndarray, title: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
    except ModuleNotFoundError:
        sns = None

    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=180)
    if sns is not None:
        sns.heatmap(
            matrix,
            vmin=0.0,
            vmax=1.0,
            cmap="RdBu_r",
            annot=True,
            fmt=".3f",
            xticklabels=SPECIES_ORDER,
            yticklabels=SPECIES_ORDER,
            square=True,
            cbar_kws={"label": "score"},
            ax=ax,
        )
    else:
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=0.0, vmax=1.0)
        ax.set_xticks(np.arange(len(SPECIES_ORDER)), labels=SPECIES_ORDER)
        ax.set_yticks(np.arange(len(SPECIES_ORDER)), labels=SPECIES_ORDER)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax)
    ax.set_xlabel("Target species")
    ax.set_ylabel("Source species")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_matrices_and_heatmaps(rows: list[dict[str, Any]], run_dir: Path) -> None:
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    models = sorted({row["model"] for row in rows})
    protocols = sorted({row["protocol"] for row in rows})
    for model in models:
        for protocol in protocols:
            for metric in metrics:
                matrix = metric_matrix(rows, model, protocol, metric)
                base_name = f"{model}_{protocol}_{metric}"
                write_matrix_csv(matrix, run_dir / "matrices" / f"{base_name}_matrix.csv")
                plot_heatmap(
                    matrix,
                    title=f"{model.upper()} {protocol.replace('_', ' ')} {metric.upper()}",
                    path=run_dir / "heatmaps" / f"{base_name}_heatmap.png",
                )


def write_run_summary(
    args: argparse.Namespace,
    run_dir: Path,
    manifest: dict[str, Any] | None,
    datasets: dict[str, TabularDataset],
    feature_columns: list[str],
) -> None:
    summary = {
        "config": {key: json_safe(value) for key, value in vars(args).items()},
        "manifest": manifest,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "species": {
            species: {
                "train_rows": int(len(dataset.y_train)),
                "test_rows": int(len(dataset.y_test)),
                "train_positive": int(dataset.y_train.sum()),
                "test_positive": int(dataset.y_test.sum()),
            }
            for species, dataset in datasets.items()
        },
    }
    (run_dir / "run_summary.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")


def print_data_summary(datasets: dict[str, TabularDataset], feature_columns: list[str]) -> None:
    print(f"Aligned feature count: {len(feature_columns)}")
    for species in SPECIES_ORDER:
        dataset = datasets[species]
        print(
            f"{species:>5} train={len(dataset.y_train):6d} "
            f"test={len(dataset.y_test):6d} features={dataset.x_train.shape[1]:4d}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = args.run_root / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = None
    if not args.skip_preprocess:
        manifest = preprocess_all(args)

    species_frames = read_species_frames(args.processed_dir)
    feature_columns = infer_common_feature_columns(species_frames)
    datasets = build_tabular_datasets(species_frames, feature_columns)
    write_run_summary(args, run_dir, manifest, datasets, feature_columns)
    print_data_summary(datasets, feature_columns)

    if args.check_data:
        print(f"Data check complete. Summary saved to: {run_dir / 'run_summary.json'}", flush=True)
        return

    all_rows: list[dict[str, Any]] = []
    if "ann" in args.models:
        print("Training ANN baseline and transfer matrices...", flush=True)
        all_rows.extend(train_ann_transfer(datasets, args, run_dir))
    if "xgb" in args.models:
        print("Training XGBoost baseline and transfer matrices...", flush=True)
        all_rows.extend(train_xgb_transfer(datasets, args, run_dir))

    write_long_metrics(all_rows, run_dir / "metrics_long.csv")
    write_matrices_and_heatmaps(all_rows, run_dir)
    print(f"Saved baseline reproduction run: {run_dir}", flush=True)
    print(f"Transfer heatmaps: {run_dir / 'heatmaps'}", flush=True)


if __name__ == "__main__":
    main()
