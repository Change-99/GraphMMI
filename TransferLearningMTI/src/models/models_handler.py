import csv
import os
import pickle
import re
from datetime import datetime
from typing import List

import numpy as np
from sklearn import metrics

from constants import *


def save_pkl_model(model, pkl_filename):
    with open(pkl_filename, 'wb') as file:
        pickle.dump(model, file)


def create_evaluation_dict(t_model_name, org_name, pred, y):
    model_name = '{0}_{1}'.format(t_model_name, org_name)
    np.nan_to_num(pred, copy=False)
    eval_dict = {'Model': model_name, 'Date': now(), 'ACC': metrics.accuracy_score(y, np.round(pred))}
    eval_dict['FPR'], eval_dict['TPR'], thresholds = metrics.roc_curve(y, pred)
    eval_dict['AUC'] = metrics.auc(eval_dict['FPR'], eval_dict['TPR'])
    eval_dict['PR'] = metrics.precision_score(y, np.round(pred), average='micro')
    eval_dict['F1_score'] = metrics.f1_score(y, np.round(pred))
    save_metrics(eval_dict)
    print(f"The ACC score of {t_model_name},{org_name} is {eval_dict['ACC']}")
    return eval_dict


def save_metrics(eval_dict):
    f_path = MODELS_PATH / 'models_evaluation.csv'
    with open(f_path, 'a') as file:
        writer = csv.DictWriter(file, eval_dict.keys(), delimiter=',', lineterminator='\n')
        if file.tell() == 0:
            writer.writeheader()  # file doesn't exist yet, write a header
        writer.writerow(eval_dict)


def create_dir_with_time(parent_path: Path):
    dir_path = parent_path / now()
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path


def list_files(path: Path) -> List[Path]:
    dir_entries = sorted(os.scandir(path),
                         key=lambda file_entry: Path(file_entry).stem)
    return [Path(dir_entry.path) for dir_entry in dir_entries]


def get_latest_run_dir(path: Path) -> Path:
    timestamp_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}$")
    run_dirs = [p for p in list_files(path) if p.is_dir() and timestamp_pattern.match(p.name)]
    if not run_dirs:
        raise FileNotFoundError(f"No timestamped run directories found under {path}")
    completed_run_dirs = [
        p for p in run_dirs
        if (p / "base_ACC.csv").exists() and (p / "xgb_ACC.csv").exists()
    ]
    if completed_run_dirs:
        return completed_run_dirs[-1]
    return run_dirs[-1]


def get_latest_csv_run_dir(path: Path) -> Path:
    timestamp_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}--\d{2}-\d{2}-\d{2}$")
    run_dirs = [p for p in list_files(path) if p.is_dir() and timestamp_pattern.match(p.name)]
    populated_dirs = [p for p in run_dirs if any(p.glob("*.csv"))]
    if not populated_dirs:
        raise FileNotFoundError(f"No timestamped result directories with csv files found under {path}")
    return populated_dirs[-1]

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
