import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data_handler import *


def run_preprocessing():
    dataset_list = DATASETS
    for data in dataset_list:
        create_train_dataset(data, remove_hot_paring=True, only_most_important=False, dist_split=False)


if __name__ == '__main__':
    run_preprocessing()
