import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.visualization_handler import *
from src.visualization.visualization_handler import create_transfer_graphs
from src.models.models_handler import get_latest_csv_run_dir


def visualization_main(run_heatmap=False, run_transfer_graphs=False, cross_org_dir_path=None, transfer_table_path=None):
    if run_heatmap:
        if cross_org_dir_path is None:
            cross_org_dir_path = get_latest_csv_run_dir(MODELS_PATH / "cross_org_tabels")
        create_heatmaps(cross_org_dir_path)

    if run_transfer_graphs:
        if transfer_table_path is None:
            transfer_table_path = get_latest_csv_run_dir(MODELS_PATH / "transfer_tables")
        create_transfer_graphs(transfer_table_path, ['ACC', 'F1_score'])


if __name__ == '__main__':
    visualization_main(run_heatmap=False, run_transfer_graphs=True)
