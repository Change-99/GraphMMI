from constants import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from src.models.models_handler import list_files, create_dir_with_time, get_latest_run_dir


def create_heatmaps(cross_org_data_path: Path):
    graph_dir_path = create_dir_with_time(MODELS_GRAPHS)
    for file in list_files(cross_org_data_path):
        heatmap_dict = ACC_HEATMAP_DICT if 'ACC' in file.stem else F1_HEATMAP_DICT
        data = pd.read_csv(file, index_col=0)
        draw_heatmap(data=data, img_path=graph_dir_path / f"{file.stem}_heatmap.png", **heatmap_dict)


def draw_heatmap(data, img_path, **kwargs):
    ax = sns.heatmap(data, **kwargs)
    plt.xlabel('Testing Dataset', fontsize=10)
    plt.ylabel('Training Dataset', fontsize=10)
    ax.xaxis.label.set_color('purple')
    ax.yaxis.label.set_color('purple')
    ax.figure.tight_layout()
    plt.savefig(img_path)
    plt.clf()


def get_legend_name(table_key_name):
    res = table_key_name.split("_")
    if 'ACC' in res: res.remove('ACC')
    if 'F1' in res: res.remove('F1')
    if 'score' in res: res.remove('score')
    str_res = "_".join(res)
    str_res = str_res.replace("xgb", 'XGB')
    if "baseline" in str_res:
        str_res = str_res.replace("baseline", "target_org_only").replace("base", 'ANN')
    else:
        str_res = str_res.replace("base", 'ANN') + "_transfer"
    return str_res


def get_upper_limit_dict():
    up_limit_dict = {}
    dir_path = get_latest_run_dir(MODELS_OBJECTS_PATH)
    file_names = [x for x in dir_path.glob('*.csv') if x.is_file()]
    for f in file_names:
        data = pd.read_csv(f, index_col=0)
        # Training summary csv files have one row whose columns are the organism names.
        up_limit_dict[f.stem] = data.iloc[0].to_dict()
    return up_limit_dict


def load_transfer_table(file_path: Path) -> pd.DataFrame:
    data = pd.read_csv(file_path)
    if {'src_org', 'dst_org'}.issubset(data.columns):
        normalized = data.copy()
    else:
        # Older transfer result files were saved transposed:
        # row 0 -> src_org, row 1 -> dst_org, rows 2.. -> transfer sizes.
        normalized = data.T.reset_index(drop=True)
        normalized.columns = ['src_org', 'dst_org'] + [str(x) for x in TRANSFER_SIZE_LIST]

    for col in [str(x) for x in TRANSFER_SIZE_LIST]:
        normalized[col] = pd.to_numeric(normalized[col], errors='coerce')
    return normalized


def create_transfer_graphs(transfer_table_path, metrics):
    graph_dir_path = create_dir_with_time(MODELS_GRAPHS)
    upper_dict = get_upper_limit_dict()
    if not Path(transfer_table_path).is_dir():
        raise FileNotFoundError(
            f"Transfer results directory not found: {transfer_table_path}. "
            f"Run notebooks/transferring.py first."
        )
    for metric in metrics:
        graph_dict = dict()
        for file in list_files(transfer_table_path):
            if metric in file.stem:
                data = load_transfer_table(file)
                data = data.set_index(data['src_org'] + "_" + data['dst_org'])
                data = data.drop(['src_org', 'dst_org'], axis=1)
                graph_dict[file.stem] = data

        if not graph_dict:
            raise FileNotFoundError(
                f"No transfer result csv files for metric {metric} were found under {transfer_table_path}. "
                f"Run notebooks/transferring.py first."
            )

        transfer_model_index = graph_dict[list(graph_dict.keys())[0]].index
        for ind in transfer_model_index:
            rows = []
            for k, trans_df in graph_dict.items():
                rows.append(pd.Series(data=trans_df.loc[ind], name=get_legend_name(k)))
            plot_df = pd.DataFrame(rows)
            draw_transfer_graph(plot_df, ind, graph_dir_path / f"{ind}_{metric}.png", metric, upper_dict)


def draw_transfer_graph(data, org_names, img_path, metric, upper_dict):
    sns.set_theme(style="darkgrid")
    ax = sns.lineplot(data=data.T, dashes=False, linewidth=2.5)
    ax.set(yticks=np.linspace(0, 1, 11))
    ax.set_ylim(0.6, 0.95)
    ax.legend(data.index)
    xgb_upper = upper_dict[f"xgb_{metric}"][org_names.split("_")[1]]
    base_upper = upper_dict[f"base_{metric}"][org_names.split("_")[1]]
    ax.axhline(xgb_upper, ls='--', c='green', xmin=0)
    ax.axhline(base_upper, ls='--', c='blue', xmin=0)
    plt.title(f"{org_names}", fontsize=15)
    plt.xlabel('#Target observations', fontsize=10)
    plt.ylabel(metric, fontsize=10)
    plt.legend(loc='lower right')
    ax.title.set_color('purple')
    ax.figure.tight_layout()
    plt.savefig(img_path)
    plt.clf()
