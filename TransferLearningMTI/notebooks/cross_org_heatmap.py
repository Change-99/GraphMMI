import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from constants import *
from src.data.data_class import ScoreObj, DataOrg
from src.handler import get_logger, timing
from src.models.models_builder import get_models_dict
from src.models.models_handler import create_dir_with_time, get_latest_run_dir
from src.models.param_class import EvalModelParam


@timing
def create_cross_org_tables(training_dict, model_list, metrics, folder_path):
    e_parm = EvalModelParam(folder_path=folder_path, to_load=True)
    scores = ScoreObj(model_list, metrics)
    for src_org_name, src_dataset_list in training_dict.items():
        e_parm.src_model_name = src_org_name
        for dst_org_name, dst_dataset_list in training_dict.items():
            logger.info(f"Start cross orgs for src_org: {src_org_name} and dst_org: {dst_org_name}")
            e_parm.data_obj = DataOrg(dst_org_name).load(datasets=dst_dataset_list, is_train="test")
            models_dict = get_models_dict(len(e_parm.data_obj))
            for model_name in model_list:
                model = models_dict[model_name]
                score_dict = model.evaluate_model(e_parm=e_parm)
                scores.add_score(model_name=model_name, score_dict=score_dict, key=dst_org_name)
            logger.info(f"Finish cross orgs for src_org: {src_org_name} and dst_org: {dst_org_name}")
    res_path = create_dir_with_time(MODELS_PATH / "cross_org_tabels")
    scores.save_results(folder_name=res_path, header=list(training_dict.keys()))


def cross_org_heatmap_main():
    global logger
    logger = get_logger("time_logger")
    # dir_path = MODELS_OBJECTS_PATH / "good_one"
    dir_path = get_latest_run_dir(MODELS_OBJECTS_PATH)
    logger.info("---Start cross_org_heatmap script---")
    create_cross_org_tables(VS4_REG_DICT, ['base', 'xgb'], ['ACC', 'F1_score'], dir_path)
    logger.info("---Finish cross_org_heatmap script---")


if __name__ == '__main__':
    cross_org_heatmap_main()
