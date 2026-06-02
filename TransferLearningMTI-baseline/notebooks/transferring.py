import sys
from pathlib import Path
import copy

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data_class import ScoreObj, DataOrg
from src.handler import get_logger, timing
from src.models.models_builder import get_models_dict
from src.models.models_handler import *
from src.models.param_class import TrainModelParam, EvalModelParam


@timing
def run_transfer(training_dict, model_list, _metrics, folder_path):
    t_parm = TrainModelParam(folder_path=folder_path, to_save=False)
    e_parm = EvalModelParam(folder_path=folder_path)
    scores = ScoreObj([f"{x}_baseline" for x in model_list] + model_list, _metrics)
    for src_org_name, src_dataset_list in training_dict.items():
        t_parm.src_model_to_load = src_org_name
        dest_dict = copy.deepcopy(training_dict)
        del dest_dict[src_org_name]
        for dst_org_name, dst_dataset_list in dest_dict.items():
            logger.info(f"Start executing src_org: {src_org_name} and dst_org: {dst_org_name}")
            execute_over_src_dst(dst_dataset_list, dst_org_name, e_parm, model_list, scores, src_org_name, t_parm)
            logger.info(f"Finish executing src_org: {src_org_name} and dst_org: {dst_org_name}")

    res_path = create_dir_with_time(MODELS_PATH / "transfer_tables")
    scores.save_results(folder_name=res_path, header=['src_org', 'dst_org'] + TRANSFER_SIZE_LIST, keep_index=False)


@timing
def execute_over_src_dst(dst_dataset_list, dst_org_name, e_parm, model_list, scores, src_org_name,
                         t_parm):
    e_parm.src_model_name = f"{src_org_name}_{dst_org_name}"
    t_parm.data_obj = DataOrg(dst_org_name).load(datasets=dst_dataset_list, is_train="train")
    e_parm.data_obj = DataOrg(dst_org_name).load(datasets=dst_dataset_list, is_train="test")
    for b_model in model_list:
        t_parm.src_model_to_load = src_org_name
        execute_learning(b_model, scores, t_parm, e_parm, b_model)
        t_parm.src_model_to_load = None
        execute_learning(b_model, scores, t_parm, e_parm, f"{b_model}_baseline")


@timing
def execute_learning(b_model, scores, t_parm: TrainModelParam, e_parm: EvalModelParam, file_name):
    for t_size in TRANSFER_SIZE_LIST:
        logger.info(f"Start running b_model: {file_name} with t_size: {t_size}")
        if t_parm.src_model_to_load is None and t_size == 0:
            scores.add_score(model_name=f"{file_name}", score_dict={}, key=e_parm.src_model_name,
                             fill_empty=True)
            continue
        baseline_models = get_models_dict(len(e_parm.data_obj))
        t_parm.part_train = t_size
        baseline_models[b_model].train_model(t_parm=t_parm)
        score_dict = baseline_models[b_model].evaluate_model(e_parm=e_parm)
        scores.add_score(model_name=f"{file_name}", score_dict=score_dict, key=e_parm.src_model_name)
        logger.info(f"Finish running b_model: {file_name} with t_size: {t_size}")


def transfering_main():
    global logger
    logger = get_logger("time_logger")
    # dir_path = MODELS_OBJECTS_PATH / "good_one"
    dir_path = get_latest_run_dir(MODELS_OBJECTS_PATH)
    logger.info("---Start transfer script---")
    # run_transfer(VS4_REG_DICT, ['xgb'], ['ACC'], dir_path)
    run_transfer(VS4_REG_DICT, ['base', 'xgb'], ['ACC', 'F1_score'], dir_path)
    logger.info("---End transfer script---")


if __name__ == '__main__':
    transfering_main()
