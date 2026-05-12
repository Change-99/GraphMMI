import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.data_class import DataOrg, ScoreObj
from src.data.data_handler import *
from src.handler import get_logger, timing
from src.models.models_builder import get_models_dict
from src.models.models_handler import create_dir_with_time
from src.models.param_class import TrainModelParam, EvalModelParam

os.environ['TV_CPP_MIN_LOG_LEVEL'] = '2'


@timing
def run_training(training_dict, model_list, metrics, folder_path):
    t_parm = TrainModelParam(folder_path=folder_path)
    e_parm = EvalModelParam(folder_path=MODELS_OBJECTS_PATH / folder_path)
    scores = ScoreObj(model_list, metrics)
    for src_org_name, datasets in training_dict.items():
        logger.info(f"Start training {src_org_name} models")
        t_parm.data_obj = DataOrg(src_org_name).load(datasets=datasets, is_train="train", val_ratio=0.1)
        e_parm.data_obj = DataOrg(src_org_name).load(datasets=datasets, is_train="test")
        e_parm.src_model_name = src_org_name
        models_dict = get_models_dict(len(t_parm.data_obj))
        for model_name in model_list:
            model = models_dict[model_name]
            logger.info(f"Start training {model_name} {src_org_name} model")
            model.train_model(t_parm=t_parm)
            logger.info(f"Finish training {model_name} {src_org_name} model")
            score_dict = model.evaluate_model(e_parm=e_parm)
            scores.add_score(model_name=model_name, score_dict=score_dict, key=src_org_name)

        logger.info(f"Finish training {src_org_name} models")
    logger.info(f"Finish all training saving results")
    scores.save_results(folder_path)


def training_main():
    global logger
    logger = get_logger("time_logger")
    dir_path = create_dir_with_time(MODELS_OBJECTS_PATH)
    logger.info("---Start learning script---")
    run_training(VS4_REG_DICT, ['base', 'xgb'], ['ACC', 'F1_score'], dir_path)
    logger.info("---End learning script---")


if __name__ == '__main__':
    training_main()
