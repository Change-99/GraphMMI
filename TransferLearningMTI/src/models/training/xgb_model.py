from sklearn.model_selection import train_test_split

from src.models.model_learner import ModelLearner
from src.models.models_handler import create_evaluation_dict
from src.models.param_class import TrainModelParam, EvalModelParam


class XgboostTrainObj(ModelLearner):

    def __init__(self):
        self.model_name = 'xgb'
        ModelLearner.__init__(self)

    def set_model(self, model):
        self.model = model

    def train_model(self, t_parm: TrainModelParam):
        if t_parm.src_model_to_load is not None:
            l_model_path = t_parm.folder_path / f"{t_parm.src_model_to_load}.dat"
            self.model.load_model(t_parm.folder_path / f"{t_parm.src_model_to_load}.dat")
        else:
            l_model_path = None
        # noinspection DuplicatedCode
        if t_parm.part_train is None:
            x, x_v, y, y_v = t_parm.data_obj.data, t_parm.data_obj.v_data, t_parm.data_obj.label, t_parm.data_obj.v_label
        elif t_parm.part_train == 0:
            return self.model
        else:
            x, x_v, y, y_v = train_test_split(t_parm.data_obj.data, t_parm.data_obj.label, train_size=t_parm.part_train,
                                              random_state=42)
        self.model = self.model.fit(x, y, eval_set=[(x_v, y_v)], verbose=False, xgb_model=l_model_path)

        if t_parm.to_save:
            self.model.save_model(t_parm.folder_path / f"{t_parm.data_obj.dataset_name}.dat")

    def evaluate_model(self, e_parm: EvalModelParam):
        if e_parm.to_load:
            self.model.load_model(e_parm.folder_path / f"{e_parm.src_model_name}.dat")
        pred = self.model.predict_proba(e_parm.data_obj.data)[:, 1]
        eval_dict = create_evaluation_dict(self.model_name, e_parm.src_model_name, pred, e_parm.data_obj.label)
        return eval_dict
