from abc import abstractmethod
from timeit import timeit

import matplotlib.pyplot as plt
import shap
from sklearn.inspection import permutation_importance

from src.models.models_handler import *
from src.models.param_class import TrainModelParam, EvalModelParam


class ModelLearner:
    model_name = None
    model = None

    def __init__(self):
        pass

    @abstractmethod
    def set_model(self, input_shape):
        raise NotImplementedError("Must override get_shap_values method")

    @abstractmethod
    # @timeit
    def train_model(self, t_parm: TrainModelParam):
        raise NotImplementedError("Must override train_model method")

    def plot_learning_curves(self, folder_name, org_name):
        pass

    @abstractmethod
    def evaluate_model(self, e_param: EvalModelParam):
        raise NotImplementedError("Must override evaluate_model method")


@abstractmethod
def get_shap_values():
    raise NotImplementedError("Must override get_shap_values method")
