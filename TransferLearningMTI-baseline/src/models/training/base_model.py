import logging
from pathlib import Path

import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from src.models.model_learner import ModelLearner, create_evaluation_dict
from src.models.param_class import TrainModelParam, EvalModelParam

logging.getLogger("tensorflow").setLevel(logging.CRITICAL)


class BaseTrainObj(ModelLearner):
    history = None

    def __init__(self):
        self.model_name = 'base'
        ModelLearner.__init__(self)

    def set_model(self, model):
        self.model = model
        self.model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['acc'])

    def train_model(self, t_parm: TrainModelParam):
        if t_parm.src_model_to_load is not None:
            self.model.load_weights(t_parm.folder_path / f"{t_parm.src_model_to_load}.weights.h5")
            for l in self.model.layers:
                l.trainable = False
            self.model.get_layer('dense_20').trainable = True
            self.model.get_layer('output').trainable = True
        # noinspection DuplicatedCode
        if t_parm.part_train is None:
            x, x_v, y, y_v = t_parm.data_obj.data, t_parm.data_obj.v_data, t_parm.data_obj.label, t_parm.data_obj.v_label
        elif t_parm.part_train == 0:
            return
        else:
            x, x_v, y, y_v = train_test_split(t_parm.data_obj.data, t_parm.data_obj.label, train_size=t_parm.part_train,
                                              random_state=42)
        self.history = self.model.fit(x, y, epochs=t_parm.epochs, validation_data=(x_v, y_v), verbose=0)
        self.plot_learning_curves(t_parm.folder_path, t_parm.data_obj.dataset_name)
        if t_parm.to_save:
            model_folder_name = t_parm.folder_path / f"{t_parm.data_obj.dataset_name}/"
            Path(model_folder_name).mkdir(parents=True, exist_ok=True)
            self.model.save_weights(t_parm.folder_path / f"{t_parm.data_obj.dataset_name}.weights.h5")

    def plot_learning_curves(self, folder_name, org_name):
        f_name = folder_name / "learning_results"
        Path(f_name).mkdir(parents=True, exist_ok=True)
        for metric in ['acc', 'loss']:
            self.draw_curve(metric, f_name, org_name)

    def draw_curve(self, metric, f_name, org_name):
        reg_metric = self.history.history[metric]
        val_metric = self.history.history[f"val_{metric}"]
        epochs = range(1, len(reg_metric) + 1)
        plt.figure()
        plt.plot(epochs, reg_metric, 'bo', label=f"Training {metric}_metric")
        plt.plot(epochs, val_metric, 'b', label=f"Validation {metric}_metric")
        plt.title(f"{metric} {self.model_name}")
        plt.legend()
        plt.savefig(f_name / f"{self.model_name}_{metric}_{org_name}.png")
        plt.clf()

    def evaluate_model(self, e_parm: EvalModelParam):
        if e_parm.to_load:
            self.model.load_weights(e_parm.folder_path / f"{e_parm.src_model_name}.weights.h5")
        pred = self.model.predict(e_parm.data_obj.data)
        eval_dict = create_evaluation_dict(self.model_name, e_parm.src_model_name, pred, e_parm.data_obj.label)
        return eval_dict
