from pathlib import Path

from src.data.data_class import DataOrg


class TrainModelParam:
    folder_path: Path
    data_obj: DataOrg
    part_train: int
    to_save: bool
    src_model_to_load: str
    epochs : int

    def __init__(self, folder_path, data_obj=None, part_train=None, to_save=True, src_name_to_load=None, epochs=100):
        self.folder_path = folder_path
        self.data_obj = data_obj
        self.part_train = part_train
        self.to_save = to_save
        self.src_model_to_load = src_name_to_load
        self.epochs = epochs


class EvalModelParam:
    folder_path: Path = None
    data_obj: DataOrg
    src_model_name: str
    to_load: bool

    def __init__(self, folder_path, test_data=None, model_name=None, to_load=False):
        self.folder_path = folder_path
        self.test_data = test_data
        self.src_model_name = model_name
        self.to_load = to_load
