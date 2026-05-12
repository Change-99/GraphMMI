import xgboost as xgb
from keras import Model
from keras.layers import Dense, Dropout, Input
from tensorflow.keras import regularizers

from src.models.training.base_model import BaseTrainObj
from src.models.training.xgb_model import XgboostTrainObj


def network_model(shape):
    x = Input(shape=(shape,), name="input")
    _model = Dense(300, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activity_regularizer=regularizers.l2(1e-5),name="dense_300")(x)
    _model = Dropout(rate=0.6,name="dropout_1")(_model)
    _model = Dense(200, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                   bias_regularizer=regularizers.l2(1e-4),
                   activity_regularizer=regularizers.l2(1e-5),name="dense_200")(_model)
    _model = Dropout(rate=0.6,name="dropout_2")(_model)
    _model = Dense(100, activation='relu',name="dense_100")(_model)
    _model = Dropout(rate=0.6,name="dropout_3")(_model)
    _model = Dense(20, activation='relu',name="dense_20")(_model)
    _model = Dense(1, activation='sigmoid',name="output")(_model)
    model = Model(x, _model, name="ann_model")
    return model


XGBS_PARAMS = {
    "objective": "binary:logistic",
    "booster": "gbtree",
    "learning_rate": 0.1,
    "gamma": 0.5,
    "max_depth": 2,
    "min_child_weight": 1,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "reg_lambda": 1,
    "n_jobs": -1,
    "verbosity": 0,
    "eval_metric": ["error", "logloss"],
    "random_state": 42,
}


def get_models_dict(shape):
    base_model = BaseTrainObj()
    base_model.set_model(network_model(shape))
    xbg_model = XgboostTrainObj()
    xbg_model.set_model(xgb.XGBClassifier(**XGBS_PARAMS))
    model_dict = {'base': base_model, 'xgb': xbg_model}
    return model_dict
