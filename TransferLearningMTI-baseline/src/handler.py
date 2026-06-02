import logging
from pathlib import Path
from functools import wraps
from time import time


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        time_log = get_logger('time_logger')
        ts = time()
        result = f(*args, **kw)
        te = time()
        t_time = te - ts
        time_log.debug(f"func: {f.__name__}, args {args}, time: {round(t_time)} sec")
        return result

    return wrap


def get_logger(logger_name):
    if logger_name in logging.Logger.manager.loggerDict:
        return logging.getLogger(logger_name)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))

    log_path = Path(__file__).resolve().parents[1] / f"{logger_name}.log"
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s;%(message)s", "%Y-%m-%d %H:%M:%S"))
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger
