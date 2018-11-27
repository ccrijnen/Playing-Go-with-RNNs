"""General utility functions"""

import json
import logging
import tensorflow as tf


def set_logger(log_dir):
    """Sets the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Args:
        log_dir: (string) where to log
    """
    logger = logging.getLogger('tensorflow')
    logger.setLevel(logging.INFO)

    # Logging to a file
    file_handler = logging.FileHandler(log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
    logger.addHandler(file_handler)

    tf.logging.set_verbosity(tf.logging.INFO)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_dicts_to_json(dicts, json_path):
    """Saves multiple dicts of ints in json file
    Args:
        dicts: (dict) of (dict) of int-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to int for json (it doesn't accept np.array, np.float, )
        for k, d in dicts.items():
            d = {kk: int(v) for kk, v in d.items()}
            dicts[k] = d
        json.dump(dicts, f, indent=4)
