from collections import namedtuple
from pathlib import Path

from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.optimizers import Adam, RMSprop

import yaml


OPTIMIZER_FN = {
    "adam": Adam,
    "rmsprop": RMSprop,
}


HParams = namedtuple("HParams", ["num_units_fc1", "num_units_fc2", "dropout", "optimizer"])


def parse_config_file(config_file):
    with config_file.open() as f:
        config_obj = yaml.safe_load(f)
        return parse_config(config_obj)


def parse_config(config_obj):
    return HParams(
        _hparam(config_obj, "num_units_fc1"),
        _hparam(config_obj, "num_units_fc2"),
        _hparam(config_obj, "dropout"),
        _hparam(config_obj, "optimizer"),
    )


def get_optimizer(name, learning_rate):
    optimizer_fn = OPTIMIZER_FN[name]
    optimizer = optimizer_fn(learning_rate)

    return optimizer


def _hparam(config_obj, name, display_name=None):
    return hp.HParam(name, hp.Discrete(config_obj[name]), display_name)
