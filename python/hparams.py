from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.optimizers import Adam, RMSprop


HP_NUM_UNITS_FC1 = hp.HParam("num_units_fc1", hp.Discrete([256, 384, 512]))
HP_NUM_UNITS_FC2 = hp.HParam("num_units_fc2", hp.Discrete([16, 32, 64]))
HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.2, 0.3, 0.4, 0.5]))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam", "rmsprop"]))


OPTIMIZER_FN = {
    "adam": Adam,
    "rmsprop": RMSprop,
}


def get_optimizer(name, learning_rate):
    # Sanity check.
    assert sorted(HP_OPTIMIZER.domain.values) == sorted(OPTIMIZER_FN.keys()), "Missing optimizer"

    optimizer_fn = OPTIMIZER_FN[name]
    optimizer = optimizer_fn(learning_rate)

    return optimizer
