import os
import random

import numpy as np
import tensorflow as tf


def ensure_reproducibility(*, seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "true"
