import os
import random

from types import ModuleType
from warnings import filterwarnings

import numpy as np
import tensorflow as tf


def ensure_reproducibility(*, seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "true"


def ignore_warnings(module: ModuleType):
    module_name = module.__name__
    filterwarnings("ignore", module=f"{module_name}.*")
