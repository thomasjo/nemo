"""
Usage:
  extract_weights.py <model> <output>

Options:
  -h --help  Show this screen.
"""

from pathlib import Path

import tensorflow.keras as keras
from docopt import docopt


def extract_weights(model_file, output_file):
    model = keras.models.load_model(str(model_file), compile=False)
    model.save_weights(str(output_file))

    base_model = model.layers[0]
    base_model.save_weights(str(output_file.with_suffix(".base.h5")))


if __name__ == "__main__":
    args = docopt(__doc__)
    model_file = Path(args["<model>"])
    output_file = Path(args["<output>"])

    extract_weights(model_file, output_file)
