"""
Usage:
  train_model.py [options] <source> <output>

Options:
  --epochs=N      Number of training epochs. [default: 25]
  --steps=N       Number of training steps per epochs. [default: 0]
  --image-size=S  Target width and height of images after pre-processing.  [default: 224]
  -h, --help      Show this screen.
"""

from docopt import docopt

from datetime import datetime
from pathlib import Path

import tensorflow as tf

from nemo.datasets import load_datasets, save_labels
from nemo.hparams import get_default_hparams
from nemo.models import compile_model, create_model, evaluate_model, fit_model
from nemo.utils import ensure_reproducibility


def train_model(datasets, metadata, epochs, steps, hparams, image_size=224):
    num_classes = len(metadata.labels)
    input_shape = (image_size, image_size, 3)

    model, base_model = create_model(input_shape, num_classes, hparams)

    learning_rate = 1e-4
    model = compile_model(model, learning_rate, hparams)

    model, history = fit_model(model, datasets, metadata, epochs, steps)
    metrics = evaluate_model(model, datasets)

    return model, history, metrics


def limit_cpu_threads():
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)


if __name__ == "__main__":
    args = docopt(__doc__)
    source_dir = Path(args["<source>"])
    output_dir = Path(args["<output>"])
    epochs = int(args["--epochs"])
    steps = int(args["--steps"])
    image_size = int(args["--image-size"])

    # Use fixed seeds and deterministic ops.
    ensure_reproducibility(seed=42)

    limit_cpu_threads()

    train_dataset, valid_dataset, test_dataset, metadata = load_datasets(source_dir)
    datasets = (train_dataset, valid_dataset, test_dataset)
    hparams = get_default_hparams()

    model, history, (loss, accuracy) = train_model(datasets, metadata, epochs, steps, hparams, image_size)

    # Ensure output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trained model.
    timestamp = datetime.utcnow()
    timestamp = timestamp.strftime("%Y-%m-%d-%H%M")
    model_file = output_dir / "nemo--{}--{:.2f}.h5".format(timestamp, accuracy)
    model.save(str(model_file))

    # Save labels used by the trained model.
    label_file = model_file.with_suffix(".yaml")
    save_labels(label_file, metadata.labels)
