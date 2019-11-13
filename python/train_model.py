"""
Usage:
  train_model.py [options] <source> <output>

Options:
  --epochs=N  Number of training epochs. [default: 25]
  --steps=N   Number of training steps per epochs. [default: 0]
  -h, --help  Show this screen.
"""

from docopt import docopt

from datetime import datetime
from pathlib import Path

from datasets import load_datasets, save_labels
from hparams import HP_DROPOUT, HP_NUM_UNITS_FC1, HP_NUM_UNITS_FC2, HP_OPTIMIZER
from models import compile_model, create_model, fit_model


LEARNING_RATE = 0.0005
IMAGE_SIZE = 224


def train_model(datasets, metadata, epochs, steps_per_epoch, hparams):
    _, _, test_dataset = datasets

    num_classes = len(metadata.labels)
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    model, base_model = create_model(input_shape, num_classes, hparams)
    model = compile_model(model, LEARNING_RATE, hparams)

    # model.evaluate(test_dataset)
    model, history = fit_model(model, datasets, metadata, epochs, steps_per_epoch)
    metrics = model.evaluate(test_dataset)

    return model, history, metrics


if __name__ == "__main__":
    args = docopt(__doc__)
    source_dir = Path(args["<source>"])
    output_dir = Path(args["<output>"])
    epochs = int(args["--epochs"])
    steps_per_epoch = int(args["--steps"])

    default_hparams = {
        HP_NUM_UNITS_FC1: 512,
        HP_NUM_UNITS_FC2: 64,
        HP_DROPOUT: 0.5,
        HP_OPTIMIZER: "rmsprop",
    }

    train_dataset, valid_dataset, test_dataset, metadata = load_datasets(source_dir)
    datasets = (train_dataset, valid_dataset, test_dataset)

    model, history, (loss, accuracy) = train_model(
        datasets, metadata, epochs, steps_per_epoch, default_hparams
    )

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
