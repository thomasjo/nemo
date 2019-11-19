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
from hparams import get_default_hparams
from models import compile_model, create_model, evaluate_model, fit_model


def train_model(datasets, metadata, epochs, steps, hparams):
    num_classes = len(metadata.labels)
    input_shape = (224, 224, 3)

    model, base_model = create_model(input_shape, num_classes, hparams)

    learning_rate = 0.0005
    model = compile_model(model, learning_rate, hparams)

    model, history = fit_model(model, datasets, metadata, epochs, steps)
    metrics = evaluate_model(model, datasets)

    return model, history, metrics


if __name__ == "__main__":
    args = docopt(__doc__)
    source_dir = Path(args["<source>"])
    output_dir = Path(args["<output>"])
    epochs = int(args["--epochs"])
    steps = int(args["--steps"])

    train_dataset, valid_dataset, test_dataset, metadata = load_datasets(source_dir)
    datasets = (train_dataset, valid_dataset, test_dataset)
    hparams = get_default_hparams()

    model, history, (loss, accuracy) = train_model(datasets, metadata, epochs, steps, hparams)

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
