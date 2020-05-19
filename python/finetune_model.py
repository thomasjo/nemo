"""
Usage:
  finetune_model.py [options] <source> <output> <model>

Options:
  --initial-epoch=N  Number of the initial epoch. [default: 0]
  --epochs=N         Number of fine-tuning epochs. [default: 25]
  --steps=N          Number of training steps per epochs. [default: 0]
  -h, --help         Show this screen.
"""

from docopt import docopt

from datetime import datetime
from pathlib import Path

from nemo.datasets import load_datasets, save_labels
from nemo.hparams import get_default_hparams
from nemo.models import compile_model, evaluate_model, fit_model, load_model


def finetune_model(model_file, initial_epoch, datasets, metadata, epochs, steps, hparams):
    # Load a pre-trained model.
    model, base_model = load_model(model_file)

    metrics = evaluate_model(model, datasets)

    # Only fine-tune the last few layers of the base model.
    base_model.trainable = True
    for layer in base_model.layers[:-8]:
        layer.trainable = False

    learning_rate = 0.000001
    model = compile_model(model, learning_rate, hparams)

    model, history = fit_model(model, datasets, metadata, epochs, steps, initial_epoch)
    metrics = evaluate_model(model, datasets)

    return model, history, metrics


if __name__ == "__main__":
    args = docopt(__doc__)
    source_dir = Path(args["<source>"])
    output_dir = Path(args["<output>"])
    model_file = Path(args["<model>"])
    initial_epoch = int(args["--initial-epoch"])
    epochs = int(args["--epochs"])
    steps = int(args["--steps"])
    image_size = int(args["--image-size"])

    hparams = get_default_hparams()

    train_dataset, valid_dataset, test_dataset, metadata = load_datasets(source_dir, image_size)
    datasets = (train_dataset, valid_dataset, test_dataset)

    model, history, (loss, accuracy) = finetune_model(
        model_file, initial_epoch, datasets, metadata, epochs, steps, hparams
    )

    # Ensure output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save fine-tuned model.
    timestamp = datetime.utcnow()
    timestamp = timestamp.strftime("%Y-%m-%d-%H%M")
    model_file = output_dir / "nemo-ft--{}--{:.2f}.h5".format(timestamp, accuracy)
    model.save(str(model_file))

    # Save labels used by the trained model.
    label_file = model_file.with_suffix(".yaml")
    save_labels(label_file, metadata.labels)
