"""
Usage:
  training_suite.py [options] <source> <output>

Options:
  --repeat=N  Number of times to repeat the training block. [default: 5]
  --epochs=N  Number of training epochs. [default: 25]
  --steps=N   Number of training steps per epochs. [default: 0]
  -h, --help  Show this screen.
"""

from docopt import docopt

from datetime import datetime
from pathlib import Path

import numpy as np

from train_model import train_model
from finetune_model import finetune_model
from hparams import get_default_hparams
from datasets import load_datasets


if __name__ == "__main__":
    args = docopt(__doc__)
    source_dir = Path(args["<source>"])
    output_dir = Path(args["<output>"])
    repeat = int(args["--repeat"])
    epochs = int(args["--epochs"])
    steps = int(args["--steps"])

    train_dataset, valid_dataset, test_dataset, metadata = load_datasets(source_dir)
    datasets = (train_dataset, valid_dataset, test_dataset)
    hparams = get_default_hparams()

    timestamp = datetime.utcnow()
    timestamp = timestamp.strftime("%Y-%m-%d-%H%M")
    output_dir = output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    accuracies = []
    for k in range(repeat):
        print("--- Starting trial: ", str(k + 1))

        print("Training model...")
        model, history, (loss, accuracy) = train_model(datasets, metadata, epochs, steps, hparams)

        model_file = "run-{}--{:.2f}.h5".format(k + 1, accuracy)
        model.save(str(model_file))

        print("Fine-tuning model...")
        ft_model, ft_history, (ft_loss, ft_accuracy) = finetune_model(
            model_file, 0, datasets, metadata, epochs, steps, hparams
        )

        accuracies.append([accuracy, ft_accuracy])

    accuracies: np.ndarray = np.array(accuracies)
    np.save(str(output_dir / "results.npy"), accuracies)

    print(accuracies.shape)
    print(accuracies.mean(axis=0))
    print(accuracies.std(axis=0))
