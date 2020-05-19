"""
Usage:
  train_models.py [options] <source> <output>

Options:
  --config=PATH  Path to hyperparameter configuration file. [default: config/hparams.yaml]
  --epochs=N     Number of training epochs. [default: 25]
  --steps=N      Number of training steps per epochs. [default: 0]
  -h, --help     Show this screen.
"""

from docopt import docopt

from pathlib import Path

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from nemo.datasets import load_datasets
from nemo.hparams import HParams, parse_config_file
from nemo.train_model import train_model
from nemo.utils import ensure_reproducibility


METRIC_ACCURACY = "accuracy"


def run_trial(source_dir, run_dir, hparams, epochs, steps_per_epoch):
    train_dataset, valid_dataset, test_dataset, metadata = load_datasets(source_dir)
    datasets = (train_dataset, valid_dataset, test_dataset)

    with tf.summary.create_file_writer(str(run_dir)).as_default():
        hp.hparams(hparams._asdict())
        _, _, (_, accuracy) = train_model(datasets, metadata, epochs, steps_per_epoch, hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


def main(source_dir, output_dir, config_file, epochs, steps_per_epoch):
    run_dir = Path(output_dir / "logs/hparam_tuning")

    hparams = parse_config_file(config_file)
    with tf.summary.create_file_writer(str(run_dir)).as_default():
        hp.hparams_config(
            hparams=hparams, metrics=[hp.Metric(METRIC_ACCURACY, display_name="accuracy")],
        )

    run_num = 0
    # TODO: Replace nested loops with zip or similar.
    for num_units_fc1 in hparams.num_units_fc1.domain.values:
        for num_units_fc2 in hparams.num_units_fc2.domain.values:
            for dropout_rate in hparams.dropout.domain.values:
                for optimizer in hparams.optimizer.domain.values:
                    hparams = HParams(num_units_fc1, num_units_fc2, dropout_rate, optimizer)

                    run_num += 1
                    run_name = "run-{}".format(run_num)
                    print("--- Starting trial:", run_name)
                    print({k: v for k, v in hparams._asdict().items()})

                    run_trial(source_dir, run_dir / run_name, hparams, epochs, steps_per_epoch)


if __name__ == "__main__":
    args = docopt(__doc__)
    source_dir = Path(args["<source>"])
    output_dir = Path(args["<output>"])
    config_file = Path(args["--config"])
    epochs = int(args["--epochs"])
    steps_per_epoch = int(args["--steps"])

    # Use fixed seeds and deterministic ops.
    ensure_reproducibility(seed=42)

    main(source_dir, output_dir, config_file, epochs, steps_per_epoch)
