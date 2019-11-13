"""
Usage:
  train_models.py [options] <source> <output>

Options:
  --epochs=N  Number of training epochs. [default: 25]
  --steps=N   Number of training steps per epochs. [default: 0]
  -h, --help  Show this screen.
"""

from docopt import docopt

from pathlib import Path

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

from datasets import load_datasets
from hparams import HP_DROPOUT, HP_NUM_UNITS_FC1, HP_NUM_UNITS_FC2, HP_OPTIMIZER
from train_model import train_model


METRIC_ACCURACY = "accuracy"


def run_trial(run_dir, hparams, source_dir, epochs, steps_per_epoch):
    train_dataset, valid_dataset, test_dataset, metadata = load_datasets(source_dir)
    datasets = (train_dataset, valid_dataset, test_dataset)

    with tf.summary.create_file_writer(str(run_dir)).as_default():
        hp.hparams(hparams)
        _, _, (_, accuracy) = train_model(datasets, metadata, epochs, steps_per_epoch, hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


def main(source_dir, output_dir, epochs, steps_per_epoch):
    run_dir = Path(output_dir / "logs/hparam_tuning")

    with tf.summary.create_file_writer(str(run_dir)).as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS_FC1, HP_NUM_UNITS_FC2, HP_DROPOUT, HP_OPTIMIZER],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name="accuracy")],
        )

    run_num = 0
    # TODO: Replace nested loops with zip or similar.
    for num_units_fc1 in HP_NUM_UNITS_FC1.domain.values:
        for num_units_fc2 in HP_NUM_UNITS_FC2.domain.values:
            for dropout_rate in HP_DROPOUT.domain.values:
                for optimizer in HP_OPTIMIZER.domain.values:
                    hparams = {
                        HP_NUM_UNITS_FC1: num_units_fc1,
                        HP_NUM_UNITS_FC2: num_units_fc2,
                        HP_DROPOUT: dropout_rate,
                        HP_OPTIMIZER: optimizer,
                    }

                    run_num += 1
                    run_name = "run-{}".format(run_num)
                    print("--- Starting trial:", run_name)
                    print({h.name: hparams[h] for h in hparams})

                    run_trial(run_dir / run_name, hparams, source_dir, epochs, steps_per_epoch)


if __name__ == "__main__":
    args = docopt(__doc__)
    source_dir = Path(args["<source>"])
    output_dir = Path(args["<output>"])
    epochs = int(args["--epochs"])
    steps_per_epoch = int(args["--steps"])

    main(source_dir, output_dir, epochs, steps_per_epoch)
