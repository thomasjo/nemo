import argparse

from datetime import datetime
from pathlib import Path

import tensorflow as tf

from nemo.datasets import load_datasets, save_labels
from nemo.hparams import get_default_hparams
from nemo.models import compile_model, create_model, evaluate_model, fit_model
from nemo.utils import ensure_reproducibility, ignore_warnings


def main(args):
    # Use fixed seeds and deterministic ops.
    ensure_reproducibility(seed=42)

    if args.dev_mode:
        args.epochs = 2
        args.steps = 2

    # Ensure output directory exists.
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, valid_dataset, test_dataset, metadata = load_datasets(args)
    datasets = (train_dataset, valid_dataset, test_dataset)
    hparams = get_default_hparams()

    model, history, (loss, accuracy) = train_model(datasets, metadata, args.epochs, args.steps, hparams, args.image_size)

    # Save trained model.
    timestamp = datetime.utcnow()
    timestamp = timestamp.strftime("%Y-%m-%d-%H%M")
    model_file = args.output_dir / "nemo--{}--{:.2f}".format(timestamp, accuracy)
    model.save(model_file)

    # Save labels used by the trained model.
    label_file = model_file.with_suffix(".yaml")
    save_labels(label_file, metadata.labels)


def train_model(datasets, metadata, epochs, steps, hparams, image_size):
    num_classes = len(metadata.labels)
    input_shape = (image_size, image_size, 3)

    learning_rate = 1e-5

    model = create_model(input_shape, num_classes, hparams)
    model = compile_model(model, learning_rate, hparams)
    model, history = fit_model(model, datasets, metadata, epochs, steps)
    metrics = evaluate_model(model, datasets, steps)

    return model, history, metrics


def limit_cpu_threads(num_threads=0):
    tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(num_threads)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, metavar="PATH", required=True, help="path to directory containing partitioned training images")
    parser.add_argument("--output-dir", type=Path, metavar="PATH", required=True, help="path to output directory for storing training artifacts")
    parser.add_argument("--num-workers", type=int, metavar="INT", default=4, help="number of parallel worker CPU threads to use")
    parser.add_argument("--epochs", type=int, metavar="INT", default=25, help="number of training epochs")
    parser.add_argument("--steps", type=int, metavar="INT", default=None, help="number of training steps per epoch")
    parser.add_argument("--batch-size", type=int, metavar="INT", default=32, help="size of batches used during training and inference")
    parser.add_argument("--image-size", type=int, metavar="INT", default=224, help="target width and height of images after pre-processing")
    parser.add_argument("--dev-mode", action="store_true", help="enable fast development mode")

    return parser.parse_args()


if __name__ == "__main__":
    # Suppress warnings from TensorFlow...
    # TODO(thomasjo): Extract as function.
    tf.get_logger().addFilter("WARNING")
    ignore_warnings(tf)

    main(parse_args())
