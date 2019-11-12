"""
Usage:
  analyze_model.py [options] <source> <output> <model>

Options:
  -h, --help  Show this screen.
"""

from docopt import docopt

import pickle
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np
from tqdm import tqdm

from datasets import dataset_from_dir, read_labels
from images import load_and_preprocess_image
from layers import Dropout
from models import load_model

# Used for auto-tuning dataset prefetch size, etc.
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = 224


def prepare_pickle_data(labels, files, predictions, accuracies):
    return {"labels": labels, "files": files, "predictions": predictions, "accuracies": accuracies}


def main(source_dir, output_dir, model_file):
    # Read labels associated with the trained model.
    label_file = model_file.with_suffix(".yaml")
    labels = read_labels(label_file)

    # Prepare dataset.
    dataset, file_count, files = dataset_from_dir(source_dir, labels, return_files=True)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(32)
    dataset = dataset.prefetch(AUTOTUNE)

    # Load trained model.
    model, base_model = load_model(model_file)

    # Force dropout during inferrence.
    for layer in model.layers:
        if isinstance(layer, Dropout):
            layer.force = True

    mc_predictions = []
    mc_accuracies = []

    print("Collecting Monte Carlo predictions...")
    for _ in tqdm(range(100)):
        predictions = []
        accuraries = []
        for batch in dataset:
            batch_images, batch_labels = batch
            batch_predictions = model.predict(batch_images)
            batch_accuracy = keras.metrics.categorical_accuracy(batch_labels, batch_predictions)

            predictions.append(batch_predictions)
            accuraries.append(batch_accuracy)

        mc_predictions.append(np.concatenate(predictions, axis=0))
        mc_accuracies.append(np.concatenate(accuraries, axis=0))

    mc_predictions = np.array(mc_predictions)
    mc_accuracies = np.array(mc_accuracies)

    pickle_file = model_file.with_suffix(".pickle")
    with open(pickle_file, "wb") as f:
        payload = prepare_pickle_data(labels, files, mc_predictions, mc_accuracies)
        pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)

    # NOTE: Force an exit for the time being. Need to determine if we want or
    # need to do further analysis work as part of this script.
    exit(0)

    # Prepare output directory for predictions.
    timestamp = datetime.utcnow()
    timestamp = timestamp.strftime("%Y-%m-%d-%H%M")
    result_dir = output_dir / "analyses" / timestamp
    result_dir.mkdir(parents=True)


if __name__ == "__main__":
    args = docopt(__doc__)
    source_dir = Path(args["<source>"])
    output_dir = Path(args["<output>"])
    model_file = Path(args["<model>"])

    main(source_dir, output_dir, model_file)
