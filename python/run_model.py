"""
Usage:
  run_model.py [options] <source> <output> <model>

Options:
  -h, --help  Show this screen.
"""

from docopt import docopt

import shutil
from datetime import datetime
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras

import numpy as np

from datasets import read_labels
from images import load_and_preprocess_image

# Used for auto-tuning dataset prefetch size, etc.
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = 224


def main(source_dir, output_dir, model_file):
    # Prepare dataset.
    files = sorted([str(file) for file in source_dir.rglob("*.png")])
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    # Read labels associated with the trained model.
    label_file = model_file.with_suffix(".yaml")
    labels = read_labels(label_file)

    # Load trained model.
    model = keras.models.load_model(str(model_file), compile=False)
    # model.summary()

    # Extract softmax class predictions.
    predictions = model.predict(dataset)

    # Convert softmax predictions to "hard" predictions.
    predictions = np.argmax(predictions, axis=1)

    # Prepare output directory for predictions.
    timestamp = datetime.utcnow()
    timestamp = timestamp.strftime("%Y-%m-%d-%H%M")
    result_dir = output_dir / "predictions" / timestamp
    result_dir.mkdir(parents=True)

    # Prepare sub-directories for all dataset labels.
    label_dirs = {}
    for name, i in labels.items():
        label_dir = result_dir / name
        label_dir.mkdir(parents=True, exist_ok=True)
        label_dirs[i] = label_dir

    # Write copies of classified images in label-based directory scheme.
    for i, image_path in enumerate(files):
        image_label = predictions[i]
        image_bytes = Path(files[i]).read_bytes()
        target_file = label_dirs[image_label] / Path(image_path).name
        target_file.write_bytes(image_bytes)


if __name__ == "__main__":
    args = docopt(__doc__)
    source_dir = Path(args["<source>"])
    output_dir = Path(args["<output>"])
    model_file = Path(args["<model>"])

    main(source_dir, output_dir, model_file)
