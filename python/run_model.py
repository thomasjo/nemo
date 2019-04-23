import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from datasets import labels_for_dir
from images import load_and_preprocess_image

# Used for auto-tuning dataset prefetch size, etc.
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = 224


if __name__ == "__main__":
    file_dir = Path(__file__).parent.resolve()
    root_dir = file_dir.parent

    # TODO: Make these configurable?
    data_dir = root_dir / "data"
    mixed_dir = data_dir / "test-mixed"
    output_dir = root_dir / "output"

    # Load trained model.
    # TODO: Make model path configurable.
    model_file = output_dir / "nemo.h5"
    model = keras.models.load_model(str(model_file), compile=False)
    # model.summary()

    # Prepare dataset.
    mixed_files = sorted([str(file) for file in mixed_dir.rglob("*.png")])
    mixed_dataset = tf.data.Dataset.from_tensor_slices(mixed_files)
    mixed_dataset = mixed_dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    mixed_dataset = mixed_dataset.batch(BATCH_SIZE)
    mixed_dataset = mixed_dataset.prefetch(AUTOTUNE)

    # Extract class predictions.
    predictions = model.predict(mixed_dataset)
    predictions = np.argmax(predictions, axis=1)

    # Prepare directory for predictions.
    result_dir = root_dir / "output" / "predictions"
    shutil.rmtree(result_dir, ignore_errors=True)
    result_dir.mkdir(parents=True)

    labels = labels_for_dir(data_dir / "train")
    assert sorted(labels.values()) == list(range(len(labels)))  # Sanity check

    # Prepare sub-directories for all dataset labels.
    label_dirs = {}
    for name, i in labels.items():
        label_dir = result_dir / name
        label_dir.mkdir(parents=True, exist_ok=True)
        label_dirs[i] = label_dir

    # Write copies of classified images in label-based directory scheme.
    for i, image_path in enumerate(mixed_files):
        image_label = predictions[i]
        image_bytes = Path(mixed_files[i]).read_bytes()
        target_file = label_dirs[image_label] / Path(image_path).name
        target_file.write_bytes(image_bytes)
