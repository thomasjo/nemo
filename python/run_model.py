import shutil

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from train_model import load_and_preprocess_image

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
    model_file = output_dir / "nemo.h5"
    model = keras.models.load_model(str(model_file), compile=False)
    model.summary()

    # Prepare training and validation datasets.
    mixed_files = sorted([str(file) for file in mixed_dir.rglob("*.png")])
    mixed_dataset = tf.data.Dataset.from_tensor_slices(mixed_files)
    mixed_dataset = mixed_dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    mixed_dataset = mixed_dataset.batch(BATCH_SIZE)
    mixed_dataset = mixed_dataset.prefetch(AUTOTUNE)

    predictions = model.predict(mixed_dataset)
    predictions[predictions >= 0] = 1  # planktic
    predictions[predictions < 0] = 0   # benthic
    predictions = predictions.astype(int)

    result_dir = root_dir / "output" / "predictions"
    shutil.rmtree(result_dir, ignore_errors=True)
    result_dir.mkdir(parents=True)

    # TODO: Create this by convention or some such.
    label_names = ["benthic", "planktic"]
    label_dirs = {}
    for label_name in label_names:
        label_dir = result_dir / label_name
        label_dir.mkdir(parents=True, exist_ok=True)
        label_dirs[label_name] = label_dir

    for i, image_path in enumerate(mixed_files):
        image_path = mixed_files[i]
        image_file = Path(image_path)

        label = predictions[i, 0]
        label_name = label_names[label]

        target_file = label_dirs[label_name] / image_file.name
        target_file.write_bytes(image_file.read_bytes())
