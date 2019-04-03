from datetime import datetime
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

# Used for auto-tuning dataset prefetching parameters.
AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess_image(image):
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize_with_pad(image, 224, 224)
    image /= 255.0

    return image


def load_and_preprocess_image(path, label=None):
    image = tf.io.read_file(path)
    image = preprocess_image(image)

    if label is not None:
        return image, label

    return image


def image_paths_and_labels(path, label_to_index):
    image_paths = []
    image_labels = []

    for file in path.rglob("*.png"):
        image_paths.append(str(file))
        image_labels.append(label_to_index[file.parent.name])

    return image_paths, image_labels


if __name__ == "__main__":
    # TODO: Make this configurable via arguments using docopt.
    data_dir = Path("/root/data")
    test_dir = data_dir / "test-new"

    label_names = sorted(path.name for path in test_dir.glob("*/") if path.is_dir())
    print("label names:", label_names)

    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print(label_to_index)

    test_paths, test_labels = image_paths_and_labels(test_dir, label_to_index)
    test_count = len(test_paths)
    print("num test images:", test_count)

    raw_test = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))
    test_ds = raw_test.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    BATCH_SIZE = 32

    test_batches = test_ds.batch(BATCH_SIZE)
    test_batches = test_batches.prefetch(AUTOTUNE)
    print(test_batches)

    print()
    print()

    model = keras.models.load_model("/root/output/nemo.h5")
    model.summary()

    loss, accuracy = model.evaluate(test_batches)
    print("loss: {:.2f}".format(loss))
    print("accuracy: {:.2f}".format(accuracy))

    # --
    print()
    print()

    mixed_dir = data_dir / "test-mixed"
    mixed_paths = [str(file) for file in mixed_dir.rglob("*.png")]
    mixed_count = len(mixed_paths)
    print("num mixed images:", mixed_count)

    raw_mixed = tf.data.Dataset.from_tensor_slices(mixed_paths)
    mixed_ds = raw_mixed.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    mixed_batches = mixed_ds.batch(BATCH_SIZE)
    mixed_batches = mixed_batches.prefetch(AUTOTUNE)

    predictions = model.predict(mixed_batches)
    predictions[predictions >= 0] = 1
    predictions[predictions < 0] = 0
    predictions = predictions.astype(int)
    print(predictions)

    output_dir = Path("/root/output/predictions")
    benthos_dir = output_dir / "benthos"
    plankton_dir = output_dir / "plankton"

    benthos_dir.mkdir(parents=True, exist_ok=True)
    plankton_dir.mkdir(parents=True, exist_ok=True)

    for i in range(mixed_count):
        image_path = mixed_paths[i]
        image_file = Path(image_path)

        label = predictions[i, 0]
        label_name = label_names[label]
        if label_name == "benthos":
            target_file = benthos_dir / image_file.name
        else:
            target_file = plankton_dir / image_file.name

        target_file.write_bytes(image_file.read_bytes())
