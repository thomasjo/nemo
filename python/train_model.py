import shutil

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import RMSprop

# Used for auto-tuning dataset prefetch size, etc.
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32


def preprocess_image(image):
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize_with_pad(image, 224, 224)
    image /= 255.0

    return image


def load_and_preprocess_image(path, label=None):
    image = tf.io.read_file(path)
    image = preprocess_image(image)

    if label is None:
        return image
    return image, label


def image_paths(path):
    image_paths = []

    for file in sorted(path.rglob("*.png")):
        image_paths.append(str(file))

    return image_paths


def image_paths_and_labels(path, label_to_index):
    image_paths = []
    image_labels = []

    for file in sorted(path.rglob("*.png")):
        image_paths.append(str(file))
        image_labels.append(label_to_index[file.parent.name])

    return image_paths, image_labels


def batched_dataset(source_dir, label_to_index=None, shuffle=True):
    if label_to_index is None:
        paths = image_paths(source_dir)
        dataset = tf.data.Dataset.from_tensor_slices((paths))
    else:
        paths, labels = image_paths_and_labels(source_dir, label_to_index)
        dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    count = len(paths)

    if shuffle:
        dataset = dataset.shuffle(count)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)

    return dataset, count


def process_labels(source_dir):
    label_names = sorted(path.name for path in source_dir.glob("*/") if path.is_dir())
    label_to_index = dict((name, index) for index, name in enumerate(label_names))

    return label_names, label_to_index


if __name__ == "__main__":
    file_dir: Path = Path(__file__).parent.resolve()
    root_dir = file_dir.parent

    # TODO: Make these configurable?
    data_dir: Path = root_dir / "data"
    train_dir: Path = data_dir / "train"
    test_dir: Path = data_dir / "test"

    # Split training data into training and validation sets.
    # TODO: Find a better way of doing this.
    orig_train_dir = train_dir
    train_dir = train_dir.with_name("train-tmp")
    valid_dir = data_dir / "valid-tmp"

    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(valid_dir, ignore_errors=True)

    train_dir.mkdir(exist_ok=True)
    valid_dir.mkdir(exist_ok=True)

    image_paths = list(orig_train_dir.rglob("*.png"))
    image_count = len(image_paths)
    train_count = round(image_count * 0.85)
    valid_count = image_count - train_count

    shuff_idx = np.random.permutation(image_count)
    train_idx = shuff_idx[:train_count]
    valid_idx = shuff_idx[train_count:]

    assert train_count == len(train_idx)
    assert valid_count == len(valid_idx)

    for i in train_idx.flat:
        old_path: Path = image_paths[i]
        new_path = train_dir / old_path.parent.name / old_path.name
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, new_path)
    for i in valid_idx.flat:
        old_path: Path = image_paths[i]
        new_path = valid_dir / old_path.parent.name / old_path.name
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old_path, new_path)

    # --

    # Fetch label names, and a map from names to indices.
    label_names, label_to_index = process_labels(train_dir)

    # Prepare batched training and validation datasets.
    train_batches, train_count = batched_dataset(train_dir, label_to_index)
    valid_batches, valid_count = batched_dataset(valid_dir, label_to_index)

    # Load a pre-trained base model to use for feature extraction.
    base_model = VGG16(include_top=False, weights="imagenet", pooling="max")
    base_model.trainable = False

    # Create model by stacking a prediction layer on top of the base model.
    prediction_layer = keras.layers.Dense(1)
    model = keras.Sequential([base_model, prediction_layer])

    # Prepare optimizer, loss function, and metrics.
    optimizer = RMSprop(lr=0.0005)
    loss = "binary_crossentropy"
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    print("\nEvaluating model before training...")
    loss0, accuracy0 = model.evaluate(train_batches)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    print("\nTraining model...")

    # Initial training parameters.
    initial_epochs = 50
    steps_per_epoch = train_count // BATCH_SIZE

    history = model.fit(
        train_batches.repeat(),
        validation_data=valid_batches,
        epochs=initial_epochs,
        steps_per_epoch=steps_per_epoch,
    )

    # Prepare batched test dataset.
    test_batches, _ = batched_dataset(test_dir, label_to_index, shuffle=False)

    print("\nEvaluating model after training...")
    loss, accuracy = model.evaluate(test_batches)
    print("final loss: {:.2f}".format(loss))
    print("final accuracy: {:.2f}".format(accuracy))

    output_dir = root_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow()
    timestamp = timestamp.strftime("%Y%m%d%H%M")
    model_file = output_dir / "nemo-{}-{:.2f}.h5".format(timestamp, accuracy)
    model.save(str(model_file), include_optimizer=False)
