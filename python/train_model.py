import random

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
IMAGE_SIZE = 224


def preprocess_image(image):
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image


def load_and_preprocess_image(path, *args):
    image = tf.io.read_file(path)
    image = preprocess_image(image)

    return (image, *args)


def augment_image(image, *args):
    # Random horizontal flipping.
    image = tf.image.random_flip_left_right(image)

    # Random rotation in increments of 90 degrees.
    rot_k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=rot_k)

    # Random light distortion.
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.random_hue(image, 0.05)
    image = tf.image.random_saturation(image, 0.9, 1.1)

    # Ensure image is still valid.
    image = tf.clip_by_value(image, 0.0, 1.0)

    return (image, *args)


def image_paths(path):
    image_paths = []

    for file in sorted(path.rglob("*.png")):
        image_paths.append(str(file))

    return image_paths


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

    # Fetch label names, and a map from names to indices.
    label_names, label_to_index = process_labels(train_dir)
    print(label_to_index)

    train_files = sorted([str(file) for file in train_dir.rglob("*.png")])
    train_files = list(filter(lambda p: p.find("sediment") < 0, train_files))

    # Make the splitting reproducible.
    random.seed(42)
    random.shuffle(train_files)

    # Read labels based on directory structure convention.
    train_labels = [label_to_index[Path(file).parent.name] for file in train_files]

    # Split training data into training and validation sets.
    split = round(0.85 * len(train_files))
    valid_files, valid_labels = train_files[split:], train_labels[split:]
    train_files, train_labels = train_files[:split], train_labels[:split]

    # Prepare training and validation datasets.
    train_dataset = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_dataset = train_dataset.shuffle(len(train_files))
    train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.map(augment_image, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.prefetch(AUTOTUNE)

    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_files, valid_labels))
    valid_dataset = valid_dataset.shuffle(len(valid_files))
    valid_dataset = valid_dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    valid_dataset = valid_dataset.batch(BATCH_SIZE)
    valid_dataset = valid_dataset.prefetch(AUTOTUNE)

    # Load a pre-trained base model to use for feature extraction.
    base_model = VGG16(include_top=False, weights="imagenet")
    base_model.trainable = False
    base_model.summary()

    # Create model by stacking a prediction layer on top of the base model.
    pooling_layer = keras.layers.GlobalMaxPooling2D()
    prediction_layer = keras.layers.Dense(1)
    model = keras.Sequential([base_model, pooling_layer, prediction_layer])

    # Prepare optimizer, loss function, and metrics.
    base_learning_rate = 0.0005
    optimizer = RMSprop(lr=base_learning_rate)
    loss = "binary_crossentropy"
    metrics = ["accuracy"]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    print("\nEvaluating model before training...")
    loss0, accuracy0 = model.evaluate(train_dataset)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    print("\nTraining model...")

    # Initial training parameters.
    initial_epochs = 100
    steps_per_epoch = len(train_files) // BATCH_SIZE

    history = model.fit(
        train_dataset.repeat(),
        validation_data=valid_dataset,
        epochs=initial_epochs,
        steps_per_epoch=steps_per_epoch,
    )

    # Prepare batched test dataset.
    test_files = sorted([str(file) for file in test_dir.rglob("*.png")])
    test_files = list(filter(lambda p: p.find("sediment") < 0, test_files))
    test_labels = [label_to_index[Path(file).parent.name] for file in test_files]

    # Prepare training and validation datasets.
    test_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    test_dataset = test_dataset.shuffle(len(test_files))
    test_dataset = test_dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    print("\nEvaluating model after training...")
    loss, accuracy = model.evaluate(test_dataset)
    print("final loss: {:.2f}".format(loss))
    print("final accuracy: {:.2f}".format(accuracy))

    output_dir = root_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow()
    timestamp = timestamp.strftime("%Y-%m-%d-%H%M")
    model_file = output_dir / "nemo--{}--{:.2f}.h5".format(timestamp, accuracy)
    model.save(str(model_file))
