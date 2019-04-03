from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
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
    file_dir = Path(__file__).parent.resolve()
    root_dir = file_dir.parent

    # TODO: Make these configurable?
    data_dir = root_dir / "data"
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"

    # Fetch label names, and a map from names to indices.
    label_names, label_to_index = process_labels(train_dir)

    # Prepare batched training dataset.
    train_batches, train_count = batched_dataset(train_dir, label_to_index)

    # Load a pre-trained base model to use for feature extraction.
    base_model = VGG16(include_top=False, weights="imagenet", pooling="max")
    base_model.trainable = False

    # Create model by stacking a prediction layer on top of the base model.
    prediction_layer = keras.layers.Dense(1)
    model = keras.Sequential([base_model, prediction_layer])
    model.compile(optimizer=RMSprop(lr=0.0005), loss="binary_crossentropy", metrics=["accuracy"])
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
        train_batches.repeat(), epochs=initial_epochs, steps_per_epoch=steps_per_epoch
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
    timestamp = timestamp.strftime("Ymd")
    model_file = output_dir / "nemo-{}-{:.2f}.h5".format(timestamp, accuracy)
    model.save(str(model_file), include_optimizer=False)
