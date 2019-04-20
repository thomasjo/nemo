import random
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path

import tensorflow as tf
from tensorflow import keras

from images import augment_image, load_and_preprocess_image

# Used for auto-tuning dataset prefetch size, etc.
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32


@contextmanager
def random_seed(seed):
    old_state = random.getstate()
    yield random.seed(seed)
    random.setstate(old_state)


def _root_dir():
    file_dir = Path(__file__).parent.resolve()
    root_dir = file_dir.parent
    return root_dir


def _data_dir():
    data_dir = _root_dir() / "data"
    return data_dir


def process_labels(source_dir):
    labels = sorted(path.name for path in source_dir.glob("*/") if path.is_dir())
    labels = dict((name, index) for index, name in enumerate(labels))
    return labels


def load_datasets():
    # TODO: Make these configurable/arguments?
    data_dir = _data_dir()
    train_dir: Path = data_dir / "train"
    test_dir: Path = data_dir / "test"

    # Fetch label names, and a map from names to indices.
    labels = process_labels(train_dir)
    num_classes = len(labels)
    assert num_classes > 2

    train_files = sorted([str(file) for file in train_dir.rglob("*.png")])
    # NOTE: Temporarily filter out "sediment" class.
    train_files = list(filter(lambda p: p.find("sediment") < 0, train_files))
    num_classes = num_classes - 1

    # Make the splitting reproducible by using a fixed seed.
    with random_seed(42):
        random.shuffle(train_files)

    # Read labels based on directory structure convention.
    train_labels = [labels[Path(file).parent.name] for file in train_files]
    train_labels = keras.utils.to_categorical(train_labels, num_classes)

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

    # Prepare batched test dataset.
    test_files = sorted([str(file) for file in test_dir.rglob("*.png")])
    # NOTE: Temporarily filter out "sediment" class.
    test_files = list(filter(lambda p: p.find("sediment") < 0, test_files))
    test_labels = [labels[Path(file).parent.name] for file in test_files]
    test_labels = keras.utils.to_categorical(test_labels, num_classes)

    # Prepare training and validation datasets.
    test_dataset = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    test_dataset = test_dataset.shuffle(len(test_files))
    test_dataset = test_dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE)
    test_dataset = test_dataset.prefetch(AUTOTUNE)

    Metadata = namedtuple("Metadata", ["train_count", "valid_count", "test_count"])
    metadata = Metadata(len(train_files), len(valid_files), len(test_files))

    return train_dataset, valid_dataset, test_dataset, metadata
