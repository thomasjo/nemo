"""
Usage:
  train_model.py [options] <source> <output>

Options:
  --epochs=N  Number of training epochs. [default: 25]
  -h, --help  Show this screen.
"""

from docopt import docopt

from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, GlobalMaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import RMSprop

from datasets import load_datasets, save_labels
from layers import Dropout

# Used for auto-tuning dataset prefetch size, etc.
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = 224


def main(source_dir, output_dir, epochs):
    train_dataset, valid_dataset, test_dataset, metadata = load_datasets(source_dir)
    num_classes = len(metadata.labels)

    # Load a pre-trained base model to use for feature extraction.
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False

    # Create model by stacking a prediction layer on top of the base model.
    model = keras.Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5, force=True))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5, force=True))
    model.add(Dense(num_classes, activation="softmax"))

    # Prepare optimizer, loss function, and metrics.
    learning_rate = 0.0005
    optimizer = RMSprop(learning_rate)
    loss = CategoricalCrossentropy()
    metrics = [CategoricalAccuracy()]

    print("Compiling model...")
    model.compile(optimizer, loss, metrics)

    print("Evaluating model before training...")
    loss0, accuracy0 = model.evaluate(test_dataset)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    # Initial training parameters.
    initial_epochs = epochs
    steps_per_epoch = metadata.train_count // BATCH_SIZE
    steps_per_epoch *= 4  # Increase steps because of image augmentations

    # Use early stopping to prevent overfitting, etc.
    early_stopping = EarlyStopping(patience=2, restore_best_weights=True)

    print("Training model...")
    history = model.fit(
        train_dataset.repeat(),
        validation_data=valid_dataset,
        epochs=initial_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[early_stopping],
    )

    print("\nEvaluating model after training...")
    loss, accuracy = model.evaluate(test_dataset)
    print("final loss: {:.2f}".format(loss))
    print("final accuracy: {:.2f}".format(accuracy))

    # Ensure output directory exists.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trained model.
    timestamp = datetime.utcnow()
    timestamp = timestamp.strftime("%Y-%m-%d-%H%M")
    model_file = output_dir / "nemo--{}--{:.2f}.h5".format(timestamp, accuracy)
    model.save(str(model_file))

    # Save labels used by the trained model.
    label_file = model_file.with_suffix(".yaml")
    save_labels(label_file, metadata.labels)


if __name__ == "__main__":
    args = docopt(__doc__)
    source_dir = Path(args["<source>"])
    output_dir = Path(args["<output>"])
    epochs = int(args["--epochs"])

    main(source_dir, output_dir, epochs)
