from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import RMSprop

from datasets import load_datasets

# Used for auto-tuning dataset prefetch size, etc.
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = 224


if __name__ == "__main__":
    train_dataset, valid_dataset, test_dataset, metadata = load_datasets()
    num_classes = len(metadata.labels)

    # Load a pre-trained base model to use for feature extraction.
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False
    # base_model.summary()

    # Create model by stacking a prediction layer on top of the base model.
    model = keras.Sequential()
    model.add(base_model)
    model.add(GlobalMaxPooling2D())
    model.add(Dense(64, activation="relu"))
    model.add(Dense(num_classes, activation="softmax"))

    # Prepare optimizer, loss function, and metrics.
    learning_rate = 0.0005
    optimizer = RMSprop(learning_rate)
    loss = CategoricalCrossentropy()
    metrics = [CategoricalAccuracy()]

    model.compile(optimizer, loss, metrics)
    # model.summary()

    print("\nEvaluating model before training...")
    loss0, accuracy0 = model.evaluate(train_dataset)
    print("initial loss: {:.2f}".format(loss0))
    print("initial accuracy: {:.2f}".format(accuracy0))

    print("\nTraining model...")

    # Initial training parameters.
    initial_epochs = 25
    steps_per_epoch = (metadata.train_count // BATCH_SIZE) * 4

    history = model.fit(
        train_dataset.repeat(),
        validation_data=valid_dataset,
        epochs=initial_epochs,
        steps_per_epoch=steps_per_epoch,
    )

    print("\nEvaluating model after training...")
    loss, accuracy = model.evaluate(test_dataset)
    print("final loss: {:.2f}".format(loss))
    print("final accuracy: {:.2f}".format(accuracy))

    # Initialize output directory.
    # NOTE: Make this configurable?
    file_dir = Path(__file__).parent.resolve()
    root_dir = file_dir.parent
    output_dir = root_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.utcnow()
    timestamp = timestamp.strftime("%Y-%m-%d-%H%M")
    model_file = output_dir / "nemo--{}--{:.2f}.h5".format(timestamp, accuracy)
    model.save(str(model_file))
