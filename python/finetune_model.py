from datetime import datetime
from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import RMSprop

from datasets import load_datasets

# Used for auto-tuning dataset prefetch size, etc.
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = 224


if __name__ == "__main__":
    file_dir: Path = Path(__file__).parent.resolve()
    root_dir = file_dir.parent

    # TODO: Make these configurable?
    output_dir = root_dir / "output"
    train_dataset, valid_dataset, test_dataset, metadata = load_datasets()

    # Load a pre-trained model.
    model_file = output_dir / "nemo.h5"
    model = keras.models.load_model(str(model_file), compile=False)

    # Prepare optimizer, loss function, and metrics.
    base_learning_rate = 0.00001
    optimizer = RMSprop(lr=base_learning_rate)
    loss = "binary_crossentropy"
    metrics = ["accuracy"]

    # Only fine-tune the last few layers of the base model.
    base_model = model.layers[0]
    for layer in base_model.layers[:-4]:
        layer.trainable = False
    for layer in base_model.layers:
        print(layer.name, layer.trainable)

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.summary()

    print("Fine-tuning model...")

    # Initial training parameters.
    initial_epochs = 100
    fine_tune_epochs = 100
    total_epochs = initial_epochs + fine_tune_epochs
    steps_per_epoch = metadata.train_count // BATCH_SIZE
    steps_per_epoch *= 4  # Increase steps because of image augmentations

    history = model.fit(
        train_dataset.repeat(),
        validation_data=valid_dataset,
        epochs=total_epochs,
        initial_epoch=initial_epochs,
        steps_per_epoch=steps_per_epoch,
    )

    print("\nEvaluating model after training...")
    loss, accuracy = model.evaluate(test_dataset)
    print("final loss: {:.2f}".format(loss))
    print("final accuracy: {:.2f}".format(accuracy))

    timestamp = datetime.utcnow()
    timestamp = timestamp.strftime("%Y-%m-%d-%H%M")
    model_file = output_dir / "nemo-ft--{}--{:.2f}.h5".format(timestamp, accuracy)
    model.save(str(model_file))
