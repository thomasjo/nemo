"""
Usage:
  finetune_model.py [options] <source> <output> <model>

Options:
  --epochs=N          Number of fine-tuning epochs. [default: 5]
  --initial-epochs=N  Number of initial training epochs. [default: 25]
  -h, --help          Show this screen.
"""

from docopt import docopt

from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.optimizers import RMSprop

from datasets import load_datasets, save_labels
from models import load_model

# Used for auto-tuning dataset prefetch size, etc.
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = 224


def main(source_dir, output_dir, model_file, epochs, initial_epochs):
    train_dataset, valid_dataset, test_dataset, metadata = load_datasets(source_dir)

    # Load a pre-trained model.
    model, base_model = load_model(model_file)

    # Only fine-tune the last few layers of the base model.
    base_model.trainable = True
    for layer in base_model.layers[:-8]:
        layer.trainable = False

    # Prepare optimizer, loss function, and metrics.
    learning_rate = 0.00001
    optimizer = RMSprop(learning_rate)
    loss = CategoricalCrossentropy()
    metrics = [CategoricalAccuracy()]

    model.compile(optimizer, loss, metrics)

    # Initial training parameters.
    total_epochs = initial_epochs + epochs
    steps_per_epoch = metadata.train_count // BATCH_SIZE
    steps_per_epoch *= 4  # Increase steps because of image augmentations

    # Use early stopping to prevent overfitting, etc.
    early_stopping = EarlyStopping(patience=1, restore_best_weights=True)

    print("Fine-tuning model...")
    history = model.fit(
        train_dataset.repeat(),
        validation_data=valid_dataset,
        epochs=total_epochs,
        initial_epoch=initial_epochs,
        steps_per_epoch=steps_per_epoch,
        callbacks=[early_stopping],
    )
    print(history)

    print("\nEvaluating model after training...")
    loss, accuracy = model.evaluate(test_dataset)
    print("final loss: {:.2f}".format(loss))
    print("final accuracy: {:.2f}".format(accuracy))

    # Save fine-tuned model.
    timestamp = datetime.utcnow()
    timestamp = timestamp.strftime("%Y-%m-%d-%H%M")
    model_file = output_dir / "nemo-ft--{}--{:.2f}.h5".format(timestamp, accuracy)
    model.save(str(model_file))

    # Save labels used by the fine-tuned model.
    label_file = model_file.with_suffix(".yaml")
    save_labels(label_file, metadata.labels)


if __name__ == "__main__":
    args = docopt(__doc__)
    source_dir = Path(args["<source>"])
    output_dir = Path(args["<output>"])
    model_file = Path(args["<model>"])
    epochs = int(args["--epochs"])
    initial_epochs = int(args["--initial-epochs"])

    main(source_dir, output_dir, model_file, epochs, initial_epochs)
