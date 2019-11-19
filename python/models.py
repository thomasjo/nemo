from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Sequential

from hparams import get_optimizer
from layers import Dropout


BATCH_SIZE = 32


def create_model(input_shape, num_classes, hparams):
    # Load a pre-trained base model to use for feature extraction.
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False

    # Create model by stacking a prediction layer on top of the base model.
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())

    model.add(Dense(hparams.num_units_fc1, activation="relu"))
    model.add(Dropout(hparams.dropout))

    model.add(Dense(hparams.num_units_fc2, activation="relu"))
    model.add(Dropout(hparams.dropout))

    model.add(Dense(num_classes, activation="softmax"))

    return model, base_model


def load_model(model_file):
    custom_objects = {"Dropout": Dropout}
    model = keras.models.load_model(str(model_file), custom_objects)
    base_model = model.layers[0]

    return model, base_model


def compile_model(model, learning_rate, hparams):
    optimizer = get_optimizer(hparams.optimizer, learning_rate)
    loss = CategoricalCrossentropy()
    metrics = [CategoricalAccuracy()]

    model.compile(optimizer, loss, metrics)

    return model


def fit_model(model, datasets, metadata, epochs, steps=0, initial_epoch=0):
    train_dataset, valid_dataset, _ = datasets

    if steps == 0:
        steps = metadata.train_count // BATCH_SIZE
        steps *= 4  # Increase steps because of image augmentations

    # Use early stopping to prevent overfitting, etc.
    early_stopping = EarlyStopping(patience=2, restore_best_weights=True)

    history = model.fit(
        train_dataset.repeat(),
        validation_data=valid_dataset,
        initial_epoch=initial_epoch,
        epochs=epochs,
        steps_per_epoch=steps,
        callbacks=[early_stopping],
        verbose=2,
    )

    return model, history


def evaluate_model(model, datasets):
    _, _, test_dataset = datasets
    metrics = model.evaluate(test_dataset, verbose=2)

    return metrics
