import tensorflow.keras as keras

from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import Sequential

from nemo.hparams import get_optimizer
from nemo.layers import Dropout


class Classifier(keras.Model):
    def __init__(self, feature_extractor: keras.Model, num_classes: int, hparams):
        super().__init__()

        self.feature_extractor = feature_extractor

        self.classifier = Sequential([
            Flatten(),
            Dense(hparams.num_units_fc1, activation="relu"),
            Dropout(hparams.dropout),
            Dense(hparams.num_units_fc2, activation="relu"),
            Dropout(hparams.dropout),
            Dense(num_classes, activation="softmax"),
        ])

    def call(self, inputs, training=None):
        x = self.feature_extractor(inputs, training=training)
        x = self.classifier(x, training=training)

        return x


def create_model(input_shape, num_classes, hparams):
    # Load a pre-trained base model to use for feature extraction.
    feature_extractor = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    feature_extractor.trainable = False

    # Create model by stacking a prediction layer on top of the base model.
    model = Classifier(feature_extractor, num_classes, hparams)

    return model


def load_model(model_file):
    custom_objects = {"Dropout": Dropout}
    model: Classifier = keras.models.load_model(str(model_file), custom_objects)

    return model


def compile_model(model, learning_rate, hparams):
    optimizer = get_optimizer(hparams.optimizer, learning_rate)
    loss = CategoricalCrossentropy()
    metrics = [CategoricalAccuracy()]

    model.compile(optimizer, loss, metrics)

    return model


def fit_model(model, datasets, metadata, epochs, steps_per_epoch=None, initial_epoch=0):
    train_dataset, valid_dataset, _ = datasets

    callbacks = list()

    # Use early stopping to prevent overfitting, etc.
    # callbacks.append(keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True))

    history = model.fit(
        train_dataset,
        validation_data=valid_dataset,
        initial_epoch=initial_epoch,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=steps_per_epoch,
        callbacks=callbacks,
        verbose=1,
    )

    return model, history


def evaluate_model(model, datasets, steps=None):
    _, _, test_dataset = datasets
    metrics = model.evaluate(test_dataset, verbose=2, steps=steps)

    return metrics
