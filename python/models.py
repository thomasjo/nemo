from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential

from layers import Dropout


def create_model(input_shape, num_classes):
    # Load a pre-trained base model to use for feature extraction.
    base_model = VGG16(include_top=False, weights="imagenet", input_shape=input_shape)
    base_model.trainable = False

    # Create model by stacking a prediction layer on top of the base model.
    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))

    return model, base_model


def load_model(model_file):
    custom_objects = {"Dropout": Dropout}
    model = keras.models.load_model(str(model_file), custom_objects)
    base_model = model.layers[0]

    return model, base_model
