from pathlib import Path

import tensorflow as tf
import tensorflow.keras as keras

import matplotlib.pyplot as plt


# Used for auto-tuning dataset prefetching parameters.
AUTOTUNE = tf.data.experimental.AUTOTUNE


def image_paths_and_labels(root_dir):
    paths = []
    labels = []

    for file_path in root_dir.rglob("*.png"):
        paths.append(str(file_path))
        labels.append(file_path.parent.name)

    return paths, labels


# TODO: Make this configurable via arguments using docopt.
data_dir = Path("/root/data")

train_dir = data_dir / "train"
train_paths, train_labels = image_paths_and_labels(train_dir)
train_count = len(train_paths)
print("num training images:", train_count)

all_labels = set(train_labels)
print("all labels:", all_labels)


def preprocess_image(image):
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize_with_pad(image, 224, 224)
    image /= 255.0

    return image


def label_for_image(path):
    return Path(path).parent.name


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = preprocess_image(image)

    return image, label


path_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
image_label_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

BATCH_SIZE = 32

ds = image_label_ds.shuffle(buffer_size=train_count)
ds = ds.repeat()
ds = ds.batch(BATCH_SIZE)
ds = ds.prefetch(buffer_size=AUTOTUNE)
print(ds)

base_model = tf.keras.applications.VGG16(weights="imagenet", include_top=False)
base_model.trainable = False
print(base_model.summary())

for image_batch, label_batch in ds.take(1):
    pass
print(image_batch.shape)

feature_batch = base_model(image_batch)
print(feature_batch.shape)

global_average_layer = keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = keras.Sequential([
    base_model,
    global_average_layer,
    prediction_layer,
])

base_learning_rate = 0.0001
model.compile(
    optimizer=keras.optimizers.RMSprop(lr=base_learning_rate),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
print(model.summary())
