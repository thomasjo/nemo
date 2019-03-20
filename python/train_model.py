from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import VGG16

# Used for auto-tuning dataset prefetching parameters.
AUTOTUNE = tf.data.experimental.AUTOTUNE


def preprocess_image(image):
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.resize_with_pad(image, 224, 224)
    image /= 255.0

    return image


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = preprocess_image(image)

    return image, label


def image_paths_and_labels(path, label_to_index):
    image_paths = []
    image_labels = []

    for file in path.rglob("*.png"):
        image_paths.append(str(file))
        image_labels.append(label_to_index[file.parent.name])

    return image_paths, image_labels


# TODO: Make this configurable via arguments using docopt.
data_dir = Path("/root/data")

train_dir = data_dir / "train"
test_dir = data_dir / "test"

label_names = sorted(path.name for path in train_dir.glob("*/") if path.is_dir())
print("label names:", label_names)

label_to_index = dict((name, index) for index, name in enumerate(label_names))
print(label_to_index)

train_paths, train_labels = image_paths_and_labels(train_dir, label_to_index)
train_count = len(train_paths)
print("num training images:", train_count)

test_paths, test_labels = image_paths_and_labels(test_dir, label_to_index)
test_count = len(test_paths)
print("num test images:", test_count)

raw_train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
raw_test = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

train_ds = raw_train.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
test_ds = raw_test.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = train_count

train_batches = train_ds.shuffle(SHUFFLE_BUFFER_SIZE)
train_batches = train_batches.batch(BATCH_SIZE)
train_batches = train_batches.prefetch(AUTOTUNE)
print(train_batches)

test_batches = test_ds.batch(BATCH_SIZE)
test_batches = test_batches.prefetch(AUTOTUNE)
print(test_batches)

print()
print()

base_model = VGG16(include_top=False, weights="imagenet", pooling="max")
base_model.trainable = False
base_model.summary()

for image_batch, label_batch in train_batches.take(1):
    print(image_batch.shape)

feature_batch = base_model(image_batch)
print(feature_batch.shape)
print()

prediction_layer = keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch)
print(prediction_batch.shape)
print()

base_learning_rate = 0.0005
model = keras.Sequential([base_model, prediction_layer])
model.compile(
    optimizer=keras.optimizers.RMSprop(lr=base_learning_rate),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

loss0, accuracy0 = model.evaluate(train_batches, steps=20)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

initial_epochs = 50
steps_per_epoch = round(train_count) // BATCH_SIZE
validation_split = 0.1
validation_steps = 20

history = model.fit(
    train_batches.repeat(),
    epochs=initial_epochs,
    steps_per_epoch=steps_per_epoch,
    # validation_split=validation_split,
    # validation_steps=validation_steps,
)
