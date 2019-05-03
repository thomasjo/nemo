"""
Usage:
  analyze_model.py <weights>

Options:
  -h --help  Show this screen.
"""

from pathlib import Path

import innvestigate as inn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from docopt import docopt
from keras.applications.vgg16 import VGG16

from datasets import dataset_from_dir, labels_for_dir, load_and_preprocess_image

# Used for auto-tuning dataset prefetch size, etc.
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = 224


if __name__ == "__main__":
    args = docopt(__doc__)
    weights_file = Path(args["<weights>"])

    file_dir: Path = Path(__file__).parent.resolve()
    root_dir = file_dir.parent

    # TODO: Make these configurable?
    train_dir = root_dir / "data" / "train"
    output_dir = root_dir / "output"

    # Fetch label names, and a map from names to indices.
    labels = labels_for_dir(train_dir)

    # Prepare training dataset.
    train_dataset, train_count = dataset_from_dir(train_dir, labels)
    train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

    # Load a pre-trained model.
    # model_file = output_dir / "nemo.h5"
    # model = keras.models.load_model(str(model_file))
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
    model = VGG16(input_shape=input_shape, include_top=False, weights=str(weights_file))
    # model.load_weights(str(weights_file))
    # model.summary()

    ds_iter = train_dataset.take(1).make_one_shot_iterator()
    xy = ds_iter.get_next()
    with tf.Session() as sess:
        image, label = sess.run(xy)

    analyzer = inn.create_analyzer("deep_taylor", model)
    # analyzer = inn.create_analyzer("guided_backprop", model)
    # analyzer = inn.create_analyzer("lrp.z", model)

    analysis = analyzer.analyze(image[None])
    analysis = analysis.sum(axis=3)
    analysis = analysis / np.max(np.abs(analysis))

    plt.figure()
    plt.imshow(analysis[0], cmap="seismic", clim=(-1, 1))
    plt.savefig(str(output_dir / "deep_taylor.png"))
