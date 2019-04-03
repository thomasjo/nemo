from datetime import datetime
from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras

from train_model import batched_dataset, image_paths, process_labels

if __name__ == "__main__":
    file_dir = Path(__file__).parent.resolve()
    root_dir = file_dir.parent

    # TODO: Make these configurable?
    data_dir = root_dir / "data"
    test_dir = data_dir / "test-new"

    label_names, label_to_index = process_labels(test_dir)
    test_batches, _ = batched_dataset(test_dir, label_to_index, shuffle=False)

    output_dir = root_dir / "output"
    model_file = output_dir / "nemo.h5"
    model = keras.models.load_model(str(model_file))
    model.summary()

    loss, accuracy = model.evaluate(test_batches)
    print("loss: {:.2f}".format(loss))
    print("accuracy: {:.2f}".format(accuracy))

    # --
    print()
    print()

    mixed_dir = data_dir / "test-mixed"
    mixed_paths = image_paths(mixed_dir)
    mixed_batches, mixed_count = batched_dataset(mixed_dir, shuffle=False)

    predictions = model.predict(mixed_batches)
    predictions[predictions >= 0] = 1
    predictions[predictions < 0] = 0
    predictions = predictions.astype(int)

    output_dir = root_dir / "output" / "predictions"

    label_dirs = {}
    for label_name in label_names:
        label_dir = output_dir / label_name
        label_dir.mkdir(parents=True, exist_ok=True)
        label_dirs[label_name] = label_dir

    for i in range(mixed_count):
        image_path = mixed_paths[i]
        image_file = Path(image_path)

        label = predictions[i, 0]
        label_name = label_names[label]

        target_file = label_dirs[label_name] / image_file.name
        target_file.write_bytes(image_file.read_bytes())
