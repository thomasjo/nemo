"""
Usage:
  finetune_model.py [options] <source> <output> <model>

Options:
  --image-size=S  Target width and height of images after pre-processing.  [default: 224]
  -h, --help      Show this screen.
"""

from docopt import docopt

from pathlib import Path

from nemo.datasets import dataset_from_dir, read_labels, AUTOTUNE, BATCH_SIZE
from nemo.images import load_and_preprocess_image
from nemo.models import evaluate_model, load_model
from nemo.utils import ensure_reproducibility


def eval_model(model_file, datasets):
    # Load a pre-trained model.
    model, base_model = load_model(model_file)

    metrics = evaluate_model(model, datasets)

    return metrics


if __name__ == "__main__":
    args = docopt(__doc__)
    source_dir = Path(args["<source>"])
    output_dir = Path(args["<output>"])
    model_file = Path(args["<model>"])
    image_size = int(args["--image-size"])

    # Use fixed seeds and deterministic ops.
    ensure_reproducibility(seed=42)

    # Prepare evaluation dataset.
    labels = read_labels(model_file.with_suffix(".yaml"))
    eval_dataset, eval_count = dataset_from_dir(source_dir, labels)
    eval_dataset = eval_dataset.map(load_and_preprocess_image(image_size), num_parallel_calls=AUTOTUNE)
    eval_dataset = eval_dataset.batch(BATCH_SIZE)
    eval_dataset = eval_dataset.prefetch(AUTOTUNE)

    datasets = (None, None, eval_dataset)
    loss, accuracy = eval_model(model_file, datasets)

    print("*" * 72)
    print(loss, accuracy)
