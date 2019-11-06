"""
Usage:
  create_report.py [options] <pickle> <output>

Options:
  -h, --help  Show this screen.
"""

from docopt import docopt

import pickle
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


DEFAULT_LABEL_LOOKUP = {"agglutinated": 0, "benthic": 1, "planktic": 2, "sediment": 3}


def main(pickle_file, output_dir):
    # Load the pickled analysis data.
    with open(pickle_file, "rb") as f:
        data: dict = pickle.load(f)

    # Grab all raw data from the analysis blob.
    label_lookup = data.get("labels", DEFAULT_LABEL_LOOKUP)
    files = data.get("files")
    predictions = data.get("predictions")
    accuracies = data.get("accuracies")

    # Find true labels for all analyzed files.
    files = [Path(file) for file in files]
    labels = [label_lookup[file.parent.name] for file in files]

    # Calculate Monte Carlo estimates.
    mc_accuracy = accuracies.mean()

    ensemble_predictions = predictions.mean(axis=0).argmax(axis=1)
    ensemble_accuracy = np.equal(labels, ensemble_predictions).mean()

    low_predictions = predictions.mean(axis=0).max(axis=1).argsort()[:10]
    low_accuracies = accuracies.mean(axis=0).argsort()[:10]

    print(f"Monte Carlo accuracy:          {mc_accuracy:.2%}")
    print(f"Monte Carlo ensemble accuracy: {ensemble_accuracy:.2%}")
    print(f"Low prediction candidates:     {low_predictions}")
    print(f"Low accuracy candidates:       {low_accuracies}")

    # Prepare output directory for analyses.
    timestamp = datetime.utcnow()
    timestamp = timestamp.strftime("%Y-%m-%d-%H%M")
    # result_dir = output_dir / "analyses" / pickle_file.stem / timestamp
    result_dir = output_dir / "analyses" / pickle_file.stem
    result_dir.mkdir(parents=True, exist_ok=True)

    prediction_file = result_dir / "prediction.pdf"
    accuracy_file = result_dir / "accuracy.pdf"

    # Default all figures to A4 size and landscape orientation.
    # matplotlib.rc("figure", figsize=(11.69, 8.27))
    matplotlib.rc("figure", figsize=(8.27, 11.69))
    matplotlib.rc("savefig", dpi=300)

    generate_report(
        prediction_file,
        predictions[:, low_predictions],
        accuracies[:, low_predictions],
        np.array(files)[low_predictions],
        np.array(labels)[low_predictions],
        label_lookup,
    )

    generate_report(
        accuracy_file,
        predictions[:, low_accuracies],
        accuracies[:, low_accuracies],
        np.array(files)[low_accuracies],
        np.array(labels)[low_accuracies],
        label_lookup,
    )


def generate_report(report_file, predictions, accuracies, files, labels, label_lookup):
    label_names = [key.capitalize() for key in label_lookup.keys()]

    with PdfPages(str(report_file)) as pdf:
        for idx in range(predictions.shape[1]):
            fig = plt.figure()
            fig.suptitle(str(files[idx]), c="lightgray")

            # Plot candidate image.
            ax = plt.subplot(3, 2, 1)
            image = plt.imread(str(files[idx]))
            ax.imshow(image)
            ax.axis("off")

            # ...
            ax = plt.subplot(3, 2, 2)
            ax.axis("off")
            ax.text(0, 0.5, f"Mean accuracy: {accuracies[:, idx].mean():4.2%}")

            if predictions.ndim == 3:
                # Plot class prediction histograms.
                for i in range(predictions.shape[2]):
                    ax = plt.subplot(3, 2, i + 3)
                    ax.set_title(label_names[i])
                    ax.hist(
                        predictions[:, idx, i],
                        align="left",
                        bins=20,
                        range=(0, 1),
                        fc="tab:blue",
                        ec="tab:blue",
                    )
                    ax.axvline(np.round(predictions[:, idx, i].mean(), 1), c="tab:red")

            fig.subplots_adjust(hspace=0.3)
            # fig.tight_layout()
            pdf.savefig()
            plt.close()


if __name__ == "__main__":
    args = docopt(__doc__)
    pickle_file = Path(args["<pickle>"])
    output_dir = Path(args["<output>"])

    main(pickle_file, output_dir)
