from pathlib import Path

import imageio
import matplotlib.cm
import numpy as np

import cv2 as cv
import skimage.segmentation as segmentation

data_dir = Path("/root/data")
raw_data_dir = data_dir / "raw"
processed_data_dir = data_dir / "processed"

print("Data directory:", data_dir)
print("-" * 72)

cmap = matplotlib.cm.ScalarMappable(cmap=matplotlib.cm.get_cmap("viridis"))


def _output_path(orig_path, suffix=None):
    file_name = orig_path.stem
    if suffix:
        file_name = "{}-{}".format(file_name, suffix)

    return processed_data_dir / "{}.png".format(file_name)


def _imread(path):
    return cv.imread(str(path), cv.IMREAD_COLOR)


def _imwrite(path, image, suffix=None):
    return cv.imwrite(str(_output_path(path, suffix)), image)


def blur(image, size):
    image = cv.medianBlur(image, size)

    return image


def binary(image, blur_size, threshold=127):
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image = blur(image, blur_size)
    _, image = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)

    return image


def adaptive_gaussian(image):
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image = blur(image)
    image = cv.adaptiveThreshold(
        image,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        11,
        2
    )

    return image


for image_file in sorted(raw_data_dir.rglob("*.tiff")):
    print(image_file)

    output_file = processed_data_dir / "{}.png".format(image_file.stem)

    # Segment/binarize using OpenCV...
    image = _imread(image_file)
    image_binary = binary(image, blur_size=75, threshold=70)

    # Find and remove the metal "border".
    labels = np.zeros_like(image_binary)
    _, labels = cv.connectedComponents(image_binary, labels)
    unique, counts = np.unique(labels[labels > 0], return_counts=True)
    idx_max = np.argmax(counts)
    label_max = unique[idx_max]
    image_binary = binary(image, blur_size=31, threshold=130)
    image_binary[labels == label_max] = 0

    # Find all regions of interest.
    labels = np.zeros_like(image_binary)
    _, labels = cv.connectedComponents(image_binary, labels)
    unique, counts = np.unique(labels[labels > 0], return_counts=True)
    image_binary[np.isin(labels, unique[counts < 1024])] = 0
    _imwrite(image_file, image_binary, suffix="binary")

    # Create an image with ROIs overlayed. Useful for visual debugging.
    image_seg = image.copy()
    print(image_seg.shape)
    image_seg[image_binary == 255, 2] = 255
    image_seg[image_binary == 255, 1] = 0
    image_seg[image_binary == 255, 0] = 127

    alpha = 0.5
    image_overlay = cv.addWeighted(image_seg, alpha, image, 1 - alpha, 0)
    _imwrite(image_file, image_overlay, suffix="overlay")

    # break
