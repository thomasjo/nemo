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

    # Binary mask for finding the "metal border".
    image = _imread(image_file)
    image_binary = binary(image, blur_size=75, threshold=70)

    # Find the "metal border" component based on area.
    n_labels, image_cc, stats, centroids = cv.connectedComponentsWithStats(image_binary)
    # Assume that the metal border is the component with the largest area,
    # after ignoring the "background" that is always labeled as 0.
    border_label = np.argmax(stats[1:, cv.CC_STAT_AREA]) + 1

    # Binary mask used for finding objects.
    image_binary = binary(image, blur_size=31, threshold=120)

    # Remove the "metal border" from the object mask.
    image_binary[image_cc == border_label] = 0

    # Find all regions of interest.
    n_labels, image_cc, stats, centroids = cv.connectedComponentsWithStats(image_binary)

    # Remove objects with fewer than specified number of pixels.
    labels, pixel_counts = np.unique(image_cc[image_cc > 0], return_counts=True)
    image_binary[np.isin(image_cc, labels[pixel_counts < 1024])] = 0

    # Save object mask image.
    _imwrite(image_file, image_binary, suffix="binary")

    # Create an image with mask overlays. Useful for visual debugging.
    image_seg = image.copy()
    image_seg[image_binary == 255, 2] = 255
    image_seg[image_binary == 255, 1] = 0
    image_seg[image_binary == 255, 0] = 127
    alpha = 0.5
    image_overlay = cv.addWeighted(image_seg, alpha, image, 1 - alpha, 0)
    _imwrite(image_file, image_overlay, suffix="overlay")

    # break
