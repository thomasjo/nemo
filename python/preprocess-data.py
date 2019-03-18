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
    # image = cv.GaussianBlur(image, (15, 15), 15)

    return image


def binary(image, blur_size, threshold=127):
    if image.ndim == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    image = blur(image, blur_size)
    # _, image = cv.threshold(image, 130, 255, cv.THRESH_BINARY)
    # _, image = cv.threshold(image, 55, 255, cv.THRESH_BINARY_INV)
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

    # Segment/binarize using scikit-image...
    # image = imageio.imread(image_file)
    # labels = segmentation.slic(image)
    # output_image = cmap.to_rgba(labels, bytes=True)
    # imageio.imwrite(output_file, output_image)

    # Segment/binarize using OpenCV...
    image = _imread(image_file)
    # print(image.shape)

    # image_blur = blur(image)
    # _imwrite(image_file, image_blur, suffix="blur")

    # image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # image_hue = image_hsv[..., 0]
    # _imwrite(image_file, image_hue, suffix="hue")

    # image_sat = image_hsv[..., 1]
    # _imwrite(image_file, image_sat, suffix="sat")

    # image_val = image_hsv[..., 2]
    # _imwrite(image_file, image_val, suffix="val")

    # image_hue_norm = np.zeros_like(image_hue)
    # image_hue_norm = cv.normalize(image_hue, image_hue_norm, 0, 255, cv.NORM_MINMAX)
    # _imwrite(image_file, image_hue_norm, suffix="hue_norm")

    # image_hue_norm_blur = cv.medianBlur(image_hue_norm, 31)
    # _imwrite(image_file, image_hue_norm_blur, suffix="hue_norm_blur")

    image_binary = binary(image, blur_size=75, threshold=70)
    # _imwrite(image_file, image_binary, suffix="binary")
    # print(np.unique(image_binary))
    # print(image_binary.dtype)

    labels = np.zeros_like(image_binary)
    _, labels = cv.connectedComponents(image_binary, labels)

    # output_image = cmap.to_rgba(labels, bytes=True)
    # imageio.imwrite(_output_path(image_file, suffix="cc"), output_image)

    unique, counts = np.unique(labels[labels > 0], return_counts=True)
    idx_max = np.argmax(counts)
    label_max = unique[idx_max]
    mask = (labels == label_max).astype(np.uint8)
    mask[mask == 1] = 255
    # print(np.unique(mask))
    # print(mask.dtype)

    # imageio.imwrite(_output_path(image_file, suffix="mask"), mask)

    # image_binary = binary(image, blur_size=31, threshold=130) - mask
    image_binary = binary(image, blur_size=31, threshold=130)
    image_binary[labels == label_max] = 0
    _imwrite(image_file, image_binary, suffix="binary")

    # print(np.unique(image_binary))
    # print(image_binary.dtype)

    labels = np.zeros_like(image_binary)
    _, labels = cv.connectedComponents(image_binary, labels)

    unique, counts = np.unique(labels[labels > 0], return_counts=True)
    # print(unique)
    # print(counts)

    debug = image_binary.copy()
    # label_max = unique[np.argsort(counts)[0]]
    # print(label_max)
    # debug[labels != label_max] = 0
    # print(np.sum(debug) / 255)

    # print(unique[counts < 1024])

    debug[np.isin(labels, unique[counts < 1024])] = 0
    _imwrite(image_file, debug, suffix="binary")

    # image_seg = np.zeros_like(image)
    image_seg = image.copy()
    # image_seg = np.dstack((image_seg, np.zeros_like(image_binary)))
    print(image_seg.shape)
    # image_seg[debug == 255, 3] = 255
    image_seg[debug == 255, 2] = 255
    image_seg[debug == 255, 1] = 0
    image_seg[debug == 255, 0] = 127

    # _imwrite(image_file, image_seg, suffix="seg")

    # image = cv.cvtColor(image, cv.COLOR_BGR2BGRA)
    # image[..., 3] = 255

    alpha = 0.5
    image_overlay = cv.addWeighted(image_seg, alpha, image, 1 - alpha, 0)
    _imwrite(image_file, image_overlay, suffix="overlay")

    # image_adaptive_gaussian = adaptive_gaussian(image)
    # _imwrite(image_file, image_adaptive_gaussian, suffix="adaptive_gaussian")

    # break
