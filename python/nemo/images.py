import tensorflow as tf


IMAGE_SIZE = 224


def preprocess_image(image):
    image = tf.io.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, IMAGE_SIZE, IMAGE_SIZE)

    return image


def load_and_preprocess_image(path, *args):
    image = tf.io.read_file(path)
    image = preprocess_image(image)

    return (image, *args)


def augment_image(image, *args):
    # Random horizontal flipping.
    image = tf.image.random_flip_left_right(image)

    # Random rotation in increments of 90 degrees.
    rot_k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k=rot_k)

    # Random light distortion.
    image = tf.image.random_brightness(image, 0.1)
    image = tf.image.random_contrast(image, 0.9, 1.1)
    image = tf.image.random_hue(image, 0.05)
    image = tf.image.random_saturation(image, 0.9, 1.1)

    # Ensure image is still valid.
    image = tf.clip_by_value(image, 0.0, 1.0)

    return (image, *args)
