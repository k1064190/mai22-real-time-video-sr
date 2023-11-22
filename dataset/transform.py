"""Define transform methods.

classes:
    NormalizePair: Normalize the given data pair with mean and standard deviation.
    RandomHorizontalFlipPair: Horizontally flip the given data pair randomly.
    RandomVerticalFlipPair: Vertically flip the given data pair randomly.
    RandomRot90Pair: Rotate 90 degrees on the given data pair randomly.
    RandomCropSRPair: Crop a patch from given SR data pair randomly.
"""
import collections

import tensorflow as tf
from tensorflow import keras

# namedtuple for convenience.
TFDataPair = collections.namedtuple('TFDataPair', ['image', 'label'])


class NormalizePair():
    """Normalize the given data pair with mean and standard deviation.

    Attributes:
        input_mean, input_std, target_mean, target_std:
            A `float`, `list`, `tuple`, `1D-array` represents mean or standard deviation for
                input or target.
            If it's not a single number, it should match the number of channel.
            For instance, for the RGB input_img, the value could be a single number, or 3 numbers.
    """

    def __init__(self, image_mean=0., image_std=1., label_mean=0., label_std=1.):
        """Initialize attributes.

        Args:
            input_mean: Please refer to Attributes. Defaults to 0.0.
            input_std: Please refer to Attributes. Defaults to 1.0.
            target_mean: Please refer to Attributes. Defaults to 0.0.
            target_std: Please refer to Attributes. Defaults to 1.0.
        """
        self.image_mean = image_mean
        self.image_std = image_std
        self.label_mean = label_mean
        self.label_std = label_std

    def __call__(self, data_pair):
        """Use `tf.Tensor` basic operators to run normalization on `tf.Tensor`."""
        image = (tf.cast(data_pair.image, tf.float32) - self.image_mean) / self.image_std
        label = (tf.cast(data_pair.label, tf.float32) - self.label_mean) / self.label_std
        return TFDataPair(image, label)


class RandomHorizontalFlipPair():
    """Horizontally flip the given data pair randomly.

    Should apply flipping on both image and label simultaneously.
    """

    def __call__(self, data_pair):
        """Use `tf.image.flip_left_right` to run hflip on `tf.Tensor`."""

        def _flip_left_right():
            """Perform hflip."""
            image = tf.image.flip_left_right(data_pair.image)
            label = tf.image.flip_left_right(data_pair.label)
            return TFDataPair(image, label)

        return tf.compat.v1.cond(tf.random.uniform([]) > 0.5, _flip_left_right, lambda: data_pair)


class RandomVerticalFlipPair():
    """Vertically flip the given data pair randomly.

    Should apply flipping on both image and label simultaneously.
    """

    def __call__(self, data_pair):
        """Use `tf.image.flip_up_down` to run vflip on `tf.Tensor`."""

        def _flip_up_down():
            """Perform vflip."""
            image = tf.image.flip_up_down(data_pair.image)
            label = tf.image.flip_up_down(data_pair.label)
            return TFDataPair(image, label)

        return tf.compat.v1.cond(tf.random.uniform([]) > 0.5, _flip_up_down, lambda: data_pair)

class RandomPermute:
    """Randomly permute the given data pair.

    Should apply permutation on both image and label simultaneously.
    """

    def __call__(self, data_pair):
        """Use `tf.random.shuffle` to run permutation on `tf.Tensor`."""

        def _permute():
            """Perform permutation."""
            """
                numpy version
                    perm = np.random.permutation(3)
                    im1 = im1[:, perm]
                    im2 = im2[:, perm]
            """
            perm = tf.random.shuffle([0, 1, 2])
            image = tf.gather(data_pair.image, perm, axis=-1)
            label = tf.gather(data_pair.label, perm, axis=-1)
            return TFDataPair(image, label)

        return tf.compat.v1.cond(tf.random.uniform([]) > 0.5, _permute, lambda: data_pair)

