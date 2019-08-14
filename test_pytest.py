import pytest
import skimage
import tensorflow as tf
from skimage import data, io
from skimage.util import img_as_ubyte
import tf_morphology as tfm
import numpy as np

tf.enable_eager_execution()

# TODO: test against scikit-image functionality


def setup_binary_test():
    horse = data.horse()
    tf_horse = horse == 1
    tf_horse = tf.cast(tf_horse, tf.int32)
    tf_horse = tf.reshape(tf_horse, (1, 328, 400, 1))
    selem = skimage.morphology.disk(6)
    tf_selem = tf.cast(selem, tf.int32)
    tf_selem = tf.reshape(tf_selem, (13, 13, 1))
    return (horse, tf_horse), (selem, tf_selem)


def setup_grayscale_test():
    # TODO:
    # return (coins, tf_coins), (selem, tf_selem)
    pass


def test_grayscale_operations(tf_func, sk_func):
    # TODO
    pass


equivalent_binary_functions = [
    (tfm.binary_closing2d, skimage.morphology.binary_closing),
    (tfm.binary_dilation2d, skimage.morphology.binary_dilation),
    (tfm.binary_erosion2d, skimage.morphology.binary_erosion),
    (tfm.binary_opening2d, skimage.morphology.binary_opening),
]


@pytest.mark.parametrize("tf_func, sk_func", equivalent_binary_functions)
def test_binary_operations(tf_func, sk_func):
    (mask, tf_mask), (selem, tf_selem) = setup_binary_test()
    tf_processed = tf_func(tf_mask, tf_selem)
    processed = sk_func(mask, selem)
    assert np.allclose(processed.astype(np.int32), np.squeeze(tf_processed))
