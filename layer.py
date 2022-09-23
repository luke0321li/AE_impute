import tensorflow as tf
import numpy as np


def conv2d_biased_layer(input, filter, bias, padding):
    output = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding=padding)
    return tf.nn.relu(output + bias)
    # Note: output has shape [?, x, x, y] while bias has shape [y]. The addition works because tensorflow has inbuilt
    # broadcasting for addition operations


def deconv2d_biased_layer(input, filter, output_shape, bias, padding):
    output = tf.nn.conv2d_transpose(input, filter, output_shape, strides=[1, 1, 1, 1], padding=padding)
    return tf.nn.relu(output + bias)


def pool_layer(input, window_size, padding):
    window_shape = [1, window_size, window_size, 1]
    return tf.nn.max_pool(input, ksize=window_shape, strides=window_shape, padding=padding)


def make_w_and_b(shape, w_stdev=0.05, b_stdev=0.1):
    return tf.Variable(tf.random_normal(shape, stddev=w_stdev)), tf.Variable(
        tf.random_normal([shape[-1]], stddev=b_stdev))


def make_w_and_b_deconv(shape, w_stdev=0.05, b_stdev=0.1):
    return tf.Variable(tf.random_normal(shape, stddev=w_stdev)), tf.Variable(
        tf.random_normal([shape[-2]], stddev=b_stdev))
