import tensorflow as tf
import numpy as np
from functools import reduce
from operator import mul

def nchw_to_nhwc(x):
    # https://stackoverflow.com/questions/37689423/convert-between-nhwc-and-nchw-in-tensorflow
    return tf.transpose(x, [0, 2, 3, 1])

def shape_nchw_to_nhcw(shape):
    # https://stackoverflow.com/questions/37689423/convert-between-nhwc-and-nchw-in-tensorflow
    return [shape[0], shape[2], shape[3], shape[1]]

def rand_range(low, high, *args):
    return tf.random_uniform([*args]) * (high - low) + low

def prod(x):
    return reduce(mul, x, 1)

def pad(img, padding, value=0):
    padding = tf.fill((len(img.shape), 2), padding)
    return tf.pad(img, padding, constant_values=value)

def rgb_to_grayscale(img):
    r = tf.multiply(0.2126, img[0])
    g = tf.multiply(0.7152, img[1])
    b = tf.multiply(0.0722, img[2])
    return np.array([r + g + b])

def conv2d(img, kernel, stride, padding):
    img = pad(img, padding)
    return tf.nn.conv2d(nchw_to_nhwc(img), kernel, strides=stride, padding='VALID', data_format='NHWC')

def max_pooling2d(img, pool_shape4d, strides, padding):
    # img = tf.reshape(pad(img, padding, 0), [1,*img.shape])
    img = pad(img, padding, 0)
    return tf.nn.max_pool(nchw_to_nhwc(img), ksize=shape_nchw_to_nhcw(pool_shape4d), strides=strides, padding='VALID', data_format='NHWC')

def average_pooling2d(img, pool_shape4d, strides, padding):
    # img = tf.reshape(pad(img, padding, 0), [1,*img.shape])
    img = pad(img, padding, 0)
    # return tf.layers.average_pooling2d(img, pool_shape, strides=stride, padding='valid', data_format='channels_first')
    return tf.nn.avg_pool(nchw_to_nhwc(img), ksize=shape_nchw_to_nhcw(pool_shape4d), strides=strides, padding='VALID', data_format='NHWC')

def relu(x):
    return tf.maximum(x, 0)
    # return tf.multiply(tf.greater(x, 0) * 1, x)
    # tf.maximum(tf.zeros_like(x), x)

def sigmoid(x):
    return tf.sigmoid(x)

def tanh(x):
    return tf.tanh(x)

def softmax(x):
    return tf.sparse_softmax(x)

def atan(x):
    return tf.atan(x)

def identity(x):
    return x

def flatten(img):
    # return tf.layers.flatten(img)
    return tf.reshape(img, [1, -1])

def mean_squared_error(y, y_hat):
    return tf.reduce_mean((y_hat - y)**2)

def mean_absolute_error(y, y_hat):
    return tf.reduce_mean(abs(y_hat - y))

def logcosh(x):
    return tf.log(tf.cosh(x))

def logcosh_error(y, y_hat):
    return logcosh(y - y_hat)