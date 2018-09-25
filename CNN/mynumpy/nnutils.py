import numpy as np
import scipy.signal
from scipy.special import expit
from functools import reduce
import operator

def rand_range(low, high, *args):
    return np.random.rand(*args) * (high - low) + low

def prod(x):
    return reduce(operator.mul, x, 1)

def _get_conv_output_shape2d(input_shape, kernel_shape, stride, padding):
    return ((input_shape[-2] + 2 * padding - kernel_shape[-2]) // stride + 1,
            (input_shape[-1] + 2 * padding - kernel_shape[-1]) // stride + 1)

def pad(img, padding, value=0):
    return np.pad(img, padding, 'constant', constant_values=value)

def as_strided(img, stride):
    res = np.zeros(_get_conv_output_shape2d(img.shape, (1,1), stride, 0))
    res[::stride, ::stride] = img
    return res

def rgb_to_grayscale(img):
    r = np.multiply(0.2126, img[0])
    g = np.multiply(0.7152, img[1])
    b = np.multiply(0.0722, img[2])
    return np.array([r + g + b])

def conv2d(img, kernel, stride=1, padding=0, mode='valid'):
    '''
    res = np.zeros(shape=_get_conv_output_shape2d(img.shape, kernel.shape, stride, padding))
    img = pad(img, padding)
    kernel_height,kernel_width = kernel.shape
    for x,y in np.ndindex(res.shape):
        x_img = stride * x
        y_img = stride * y
        res[x,y] = np.sum(img[x_img:x_img+kernel_height, y_img:y_img+kernel_width] * kernel)
    return res
    '''
    # scipy is faster
    img = pad(img, padding)
    return scipy.signal.convolve2d(img, kernel, mode=mode)[::stride, ::stride]

def pooling2d(img, pool_shape, stride, padding, func):
    res = np.zeros(shape=_get_conv_output_shape2d(img.shape, pool_shape, stride, padding))
    height, width = pool_shape
    for (x, y), e in np.ndenumerate(res):
        x_img = stride * x
        y_img = stride * y
        res[x, y] = func(img[x_img:x_img + height, y_img:y_img + width])
    return res

def max_pooling2d(img, pool_shape, stride, padding):
    return pooling2d(img, pool_shape, stride, padding, np.max)

def average_pooling2d(img, pool_shape, stride, padding):
    return pooling2d(img, pool_shape, stride, padding, np.average)

def max_unpooling2d(img, grad, pool_shape, stride, padding):
    img = pad(img, padding)
    res = np.zeros_like(img)
    height, width = pool_shape
    for (x, y), e in np.ndenumerate(grad):
        x_img = stride * x
        y_img = stride * y
        w_res = res[x_img:x_img + height, y_img:y_img + width]
        w_img = img[x_img:x_img + height, y_img:y_img + width]
        w_res[w_img == np.max(w_img)] += e
    return res if padding == 0 else res[padding:-padding, padding:-padding]

def average_unpooling2d(img, grad, pool_shape, stride, padding):
    img = pad(img, padding)
    res = np.zeros_like(img)
    height, width = pool_shape
    for (x, y), e in np.ndenumerate(grad):
        x_img = stride * x
        y_img = stride * y
        res[x_img:x_img + height, y_img:y_img + width] += e / (height * width)
    return res if padding == 0 else res[padding:-padding, padding:-padding]

def relu(x):
    return np.multiply(np.greater(x, 0) * 1, x)  # np.maximum(np.zeros_like(x), x)

def drelu(s):
    return np.greater(s, 0) * 1

def sigmoid(x):
    # x = np.clip(x, -500, 500)
    # return 1. / (1 + np.exp(-x))
    return expit(x)

def dsigmoid(s):
    s = sigmoid(s)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def dtanh(s):
    return 1 - tanh(s) ** 2

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def atan(x):
    return np.arctan(x)

def datan(s):
    return 1 / (1 + np.multiply(s, s))

def identity(x):
    return x

def didentity(s):
    return np.ones_like(s)

def flatten(img):
    return np.reshape(img, (1, -1))

def mean_squared_error(y, y_hat):
    return np.mean(np.subtract(y_hat, y)**2)

def mean_absolute_error(y, y_hat):
    return np.mean(np.abs(np.subtract(y_hat, y)))

def logcosh(x):
    return np.log(np.cosh(x))

def logcosh_error(y, y_hat):
    return logcosh(y - y_hat)