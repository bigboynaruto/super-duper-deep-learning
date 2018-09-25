from .nnutils import *
import tensorflow as tf

def _get_conv_output_shape2d(input_shape, kernel_shape, stride, padding):
    return ((input_shape[-2] + 2 * padding - kernel_shape[-2]) // stride + 1,
            (input_shape[-1] + 2 * padding - kernel_shape[-1]) // stride + 1)

def _get_activation_function_from_name(name):
    name = name or 'none'
    return {
        'tanh': tanh,
        'sigmoid': sigmoid,
        'atan': atan,
        'relu': relu,
        'none': identity
    }.get(name)

class CNNLayer:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        raise NotImplementedError

    def forward_batch(self, batch):
        return tf.reshape(tf.map_fn(self.forward, batch), [-1, *self.output_shape])

    def _get_variables(self):
        return []


class Conv2D(CNNLayer):
    def __init__(self, input_shape, num_kernels=1, kernel_size=3, stride=1, padding=0, activation=None):
        assert len(input_shape) == 3
        assert stride > 0
        assert padding >= 0
        assert kernel_size <= input_shape[-1] + padding * 2
        assert kernel_size <= input_shape[-2] + padding * 2

        if stride != 1:
            raise ValueError('Currently only stride=1 is supported')

        if padding != 0:
            raise ValueError('Currently only padding=0 is supported')

        output_shape = [
            num_kernels,
            *_get_conv_output_shape2d(input_shape[-2:], (kernel_size, kernel_size), stride, padding)]

        super(Conv2D, self).__init__(input_shape, output_shape)

        self.stride = [stride, stride, stride, stride]
        self.padding = padding

        self.kernel_shape = (input_shape[0], kernel_size, kernel_size)
        self.kernels = tf.Variable(rand_range(-0.1, 0.1, kernel_size, kernel_size, input_shape[0], num_kernels), dtype=tf.float32)

        self.b = tf.Variable(tf.zeros(shape=self.output_shape))

        self.activation = _get_activation_function_from_name(activation)

    def forward(self, img):
        return self.forward_batch(tf.reshape(img, [1, *self.input_shape]))

    def forward_batch(self, batch):
        res = conv2d(batch, self.kernels, self.stride, self.padding)
        res = tf.reshape(res, [-1, *self.output_shape])
        res = self.activation(tf.map_fn(lambda x: tf.add(x, self.b), res))
        return res

    @property
    def num_kernels(self):
        return len(self.kernels.shape[0].value)

    def _get_variables(self):
        return [self.kernels, self.b]


class _Pooling2D(CNNLayer):
    def __init__(self, input_shape, pool_shape=(3, 3), stride=1, padding=0):
        assert len(input_shape) == 3
        assert len(pool_shape) == 2

        output_shape = (
            input_shape[0],
            *_get_conv_output_shape2d(input_shape, pool_shape, stride, padding))

        super(_Pooling2D, self).__init__(input_shape, output_shape)

        self.pool_shape = pool_shape
        self.stride = stride
        self.padding = padding

        self._pool_shape4d = [1, 1, *pool_shape]
        self._strides4d = [stride, stride, stride, stride]

    def forward(self, img):
        return self.forward_batch(tf.reshape(img, [1, *self.input_shape]))

    def forward_batch(self, batch):
        res = self._pooling_func(batch)
        return tf.reshape(res, shape=[-1, *self.output_shape])

    def _pooling_func(self, x):
        raise NotImplementedError


class MaxPooling2D(_Pooling2D):
    def __init__(self, input_shape, pool_shape=(3, 3), stride=1, padding=0):
        super(MaxPooling2D, self).__init__(input_shape, pool_shape, stride, padding)

    def _pooling_func(self, x):
        return max_pooling2d(x, self._pool_shape4d, self._strides4d, self.padding)


class AveragePooling2D(_Pooling2D):
    def __init__(self, input_shape, pool_shape=(3, 3), stride=1, padding=0):
        super(AveragePooling2D, self).__init__(input_shape, pool_shape, stride, padding)

    def _pooling_func(self, x):
        return average_pooling2d(x, self._pool_shape4d, self._strides4d, self.padding)


class Flatten(CNNLayer):
    def __init__(self, input_shape):
        output_shape = (1, prod(input_shape))
        super(Flatten, self).__init__(input_shape, output_shape)

    def forward(self, img):
        return flatten(img)

    def forward_batch(self, batch):
        return tf.reshape(batch, [-1, *self.output_shape])


class Dense(CNNLayer):
    def __init__(self, input_size, output_size, activation='sigmoid'):
        super(Dense, self).__init__((1, input_size), (1, output_size))

        self.activation = _get_activation_function_from_name(activation)

        if activation == 'sigmoid':
            W_thresh = 4 * tf.sqrt(tf.divide(6., input_size + output_size))
        else:
            W_thresh = tf.sqrt(tf.divide(6., input_size + output_size))

        self.W = tf.Variable(rand_range(-W_thresh, W_thresh, input_size, output_size))
        self.b = tf.Variable(tf.zeros(self.output_shape, dtype=tf.float32))

    def forward(self, x):
        y = tf.add(tf.matmul(x, self.W), self.b)
        y = self.activation(y)
        return y

    def _get_variables(self):
        return [self.W, self.b]