from mynumpy.nnutils import *
from mynumpy.nnutils import _get_conv_output_shape2d

def _get_activation_derivative_from_name(name):
    name = name or 'none'
    return {
        'tanh': (tanh, dtanh),
        'sigmoid': (sigmoid, dsigmoid),
        'atan': (atan, datan),
        'relu': (relu, drelu),
        'none': (identity, didentity)
    }.get(name)

class CNNLayer:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad, learning_rate=1):
        raise NotImplementedError


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

        output_shape = (
            num_kernels,
            *_get_conv_output_shape2d(input_shape[-2:], (kernel_size, kernel_size), stride, padding))
        super(Conv2D, self).__init__(input_shape, output_shape)

        self.stride = stride
        self.padding = padding

        self.kernel_shape = (input_shape[0], kernel_size, kernel_size)
        self.kernels = rand_range(-0.1, 0.1, num_kernels, *self.kernel_shape)

        for k in self.kernels:
            k -= np.mean(k)

        self.b = np.zeros(shape=self.output_shape)

        self.activation, self.derivative = _get_activation_derivative_from_name(activation)

    def forward(self, img):
        img = np.reshape(img, self.input_shape)
        stride, padding = self.stride, self.padding
        output = np.zeros(shape=self.output_shape)
        for o, kernel in zip(output, self.kernels):
            for i, k in zip(img, kernel):
                o[:,:] += conv2d(i, k, stride, padding)

        output += self.b

        self.cached = img, output

        return self.activation(output)

    def backward(self, grad, learning_rate=1):
        img, output = self.cached  # pad(self.cached_input, self.padding)
        grad = np.reshape(grad, self.output_shape) * self.derivative(output)

        stride, padding = self.stride, self.padding
        dX = np.zeros_like(img)
        for g, kernel in zip(grad, self.kernels):
            dx = [conv2d(g, np.rot90(np.rot90(k)), stride, padding, mode='full') for k in kernel]
            dX += np.array(dx)

        for g, kernel in zip(grad, self.kernels):
            g = np.rot90(np.rot90(g))
            for k, i in zip(kernel, img):
                k -= learning_rate * np.rot90(np.rot90(conv2d(i, g, stride, padding)))

        self.b -= learning_rate * grad

        return dX

    @property
    def num_kernels(self):
        return len(self.kernels)


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

    def forward(self, img):
        self.cached = img
        img = np.reshape(img, self.input_shape)
        return np.array([self._pooling_func(x) for x in img])

    def backward(self, grad, learning_rate=1):
        return np.array([self._unpooling_func(img, g) for g, img in zip(grad, self.cached)])

    def _pooling_func(self, x):
        return np.zeros(self.output_shape)

    def _unpooling_func(self, prev, y):
        return np.zeros(self.input_shape)


class MaxPooling2D(_Pooling2D):
    def __init__(self, input_shape, pool_shape=(3, 3), stride=1, padding=0):
        super(MaxPooling2D, self).__init__(input_shape, pool_shape, stride, padding)

    def _pooling_func(self, x):
        return max_pooling2d(x, self.pool_shape, self.stride, self.padding)

    def _unpooling_func(self, prev, y):
        return max_unpooling2d(prev, y, self.pool_shape, self.stride, self.padding)


class AveragePooling2D(_Pooling2D):
    def __init__(self, input_shape, pool_shape=(3, 3), stride=1, padding=0):
        super(AveragePooling2D, self).__init__(input_shape, pool_shape, stride, padding)

    def _pooling_func(self, x):
        return average_pooling2d(x, self.pool_shape, self.stride, self.padding)

    def _unpooling_func(self, prev, y):
        return average_unpooling2d(prev, y, self.pool_shape, self.stride, self.padding)


class Flatten(CNNLayer):
    def __init__(self, input_shape):
        output_shape = (1, prod(input_shape))
        super(Flatten, self).__init__(input_shape, output_shape)

    def forward(self, img):
        return flatten(img)

    def backward(self, grad, learning_rate=1):
        return np.reshape(grad, self.input_shape)


class Dense(CNNLayer):
    def __init__(self, input_size, output_size, activation='sigmoid'):
        super(Dense, self).__init__((1, input_size), (1, output_size))

        self.activation, self.derivative = _get_activation_derivative_from_name(activation)

        if activation == 'sigmoid':
            W_thresh = 4 * np.sqrt(6 / (input_size + output_size))
        else:
            W_thresh = np.sqrt(6 / (input_size + output_size))

        self.W = rand_range(-W_thresh, W_thresh, input_size, output_size)  # np.random.rand(input_size, output_size)
        self.b = np.zeros(self.output_shape)

    def forward(self, x):
        y = np.dot(x, self.W) + self.b
        y_act = self.activation(y)
        self.cached = x, y
        return y_act

    def backward(self, grad, learning_rate=1):
        x, y = self.cached
        grad = grad * self.derivative(y)
        dX = np.dot(grad, self.W.T)

        self.b -= learning_rate * grad
        self.W -= learning_rate * np.dot(x.T, grad)

        return dX