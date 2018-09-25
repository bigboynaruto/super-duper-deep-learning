from tqdm import tqdm
from mynumpy.layers import CNNLayer
from mynumpy.nnutils import *

class CNN:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: CNNLayer):
        self.layers.append(layer)

    def _forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def _backward(self, grad, learning_rate=1):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)
        return grad

    def train(self, x_train, y_train, epochs=1, batch_size=1, learning_rate=1., loss_function='mean_squared_error', verbosity=2):
        assert len(x_train) == len(y_train)
        assert x_train[0].shape == self.input_shape
        assert y_train[0].shape == self.output_shape
        assert verbosity in [1,2,3]

        if batch_size > 0:
            warnings.warn('Not sure if batch_size is working properly')

        loss_function = {
            'mean_squared_error' : mean_squared_error,
            'mean_absolute_error' : mean_absolute_error,
            'logcosh_error' : logcosh_error
        }.get(loss_function)

        if batch_size > 0:
            batches = np.ceil(len(x_train) / batch_size)
            x_train = np.array_split(x_train, batches)
            y_train = np.array_split(y_train, batches)

        total = len(x_train)

        for epoch in range(epochs):
            pbar = zip(x_train, y_train)
            if verbosity == 2:
                pbar = tqdm(pbar, total=total, desc=('Epoch %3d' % (epoch + 1)))

            error = 0
            for x_batch, y_batch in pbar:
                grad = np.zeros(self.output_shape)
                for x, y in zip(x_batch, y_batch):
                    output = self._forward(x)
                    grad += output - y
                    error += loss_function(y, output)

                self._backward(grad / batch_size, learning_rate)

            if verbosity in [1,2]:
                print('Loss: %f' % (error / total))

    def predict(self, x_test):
        return [self._forward(x) for x in x_test]

    @property
    def input_shape(self):
        return None if len(self.layers) == 0 else self.layers[0].input_shape

    @property
    def output_shape(self):
        return None if len(self.layers) == 0 else self.layers[-1].output_shape


class OneHotEncoder:
    def __init__(self, abc):
        self.abc = np.unique(abc)

    def encode(self, a):
        encoded = np.zeros(shape=(len(self.abc)))
        idxs = np.where(self.abc == a)[0]
        if len(idxs) == 0:
            raise Exception('No %s in alphabet!' % str(a))
        encoded[idxs[0]] = 1
        return encoded

    def decode(self, x, default_value=None):
        if default_value is None:
            assert len(x) == len(self.abc)

            idxs = np.where(np.array(x) != 0)[0]
            if len(idxs) != 1:
                raise Exception('Cannot decode %s!' % str(x))
            return self.abc[idxs[0]]
        else:
            if len(x) != len(self.abc):
                return default_value

            idxs = np.where(np.array(x) != 0)[0]
            if len(idxs) != 1:
                return default_value
            return self.abc[idxs[0]]

    def to_dict(self):
        res = {k:self.encode(k) for k in self.abc}
        return res

    @property
    def size(self):
        return len(self.abc)
