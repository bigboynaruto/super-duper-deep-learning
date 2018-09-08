from layers import RNNLayer
from nnutils import quadratic_loss 
from functools import reduce

class RNN:
    def __init__(self, loss_function=quadratic_loss):
        self.layers = []
        self._loss_function = loss_function or quadratic_loss

    def add_layer(self, layer:RNNLayer):
        self.layers.append(layer)

    def train(self, X_train, Y_train, epochs=1, batch_size=-1, learning_rate=1):
        if batch_size > 0:
            batches = np.ceil(len(X_train) / batch_size)
            X_train = np.array_split(X_train, batches)
            Y_train = np.array_split(Y_train, batches)

        for epoch in range(epochs):
            total_loss = 0
            for X,Y in zip(X_train, Y_train):
                outputs,loss = self.test(X, Y)
                total_loss += loss

                for output,y in reversed(list(zip(outputs, Y))):
                    error = output - y
                    reduce(lambda error,layer: layer.backward(error), reversed(self.layers), error)

                for layer in self.layers:
                    layer.adjust_weights(learning_rate)

            print('EPOCH %4d: loss=%.5f' % (epoch + 1, total_loss))

    def test(self, X_test, Y_test):
        outputs = []
        for x,y in zip(X_test, Y_test):
            output = x
            for layer in self.layers:
                output = layer.forward(output)
            outputs.append(output)

        loss = self.loss(Y_test, outputs)

        return outputs, loss

    def predict(self, X):
        outputs = []
        for x in X:
            output = x
            for layer in self.layers:
                output = layer.forward(output)
            outputs.append(output)

        return outputs

    def loss(self, y, y_hat):
        return self._loss_function(y, y_hat)
