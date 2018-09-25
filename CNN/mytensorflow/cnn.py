from tqdm import tqdm
from mytensorflow.layers import CNNLayer
from mytensorflow.nnutils import *
import tensorflow as tf

class CNN:
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: CNNLayer):
        self.layers.append(layer)
        self.session = tf.Session()

        self.feeds = []

    def _forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward_batch(output)
        return output

    def _get_variables(self):
        variables = []
        for layer in self.layers:
            variables += layer._get_variables()
        return variables

    def _create_graph(self, x_batch, y_batch, optimizer, loss_function):
        output = self._forward(x_batch)
        cost = loss_function(y_batch, output)
        train = optimizer.minimize(cost)
        return output, train, cost

    def train(self, x_train, y_train, epochs=1, batch_size=1, learning_rate=1., loss_function='mean_squared_error', verbosity=2):
        assert len(x_train) == len(y_train)
        assert verbosity in [1,2,3]

        loss_function = {
            'mean_squared_error' : mean_squared_error,
            'mean_absolute_error' : mean_absolute_error,
            'logcosh_error' : logcosh_error
        }.get(loss_function)

        batches = 1
        if batch_size > 0:
            batches = np.ceil(len(x_train) / batch_size)
            x_train = np.array_split(x_train, batches)
            y_train = np.array_split(y_train, batches)

        total = len(x_train)

        sess = self.session

        init = tf.variables_initializer(self._get_variables())
        sess.run(init)

        xs = tf.placeholder(tf.float32, shape=[None, *self.input_shape])
        ys = tf.placeholder(tf.float32, shape=[None, *self.output_shape])

        self.feeds = [xs, ys]

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        self.predictor,train,cost = self._create_graph(xs, ys, optimizer, loss_function)

        for epoch in range(epochs):
            pbar = zip(x_train, y_train)
            if verbosity == 2:
                pbar = tqdm(pbar, total=total, desc=('Epoch %3d' % (epoch + 1)))

            error_total = 0

            for x_batch, y_batch in pbar:
                _,error = sess.run([train, cost], feed_dict={xs:x_batch, ys:y_batch})
                error_total += error

            if verbosity in [1,2]:
                print('Loss: %f' % (error_total / batches or 1))

    def predict(self, x_test):
        sess = self.session
        res = sess.run(self.predictor, feed_dict={self.feeds[0] : x_test})
        # self._forward(np.array(x_test))
        return res

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