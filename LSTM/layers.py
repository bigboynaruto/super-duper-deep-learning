import numpy as np
from nnutils import rand_range, tanh, sigmoid, dtanh, dsigmoid

class RNNLayer:
    def __init__(self, input_size, output_size):
        self._input_size = input_size
        self._output_size = output_size

    def forward(self, x):
        return x

    def backward(self, grad):
        return grad

    def adjust_weights(self):
        pass

    @property
    def input_size(self):
        return self._input_size

    @property
    def output_size(self):
        return self._output_size

class _LSTMCell(RNNLayer):
    def __init__(self, x, lstm, c_t_1=None, h_t_1=None):
        c_t_1 = c_t_1 if c_t_1 is not None else np.zeros(shape=(1, lstm.output_size))
        h_t_1 = h_t_1 if h_t_1 is not None else np.zeros(shape=(1, lstm.output_size))

        self.lstm = lstm
        self.c_t_1 = c_t_1
        self.h_t_1 = h_t_1

        x = np.array(x).reshape(1, -1)
        X = np.concatenate([x, h_t_1], axis=1)
        self.X = X

        self.a = tanh(np.dot(X, lstm.Wa) + lstm.ba)
        self.i = sigmoid(np.dot(X, lstm.Wi) + lstm.bi)
        self.f = sigmoid(np.dot(X, lstm.Wf) + lstm.bf)
        self.o = sigmoid(np.dot(X, lstm.Wo) + lstm.bo)
        self.c = self.a * self.i + c_t_1 * self.f
        self.h = self.c * self.o
    
    @property
    def state(self):
        return self.h, self.c

    def backward(self, dh_t_1, dc_t_1):
        dc = self.o * dh_t_1 + dc_t_1
        di = self.a * dc * dsigmoid(self.i) 
        df = self.c_t_1 * dc * dsigmoid(self.f)
        do = self.c * dh_t_1 * dsigmoid(self.o) 
        da = self.i * dc * dtanh(self.a)

        self.lstm.dWi += np.dot(self.X.T, di)
        self.lstm.dWf += np.dot(self.X.T, df)
        self.lstm.dWo += np.dot(self.X.T, do)
        self.lstm.dWa += np.dot(self.X.T, da)
        self.lstm.dbi += di
        self.lstm.dbf += df       
        self.lstm.dbo += do
        self.lstm.dba += da       

        Ws = [self.lstm.Wi, self.lstm.Wf, self.lstm.Wo, self.lstm.Wa]
        ds = [di, df, do, da]
        dX = sum([np.dot(d, W.T) for W,d in zip(Ws, ds)])

        self.lstm.dc_acc = dc * self.f
        self.lstm.dh_acc = dX[:, self.lstm.input_size:]

        return dX[:, :self.lstm.input_size]

class LSTM(RNNLayer):
    def __init__(self, input_size, output_size):
        super(LSTM, self).__init__(input_size, output_size)

        io_size = input_size + output_size
        
        self.Wf = rand_range(-0.1, 0.1, io_size, output_size)
        self.Wi = rand_range(-0.1, 0.1, io_size, output_size)
        self.Wo = rand_range(-0.1, 0.1, io_size, output_size)
        self.Wa = rand_range(-0.1, 0.1, io_size, output_size)
        self.bf = rand_range(-0.1, 0.1, 1, output_size)
        self.bi = rand_range(-0.1, 0.1, 1, output_size)
        self.bo = rand_range(-0.1, 0.1, 1, output_size)
        self.ba = rand_range(-0.1, 0.1, 1, output_size) 

        self.dWo = np.zeros(shape=(io_size, output_size))
        self.dWf = np.zeros(shape=(io_size, output_size))
        self.dWi = np.zeros(shape=(io_size, output_size))
        self.dWa = np.zeros(shape=(io_size, output_size))
        self.dbo = np.zeros(shape=(1, output_size))
        self.dbf = np.zeros(shape=(1, output_size))
        self.dbi = np.zeros(shape=(1, output_size))
        self.dba = np.zeros(shape=(1, output_size))
        
        self.dh_acc = np.zeros(shape=(1,output_size))
        self.dc_acc = np.zeros(shape=(1,output_size))

        self.history = []

    def backward(self, grad):
        dh = grad + self.dh_acc
        dc = self.dc_acc
        return self.history.pop().backward(dh, dc)

    def forward(self, X):
        if len(self.history) == 0:
            self.history.append(_LSTMCell(X, self))
        else:
            self.history.append(_LSTMCell(X, self, *self.history[-1].state))

        return self.history[-1].h

    def adjust_weights(self, learning_rate = 1):
        self.Wa -= learning_rate * self.dWa
        self.Wi -= learning_rate * self.dWi
        self.Wf -= learning_rate * self.dWf
        self.Wo -= learning_rate * self.dWo
        self.ba -= learning_rate * self.dba
        self.bi -= learning_rate * self.dbi
        self.bf -= learning_rate * self.dbf
        self.bo -= learning_rate * self.dbo

        self.dWa = np.zeros_like(self.Wa)
        self.dWi = np.zeros_like(self.Wi) 
        self.dWf = np.zeros_like(self.Wf) 
        self.dWo = np.zeros_like(self.Wo) 
        self.dba = np.zeros_like(self.ba)
        self.dbi = np.zeros_like(self.bi) 
        self.dbf = np.zeros_like(self.bf) 
        self.dbo = np.zeros_like(self.bo) 

        self.dh_acc = np.zeros_like(self.dh_acc)
        self.dc_acc = np.zeros_like(self.dc_acc)
