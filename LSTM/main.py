import numpy as np
from rnn import RNN
from layers import LSTM

def main():
    input_size, output_size = 3,3
    rnn = RNN()
    rnn.add_layer(LSTM(input_size, output_size))

    X_train = [[[1,0,0]], [[0,1,0]], [[0,0,1]]]
    Y_train = [[[0,1,0]], [[0,0,1]], [[1,0,0]]]
    
    epochs = 1000
    rnn.train(X_train, Y_train, epochs=epochs)
    for p,y in zip(rnn.predict(X_train), Y_train):
        _p = np.zeros_like(p).astype(int)
        _p[:, np.argmax(p)] = 1
        print('%30s %10s %10s' % (p.reshape(1,-1), _p, np.array(y)))

if __name__ == "__main__":
    main()
