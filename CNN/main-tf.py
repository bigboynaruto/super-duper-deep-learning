from mytensorflow.cnn import CNN, OneHotEncoder
from mytensorflow.layers import *
from mynumpy.nnutils import mean_squared_error as mse

def main():
    # load datasets
    mnist = tf.keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = np.true_divide(X_train, 255)
    X_test = np.true_divide(X_test, 255)

    # X_train = [np.subtract(x, np.mean(x)) / np.std(x) for x in X_train]
    # X_test = [np.subtract(x, np.mean(x)) / np.std(x) for x in X_test]

    # take subset of datasets
    train_size, test_size = 10000, 1000
    X_train = X_train[:train_size]
    Y_train = Y_train[:train_size]
    X_test = X_test[:test_size]
    Y_test = Y_test[:test_size]

    # reshape to 3d-arrays
    X_train = [np.array([x], dtype='float32') for x in X_train]
    X_test = [np.array([x], dtype='float32') for x in X_test]

    # encode labels
    encoder = OneHotEncoder(np.concatenate((Y_train, Y_test)))
    Y_train_encoded = np.array([np.array([encoder.encode(y)]) for y in Y_train])
    Y_test_encoded = np.array([np.array([encoder.encode(y)]) for y in Y_test])

    image_shape = X_train[0].shape

    # create Convolution Neural Network
    cnn = CNN()

    # add layers
    print(cnn.output_shape)
    cnn.add_layer(Conv2D(image_shape, num_kernels=32, kernel_size=3, activation='relu'))

    print(cnn.output_shape)
    cnn.add_layer(MaxPooling2D(cnn.output_shape, pool_shape=(3, 3)))

    print(cnn.output_shape)
    cnn.add_layer(Conv2D(cnn.output_shape, num_kernels=32, kernel_size=3, activation='relu'))

    print(cnn.output_shape)
    cnn.add_layer(MaxPooling2D(cnn.output_shape, pool_shape=(3, 3)))

    print(cnn.output_shape)
    cnn.add_layer(Flatten(cnn.output_shape))

    print(cnn.output_shape)
    cnn.add_layer(Dense(cnn.output_shape[-1], 128, activation='relu'))
    print(cnn.output_shape)
    cnn.add_layer(Dense(cnn.output_shape[-1], len(encoder.abc), activation='sigmoid'))
    print(cnn.output_shape)

    # train
    epochs = 10
    cnn.train(X_train, Y_train_encoded, epochs=epochs, batch_size=32, learning_rate=0.1, verbosity=2)

    # make prediction
    prediction = np.reshape(cnn.predict(X_test), (test_size,-1))  # np.reshape(cnn.predict(X_test), Y_test.shape)
    prediction_rounded = np.reshape(np.round(prediction), (test_size, -1))

    print('Test loss        : %f' % (mse(prediction, Y_test_encoded)))
    print('Rounded test loss: %f' % (mse(prediction_rounded, Y_test_encoded)))

    total_ok = 0
    predict_table = np.zeros(shape=(11,10))
    for p, p_rounded, ye, y in zip(prediction, prediction_rounded, Y_test_encoded, Y_test):
        enc = encoder.decode(p_rounded, -1)
        ok = enc == y
        predict_table[enc, y] += 1
        if ok:
            total_ok += 1

    print('Test accuracy: %f' % (total_ok / test_size))
    
    print('p\y 0   1   2   3   4   5   6   7   8   9')
    for i in range(11):
        print('?' if i == 10 else i, end=' ')
        for k in predict_table[i, :]:
            print('%3d' % k, end=' ')
        print('')

if __name__ == '__main__':
    main()
