import tensorflow as tf
from cnn import CNN, OneHotEncoder
from layers import *

def main():
    # load datasets
    mnist = tf.keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = np.true_divide(X_train, 255)
    X_test = np.true_divide(X_test, 255)

    # X_train = [np.subtract(x, np.mean(x)) / np.std(x) for x in X_train]
    # X_test = [np.subtract(x, np.mean(x)) / np.std(x) for x in X_test]

    # take subset of datasets
    train_size, test_size = 50, 20
    X_train = X_train[:train_size]
    Y_train = Y_train[:train_size]
    X_test = X_test[:test_size]
    Y_test = Y_test[:test_size]

    # reshape to 3d-arrays
    X_train = [np.array([x], dtype='float64') for x in X_train]
    X_test = [np.array([x], dtype='float64') for x in X_test]

    # encode labels
    encoder = OneHotEncoder(np.concatenate((Y_train, Y_test)))
    Y_train_encoded = np.array([np.array([encoder.encode(y)]) for y in Y_train])
    Y_test_encoded = np.array([np.array([encoder.encode(y)]) for y in Y_test])

    image_shape = X_train[0].shape

    # create Convolution Neural Network
    cnn = CNN()

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

    epochs = 30
    cnn.train(X_train, Y_train_encoded, epochs=epochs, learning_rate=0.1, verbosity=2)

    prediction = np.reshape(cnn.predict(X_test), (test_size,-1)) 
    prediction_rounded = np.reshape(np.round(prediction), (test_size, -1))
    print('Test loss        : %f' % (mean_squared_error(prediction, Y_test_encoded)))
    print('Test loss rounded: %f' % (mean_squared_error(prediction_rounded, Y_test_encoded)))
    total_ok = 0
    for p, p_rounded, ye, y in zip(prediction, prediction_rounded, Y_test_encoded, Y_test):
        enc = encoder.decode(p_rounded, -1)
        ye = np.reshape(ye, (-1))
        print('%s\n %s (%s) : %s (%s)' % (p, p_rounded, enc, ye, y))
        if enc == y:
            total_ok += 1

    print('Test accuracy: %f' % (total_ok / test_size))


if __name__ == '__main__':
    main()
