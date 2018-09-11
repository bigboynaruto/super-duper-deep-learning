from cnn import CNN
from layers import *

def test_Conv2D():
    img = np.array([[[1,2,3],
                     [7,8,9],
                     [4,5,6.]]])

    c = Conv2D(img.shape, kernel_size=2)
    c.kernels[0] = np.array([[[-1.,0],
                              [0,2]]])

    output = c.forward(img)
    output_ok = np.array([[[-6.,-5],
                           [9,10]]])

    assert np.all(output == output_ok)

    y = np.zeros(c.output_shape)
    error = output - y

    kernel_ok = c.kernels[0] - np.array([[[12,4],
                                          [135,127]]])

    error_output = c.backward(error)
    error_output_ok = np.array([[[-12,-10,0],
                                 [18,26,5],
                                 [0,-9,-10]]])

    assert np.all(c.kernels[0] == kernel_ok)
    print(error_output)
    print(error_output_ok)
    assert np.all(error_output == error_output_ok)

if __name__ == '__main__':
    test_Conv2D()

    np.random.seed(0)

    img = np.random.randint(-1, 2, (4, 1, 8, 8)).astype('float64')

    cnn = CNN()

    print(cnn.output_shape)
    cnn.add_layer(Conv2D(img[0].shape, num_kernels=4, kernel_size=4, stride=1, padding=0, activation='tanh'))
    print(cnn.output_shape)

    cnn.add_layer(Conv2D(cnn.output_shape, num_kernels=4, kernel_size=4, stride=1, padding=0, activation='atan'))
    print(cnn.output_shape)

    cnn.add_layer(Flatten(cnn.output_shape))
    print(cnn.output_shape)

    cnn.add_layer(Dense(cnn.output_shape[-1], 1, activation='atan'))
    print(cnn.output_shape)

    epochs = 3000

    target = np.array([np.reshape(np.sum(i) * 0.15673, (1,1)) for i in img])

    cnn.train(img, target, epochs, verbosity=1, learning_rate=0.1)
    prediction = np.array(cnn.predict(img))

    print('TARGET:')
    print(np.reshape(target, (1, -1)))

    print('PREDICTION:')
    print(np.reshape(prediction, (1, -1)))
    prediction = np.round(prediction)
    print(np.reshape(prediction, (1, -1)))