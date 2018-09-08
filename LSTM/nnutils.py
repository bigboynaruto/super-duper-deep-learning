import numpy as np

def sigmoid(x): 
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))

def dsigmoid(s, activated=True): 
    return dsigmoid_activated(sigmoid(s))

def dsigmoid_activated(s):
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def dtanh(s):
    return dtanh_activated(tanh(s))

def dtanh_activated(s):
    return 1 - s**2

def softmax(x):
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)

def dsoftmax(x):
    s = x.reshape(-1,1)
    return np.diagflat(s) - np.dot(s, s.T)

def cross_entropy(y, y_hat):
    y_hat = np.clip(y_hat, 1e-11, 1)
    return -np.sum(y * np.log(y_hat))

def quadratic_loss(y, y_hat):
    return np.sum(np.subtract(y_hat, y)**2) / len(y_hat)

def rand_range(low, high, *args): 
    return np.random.rand(*args) * (high - low) + low
