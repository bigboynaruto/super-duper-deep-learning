from abc import ABCMeta, abstractmethod
import numpy as np
import math


def count_where(l, cond):
    '''
    Count a number of elements that meet condition `cond`.
    '''
    return sum(1 if cond(el) else 0 for el in l)


def probability(X, Y, label, dim, val):
    '''
    Probability of finding `label` for data with `val` on position `dim`.
    '''
    c = count_where(zip(X, Y), lambda x: x[0][dim] == val and x[1] == label)
    s = count_where(Y, lambda y: y == label)
    return c / s


def find_nearest(x, values):
    '''
    Finds nearest point to `x` from `values`.
    '''
    d = np.abs(values - x)
    return values[np.argmin(d)]


def gauss(x, mean, std):
    '''
    Calculates probability density of Normal distribution.
    '''
    return math.exp(-(x - mean)**2 / (2 * std**2)) / \
        math.sqrt(2 * math.pi * std**2)


class Classifier:
    '''
    Base class for classifiers.
    '''
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def train(self, X, Y):
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError("Abstract method")


class NaiveBayesAbstract(Classifier):
    '''
    Base class for NB-classifiers.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, classes):
        super().__init__()
        self._classes = list(set(classes))
        pass

    @abstractmethod
    def _estimate_probability(self, x, y):
        raise NotImplementedError("Abstract method")

    def _predict_single(self, x):
        probs = [
            self._estimate_probability(x, y)
            for y in self._classes
        ]
        arg = np.argmax(probs)
        return self._classes[arg]

    def predict(self, X):
        return [self._predict_single(x) for x in X]


class GaussianNaiveBayes(NaiveBayesAbstract):
    '''
    Gaussian Naive Bayes implementation for continuous data.
    '''

    def __init__(self, classes):
        super().__init__(classes)

    def train(self, X, Y):
        if len(X) != len(Y):
            raise Exception(
                'Inconsistent feature matrix and label vector size')

        self._n_dim = len(X[0])

        self._params = {
            y: [
                self._estimate_gaussian_params(X, Y, y, dim)
                for dim in range(self._n_dim)
            ]
            for y in self._classes
        }

    def _estimate_gaussian_params(cls, X, Y, label, dim):
        xs = [x[dim] for x, y in zip(X, Y) if y == label]
        return np.mean(xs), np.std(xs)

    def _estimate_probability(self, x, y):
        params = self._params[y]
        return np.prod([
            gauss(xx, *p)
            for xx, p in zip(x, params)
        ])


class NaiveBayes(NaiveBayesAbstract):
    '''
    Naive Bayes implementation for categorical data.
    '''

    def __init__(self, classes):
        super().__init__(classes)

    def train(self, X, Y, alpha=1):
        if len(X) != len(Y):
            raise Exception(
                'Inconsistent feature matrix and label vector size')

        self._n_dim = len(X[0])

        # classes
        self._classes = list(set(Y))

        # number of categories by dimension
        self._xs = [np.unique(xs) for xs in np.transpose(X)]

        # probabilities
        # n_dim x n_classes x k,
        # where k=len(xs[i]) - number of categories in dimension i
        self._probs = [
            {
                y: {
                    x: probability(X, Y, y, dim, x)
                    for x in self._xs[dim]
                }
                for y in self._classes
            }
            for dim in range(self._n_dim)
        ]

        # smoothing
        for prob_dim in self._probs:
            for y, x_dict in prob_dim.items():
                n = len(x_dict)
                for x in x_dict.keys():
                    v = x_dict[x]
                    x_dict[x] = (v * n + alpha) / (n + alpha * n)

    def _estimate_probability(self, x, y):
        return np.prod([
            prob[y].get(find_nearest(x_given, x_data), 0)
            for x_given, x_data, prob in zip(x, self._xs, self._probs)
        ])
