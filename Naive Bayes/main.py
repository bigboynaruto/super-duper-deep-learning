from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn import metrics
from classifiers import NaiveBayes, GaussianNaiveBayes
import numpy as np


def predict_sklearn(X_train, y_train, X_test, y_test):
    # training the model on training set
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)

    # making predictions on the testing set
    return gnb.predict(X_test)


def predict_custom(X_train, y_train, X_test, y_test):
    classifier = NaiveBayes(np.append(y_train, y_test))
    classifier.train(X_train, y_train)
    return classifier.predict(X_test)


def predict_custom_gaussian(X_train, y_train, X_test, y_test):
    classifier = GaussianNaiveBayes(np.append(y_train, y_test))
    classifier.train(X_train, y_train)
    return classifier.predict(X_test)


# load the iris dataset
iris = load_iris()

# store the feature matrix (X) and response vector (y)
X = iris.data
y = iris.target

# splitting X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1)

y_pred_sk = predict_sklearn(X_train, y_train, X_test, y_test)
y_pred = predict_custom(X_train, y_train, X_test, y_test)
y_pred_gauss = predict_custom_gaussian(X_train, y_train, X_test, y_test)

# comparing actual response values (y_test) with predicted response values
# (y_pred)
print(
    "Sklearn model accuracy(in %):",
    metrics.accuracy_score(
        y_test,
        y_pred_sk) *
    100)
print(
    "Custom model accuracy (in %):",
    metrics.accuracy_score(
        y_test,
        y_pred) * 100)
print(
    "Custom gaussian model accuracy (in %):",
    metrics.accuracy_score(
        y_test,
        y_pred_gauss) *
    100)
