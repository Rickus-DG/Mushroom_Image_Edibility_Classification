#  Ideas for gradient descent implementation inspired by the work of Patrick Loeber.
#
#  https://www.python-engineer.com/courses/mlfromscratch/07_svm/

import os
import sys
import numpy as np

import numpy as np
from statistics import mode


class SVC:

    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000, kernel='linear'):

        self.w = None
        self.b = None
        self.separate_models = []
        self.vs = []
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.num_classes = 0
        self.separate_classes = []
        self.kernel = kernel

    def dot(self, x1, x2):
        """Defines the dot product that will be used in this project"""

        if self.kernel == 'linear':
            return np.dot(x1, x2)

    def fit(self, X_train, y_train):
        """Fits support vector classifier with gradient descent"""

        self.w = np.zeros(X_train.shape[1])
        self.b = 0

        for i in np.arange(self.n_iters):

            for j in np.arange(X_train.shape[0]):
                sample = X_train[j]
                if y_train[j] * (self.dot(sample, self.w) - self.b) >= 1:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - self.dot(sample, y_train[j]))
                    self.b -= self.learning_rate * y_train[j]

    def pred(self, X):
        """Returns on what side of the line the X sample is therefore the binary classification"""

        return np.sign(self.dot(X, self.w) - self.b)

    def prob(self, X):
        """Returns distance from the line, used as a measure of how distinct a point is"""

        return self.dot(X, self.w) - self.b

    def proba_score(self, X_test):
        """Gets distance from the line for all samples in a bag"""

        prob = np.zeros(X_test.shape[0])
        for i in np.arange(X_test.shape[0]):
            prob[i] = self.prob(X_test[i])
        return prob

    def predict(self, X_test):
        """Gets classification labels for all samples in a bag"""

        y_pred = np.empty(X_test.shape[0])
        for i in np.arange(X_test.shape[0]):
            y_pred[i] = self.pred(X_test[i])

        return y_pred

    def score(self, X_test, y_test):
        """Predict a label for all samples in X_test then return how many matched the labels in y_test."""

        correct = np.sum(self.predict(X_test) == y_test)
        return correct / X_test.shape[0]


class SVC_multi:

    def __init__(self, learning_rate=0.01, lambda_param=0.01, n_iters=1000, kernel='linear'):
        self.w = None
        self.b = None
        self.separate_models = []
        self.vs = []
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.num_classes = 0
        self.separate_classes = []
        self.kernel = kernel

    def fit(self, X_train, y_train, method='one_v_one'):
        """Fits multi-calss support vector classifier with method chosen"""

        if method == 'one_v_one':
            self.fit_one_v_one(X_train, y_train)
        elif method == 'one_v_rest':
            self.fit_one_v_rest(X_train, y_train)
        else:
            raise Exception("Give method='one_v_one' or method='one_v_rest' \n Default: method='one_v_one'")

    def fit_one_v_one(self, X_train, y_train):
        """Fits an array of support vector classifiers using one vs one strategy."""

        class_names = set(y_train)
        self.num_classes = len(class_names)
        self.separate_classes = []
        self.separate_models = []

        # Create list of lists to contain the values for the separate classes
        for num in np.arange(self.num_classes):
            self.separate_classes.append([])

        # Append samples onto correct sublist of separate_classes with index being the sample's label
        for i in np.arange(X_train.shape[0]):
            self.separate_classes[int(y_train[i])].append(X_train[i])

        # Turn each sublist into a numpy array
        for num in np.arange(self.num_classes):
            self.separate_classes[num] = np.array(self.separate_classes[num])

        # Train models to be stored in separate_models and save the labels being compared in self.vs
        for i in np.arange(self.num_classes):
            for j in np.arange(i + 1, self.num_classes):
                svm = SVC(self.learning_rate, self.lambda_param, self.n_iters, self.kernel)
                y = np.append(np.ones(self.separate_classes[i].shape[0]) * -1,
                              np.ones(self.separate_classes[j].shape[0]))
                X = np.append(self.separate_classes[i], self.separate_classes[j], axis=0)
                self.vs.append([i, j])
                svm.fit(X, y)
                self.separate_models.append(svm)

    def fit_one_v_rest(self, X_train, y_train):
        """Fits an array of support vector classifiers using one vs rest strategy."""

        class_names = set(y_train)
        self.num_classes = len(class_names)
        self.separate_classes = []
        self.separate_models = []

        for i in np.arange(self.num_classes):
            self.vs.append([i, self.num_classes + 1])

        # Create list of lists to contain the values for the separate classes
        for num in np.arange(self.num_classes):
            self.separate_classes.append([])

        # Append samples onto correct sublist of separate_classes with index being the sample's label
        for i in np.arange(X_train.shape[0]):
            self.separate_classes[int(y_train[i])].append(X_train[i])

        # Turn each sublist into a numpy array
        for num in np.arange(self.num_classes):
            self.separate_classes[num] = np.array(self.separate_classes[num])

        # Train models to be stored in separate_models
        for i in np.arange(self.num_classes):
            rest = self.separate_classes.copy()
            rest.pop(i)
            rest_np = []
            for label in rest:
                for val in label:
                    rest_np.append(val)
            class_selected = np.array(self.separate_classes[i])
            rest = np.array(rest_np)
            svm = SVC(self.learning_rate, self.lambda_param, self.n_iters, self.kernel)
            y = np.append(np.ones(class_selected.shape[0]) * -1, np.ones(rest.shape[0]))
            X = np.append(class_selected, rest, axis=0)
            svm.fit(X, y)
            self.separate_models.append(svm)

    def predict(self, X):
        """Used for predicting labels for all samples in a bag for a multi-classification problem"""

        results = []
        scores = np.zeros([X.shape[0], self.num_classes])

        y = np.empty(X.shape[0])
        for model in self.separate_models:
            results.append(model.proba_score(X))
        results = np.array(results).T

        for j in np.arange(results.shape[1]):
            vs = self.vs[j]
            for i in np.arange(results.shape[0]):
                val = results[i, j]
                if val <= 0:
                    scores[i, vs[0]] += np.absolute(val)
                if val >= 0:
                    if vs[1] != self.num_classes + 1:
                        scores[i, vs[1]] += np.absolute(val)

        for i in np.arange(y.shape[0]):
            score = scores[i, :].tolist()
            y[i] = score.index(max(score))

        return y

    def score(self, X_test, y_test):
        """Predict a label when multi-class classification is done for all samples
           in X_test then return how many matched the labels in y_test."""

        correct = np.sum(self.predict(X_test) == y_test)
        return correct / X_test.shape[0]