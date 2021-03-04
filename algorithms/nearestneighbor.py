import numpy as np
from statistics import mode


class KnnClassifier:
    """This class implements the k nearest neighbor algorithm for classification"""

    def __init__(self, k):
        "Sets both X_train and y_train to a blank np.array."

        self.k = k
        self.X_train = np.array([])
        self.y_train = np.array([])

    def fit(self, X_train, y_train):
        """Assign training data to the attributes."""

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

    def __pred(self, v):
        """Run nearest neighbour algorithm on a sample and return a predicted label."""

        allV = np.full(self.X_train.shape, v)
        diff = np.asarray(self.X_train - allV)
        diffsquared = diff * diff
        squaresum = np.sum(diffsquared, axis=1)
        dists = np.sqrt(squaresum)
        distances = list(zip(dists, self.y_train))
        dtype = [('dist', float), ('label', float)]
        struct_dist = np.array(distances, dtype=dtype)
        struct_dist = np.sort(struct_dist, order='dist', axis=0, kind='mergesort')
        top_dists = np.array(struct_dist[:self.k])
        return mode(top_dists['label'].tolist())

    def score(self, X_test, y_test):
        """Predict a label for all samples in X_test then return how many matched the labels in y_test."""

        correct = np.sum(self.predict(X_test) == y_test)
        return correct / X_test.shape[0]

    def predict(self, X_test):
        """Predict a label for all samples in X_test"""

        predicted = np.zeros(X_test.shape[0])
        for i in np.arange(X_test.shape[0]):
            predicted[i] = self.__pred(X_test[i])
        return predicted
