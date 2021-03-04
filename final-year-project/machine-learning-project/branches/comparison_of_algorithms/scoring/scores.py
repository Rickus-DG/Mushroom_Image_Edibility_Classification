import numpy as np
import math

class cross_validation_score:
    """Does Scoring Cross Validation"""

    def score(self, model, X, y, cv=5, scaler=None):
        """Given model and X, y returns cross val scores of number of folds cv"""

        totalSize = X.shape[0]

        sizes = self.getSegmentSizes(X, cv)
        scores = list()
        counter = 0
        for size in sizes:
            lowerBound = int(counter)
            upperBound = int(counter + size)
            X_test = np.array(X[lowerBound:upperBound,:])
            y_test = np.array(y[lowerBound:upperBound])
            X_train = np.vstack((np.array(X[:lowerBound,:]), np.array(X[upperBound:,:])))
            y_train = np.hstack((np.array(y[:lowerBound]), np.array(y[upperBound:])))

            if scaler != None:
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)

            scores.append(model.score(X_test, y_test))

            counter = counter + size
        return scores

    def getSegmentSizes(self, X, cv):
        """Returns the number of samples which should be in each fold"""

        totalSize = X.shape[0]
        if cv > totalSize:
            raise Exception("Set too small")
        segment = int(np.floor(totalSize/cv))
        sizes = np.empty([cv])
        sizes.fill(int(segment))
        leftOver = totalSize - (segment * cv)

        sizes[:leftOver] = sizes[:leftOver] + 1

        return sizes


class confusion_matrix:

    def __init__(self):
        self.truePositives = 0
        self.falsePositives = 0
        self.trueNegatives = 0
        self.falseNegatives = 0

    def generate(self, y_pred, y_test):
        """Create confusion matrix"""

        self.truePositives = 0
        self.falsePositives = 0
        self.trueNegatives = 0
        self.falseNegatives = 0

        for i in np.arange(y_test.shape[0]):
            if y_test[i] == 1 and y_pred[i] == 1:
                self.truePositives += 1

            if y_test[i] == 0 and y_pred[i] == 1:
                self.falsePositives += 1

            if y_test[i] == 0 and y_pred[i] == 0:
                self.trueNegatives += 1

            if y_test[i] == 1 and y_pred[i] == 0:
                self.falseNegatives += 1

        return np.array([[self.truePositives, self.falsePositives], [self.falseNegatives, self.trueNegatives]])

    def sensitivity(self):
        """Get sensitivity score"""

        return self.truePositives/(self.truePositives + self.falseNegatives)

    def specificity(self):
        """Get specificity score"""

        return self.trueNegatives/(self.trueNegatives + self.falsePositives)

    def precision(self):
        """Get precision score"""

        return self.truePositives/(self.truePositives + self.falsePositives)

    def recall(self):
        """Get recall score"""

        return self.truePositives/(self.truePositives + self.falseNegatives)

    def accuracy(self):
        """Get accuracy score"""

        return (self.truePositives + self.trueNegatives)/(self.truePositives + self.trueNegatives + self.falsePositives + self.falseNegatives)


