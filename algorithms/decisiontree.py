#  Ideas for functions strongly influenced by the work of Jason Brownlee
#
#  https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

import numpy as np
import math


class TreeClassifier:

    def __init__(self, classes=[0,1]):
        self.X_train = None
        self.y_train = None
        self.train_data = None
        self.classes = classes
        self.root = None

    def fit(self, X_train, y_train, tree_height=10, length_stopping_criterion=5):
        """Fits and builds tree for the model"""

        self.X_train = X_train
        self.y_train = np.array(np.reshape(y_train, (y_train.shape[0], 1)))
        self.train_data = np.hstack((X_train, self.y_train))
        self.root = self.__generate_tree(self.train_data, tree_height, length_stopping_criterion)

    def fit_train_data(self, train_data, tree_height=10, length_stopping_criterion=5):
        """Fits and builds tree for data that has targets include in data-set"""

        self.train_data = train_data
        self.root = self.__generate_tree(self.train_data, tree_height, length_stopping_criterion)

    def predict(self, X_test):
        """Takes all samples in X_test and predicts a label"""

        predicted = np.zeros(X_test.shape[0])
        for i in np.arange(X_test.shape[0]):
            predicted[i] = self.pred(self.root, X_test[i])
        return predicted
    
    def __scoring_metric(self, all_labels):
        """Calculates the gini index for a dividing the data"""

        num_labels = 0
        for labels in all_labels:
            num_labels += len(labels)

        gini_ind = float(0)
        for labels in all_labels:
            labels_length = len(labels)

            if labels_length != 0:
                score_holder = float(0)

                for target in self.classes:
                    frac = [sample[-1] for sample in labels].count(target) / labels_length
                    score_holder += math.pow(frac, 2)
                gini_ind += (1.0 - score_holder) * (labels_length / num_labels)

        return gini_ind

    def __pos_divide(self, col, attrib_val, bag):
        """Creates the possible divides"""

        division_l = list()
        division_r = list()
        for sample in bag:
            if attrib_val > sample[col]:
                division_l.append(sample)
            if attrib_val <= sample[col]:
                division_r.append(sample)
        return division_l, division_r

    def __best_divide(self, data):
        """Returns best divide"""

        top_score = np.inf
        top_col = np.inf
        top_all_labels = None
        top_value = np.inf

        for col in range(len(data[0]) - 1):
            for sample in data:
                all_labels = self.__pos_divide(col, sample[col], data)
                gini = self.__scoring_metric(all_labels)
                if gini < top_score:
                    top_score = gini
                    top_col = col
                    top_all_labels = all_labels
                    top_value = sample[col]
        return {'col': top_col, 'value': top_value, 'all_labels': top_all_labels}

    def __leaf(self, all_labels):
        """Processes node when it is a leaf node"""

        results = [sample[-1] for sample in all_labels]
        return max(set(results), key=results.count)

    def __divide(self, tree_node, tree_height, length_stopping_criterion, current_depth):
        """Recursive method for creating the divisions for the tree"""

        l = tree_node['all_labels'][0]
        r = tree_node['all_labels'][1]
        l_length = len(l)
        r_length = len(r)
        del (tree_node['all_labels'])

        # This is for case which has no divide
        if not l or not r:
            combined = l + r
            tree_node['l'] = self.__leaf(combined)
            tree_node['r'] = self.__leaf(combined)
            return

        # Checks the desired height of the tree has not been exceeded
        if current_depth >= tree_height:
            tree_node['l'] = self.__leaf(l)
            tree_node['r'] = self.__leaf(r)
            return
        # Sorts out the left side of the node
        if l_length <= length_stopping_criterion:
            tree_node['l'] = self.__leaf(l)
        else:
            tree_node['l'] = self.__best_divide(l)
            self.__divide(tree_node['l'], tree_height, length_stopping_criterion, current_depth + 1)

        # Sorts out the right side of the node
        if r_length <= length_stopping_criterion:
            tree_node['r'] = self.__leaf(r)
        else:
            tree_node['r'] = self.__best_divide(r)
            self.__divide(tree_node['r'], tree_height, length_stopping_criterion, current_depth + 1)

    def __generate_tree(self, train, tree_height, length_stopping_criterion):
        """Creates the decision tree by starting the recursive best_divide method"""

        current_depth = 1
        root_node = self.__best_divide(train)
        self.__divide(root_node, tree_height, length_stopping_criterion, current_depth)
        return root_node

    def pred(self, tree_node, sample):
        """Traverses the tree and returns the tree-node that the sample lands in"""

        has_left = isinstance(tree_node['l'], dict)
        has_right = isinstance(tree_node['r'], dict)

        if tree_node['value'] > sample[tree_node['col']]:
            if has_left:
                return self.pred(tree_node['l'], sample)
            else:
                return tree_node['l']
        else:
            if has_right:
                return self.pred(tree_node['r'], sample)
            else:
                return tree_node['r']

    def score(self, X_test, y_test):
        """Predict a label for all samples in X_test then return how many matched the labels in y_test."""

        correct = np.sum(self.predict(X_test) == y_test)
        return correct / X_test.shape[0]

    def print_tree(self):
        self.__show_tree(self.root, 1)

    def __show_tree(self, root, depth=0):
        """Shows Tree Structure"""

        is_dict = isinstance(root, dict)

        if is_dict:
            print('_' * depth,"Col",root['col'], "<",root['value'])
            self.__show_tree(root['l'], depth + 1)
            self.__show_tree(root['r'], depth + 1)

        else:
            print('_' * depth, root)


class RandomForestClassifier:

    def __init__(self, classes=[0, 1]):
        self.X_train = None
        self.y_train = None
        self.train_data = None
        self.classes = classes
        self.tree_height = None
        self.length_stopping_criterion = None
        self.num_trees = None
        self.trees = None

    def fit(self, X_train, y_train, tree_height=10, length_stopping_criterion=1, num_trees=100, ratio=1):
        """Fits and builds tree for the model"""

        self.X_train = X_train
        self.y_train = np.array(np.reshape(y_train, (y_train.shape[0], 1)))
        self.train_data = np.hstack((X_train, self.y_train))
        self.tree_height = tree_height
        self.length_stopping_criterion = length_stopping_criterion
        self.num_trees = num_trees

        self.trees = list()
        for i in range(self.num_trees):
            sample = self.__rand_sub_set(self.train_data, ratio)
            tree = TreeClassifier(classes=self.classes)
            tree.fit_train_data(sample, tree_height=self.tree_height,
                                length_stopping_criterion=self.length_stopping_criterion)
            self.trees.append(tree)

    def __rand_sub_set(self, train_data, ratio):
        """Returns random subset of the train_data of size determined by ratio allowing multiples of the same sample"""

        num_samples = int(np.around(train_data.shape[0] * ratio))
        random_indices = np.random.choice(train_data.shape[0], size=num_samples, replace=True)
        random_rows = train_data[random_indices, :]
        return random_rows

    def __bagging(self, trees, sample):
        """Carries out bagging, returns predictions from each tree in the trees parameter for sample row"""

        predictions = [tree.pred(tree.root, sample) for tree in trees]
        return max(set(predictions), key=predictions.count)

    def predict(self, X_test):
        """Takes all samples in X_test and predicts a label"""

        predictions = [self.__bagging(self.trees, sample) for sample in X_test]
        return np.array(predictions)

    def score(self, X_test, y_test):
        """Predict a label for all samples in X_test then return how many matched the labels in y_test."""

        correct = np.sum(self.predict(X_test) == y_test)
        return correct / X_test.shape[0]