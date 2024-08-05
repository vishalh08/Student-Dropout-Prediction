

# CS 613 - Machine Learning Project - Student Dropout prediction                                                                                                                                                                                                                                                                                                                            
# This program code is implementation of Decision Tree algorithm
# Authors - Aditya Akolkar, Pranjalee Kshirsagar.

import pandas as pd
import numpy as np
from collections import Counter
import os


# Simple Standard Scaler
class StandardScaler:
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Decision Tree classifier for binary classification
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, max_depth=7, min_samples_split=1, min_samples_leaf=2, pruning=True):
        self.max_depth = max_depth
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.pruning = pruning
        
    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(2)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(value=predicted_class)

        # Stopping criteria
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return node
        
        if self.pruning and (self.max_depth is not None and depth >= self.max_depth or
                            len(set(y)) == 1 or
                            len(X) == 0 or len(X[0]) == 0 or
                            len(X) < self.min_samples_split):
            return Node(value=np.bincount(y).argmax())
        if len(set(y)) == 1:  # If all instances in the subset belong to the same class
            return Node(value=y[0])
        if len(X) == 0 or len(X[0]) == 0:  # If the dataset is empty or no more features to split on
            return Node(value=Counter(y).most_common(1)[0][0])

        # Select the best split
        num_features = X.shape[1]
        best_feature, best_threshold = None, None
        best_gain = -1
        current_impurity = self._entropy(y)
        for feature in range(num_features):
            thresholds, classes = zip(*sorted(zip(X[:, feature], y)))
            for i in range(1, len(y)):  # split at each unique value
                if thresholds[i] == thresholds[i - 1]:
                    continue
                left_y, right_y = classes[:i], classes[i:]
                gain = self._information_gain(left_y, right_y, current_impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = (thresholds[i] + thresholds[i - 1]) / 2  # mean value
                
        if best_gain == -1:
            return node
        
        # Split
        indices_left = X[:, best_feature] < best_threshold
        X_left, y_left = X[indices_left], y[indices_left]
        X_right, y_right = X[~indices_left], y[~indices_left]

        if len(X_left) < self.min_samples_leaf or len(X_right) < self.min_samples_leaf:
            return Node(value=np.bincount(y).argmax())

        left_child = self._grow_tree(X_left, y_left, depth + 1)
        right_child = self._grow_tree(X_right, y_right, depth + 1)
        return Node(best_feature, best_threshold, left_child, right_child)
    
    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.root
        while not node.is_leaf_node():
            if inputs[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / counts.sum()
        return 1 - sum(p ** 2 for p in probabilities)

    def _information_gain(self, left_y, right_y, current_impurity):
        p = float(len(left_y)) / (len(left_y) + len(right_y))
        return current_impurity - p * self._entropy(left_y) - (1 - p) * self._entropy(right_y)
    
def evaluate_model(y_true, y_pred):

    # Calculate Recall
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    recall = tp / (tp + fn)
    
    # Calculate Preicision:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    precision = tp / (tp + fp)

    # Calculate the F1 score:
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Compute confusion matrix
    # Identify unique classes
    classes = np.unique(np.concatenate((y_true, y_pred)))
    class_indices = {k: v for v, k in enumerate(classes)}
    
    # Create a square matrix for these classes, initialized to 0
    conf_matrix = np.zeros((len(classes), len(classes)), dtype=int)
    
    # Increment the appropriate cell for each pair of true/predicted labels
    for actual, predicted in zip(y_true, y_pred):
        i = class_indices[actual]
        j = class_indices[predicted]
        conf_matrix[i, j] += 1

    return recall, precision, f1, conf_matrix

# Load your data
np.random.seed(0)
# Get the current working directory
current_dir = os.getcwd()
# Construct the file path
train_file_path = os.path.join(current_dir, 'data_train.csv')
train_data = pd.read_csv(train_file_path, header=0)
test_file_path = os.path.join(current_dir, 'data_test.csv')
test_data = pd.read_csv(test_file_path, header=0)

# Set option to opt-in to the future behavior
#pd.set_option('future.no_silent_downcasting', True)
target_column_index = train_data.shape[1] - 1

# Remove data with Enrolled target.
train_data = train_data[train_data.iloc[:, target_column_index] != 'Enrolled']
test_data = test_data[test_data.iloc[:, target_column_index] != 'Enrolled']

train_data.iloc[:, target_column_index] = train_data.iloc[:, target_column_index].replace({'Dropout': 0, 'Graduate': 1})
test_data.iloc[:, target_column_index] = test_data.iloc[:, target_column_index].replace({'Dropout': 0, 'Graduate': 1})

# Convert data to NumPy arrays for simplicity
X_train = np.asarray(train_data.iloc[:, :-1])
y_train = np.asarray(train_data.iloc[:, target_column_index])

X_test = np.asarray(test_data.iloc[:, :-1])
y_test = np.asarray(test_data.iloc[:, target_column_index])

# Scale the features using mean of x_train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = DecisionTreeClassifier()

clf.fit(X_train_scaled, y_train.astype(int))

# Predictions
predictions = clf.predict(X_test_scaled)

# Change the target values back to original
origina_target = {0:'Dropout', 1: 'Graduate'}
predictions_plot = np.vectorize(origina_target.get)(predictions)

# Evaluation (Here we just do accuracy for simplicity)
accuracy = np.mean(predictions == y_test)
print(f'Accuracy: {accuracy*100:.2f}%')

recall, precision, f1, conf_matrix = evaluate_model(y_test, predictions)
print(f"Recall: {recall*100:.2f}%")
print(f"Precision: {precision*100:.2f}%")
print(f"F1 score: {f1*100:.2f}%")
print(f"Confusion Matrix:\n",conf_matrix)

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(3, 2))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# Accuracy: 97.71%
# Recall is: 98.19%
# Precision is: 98.06%
# F1 score is: 98.12%
# Confusion Matrix is:
#  [[1378   43]
#  [  40 2169]]