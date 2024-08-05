import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path, sep=';')
    data = data[data['Target'] != 'Enrolled']   
    data['Target'] = (data['Target'] == 'Dropout').astype(int)
    data.drop(['International', 'Nacionality', "Mother's qualification", "Curricular units 1st sem (evaluations)", 
               "Mother's occupation", "Father's occupation", "Father's qualification", 'Unemployment rate',
               "Application order", "GDP", 'Inflation rate'], axis=1, inplace=True)
    return data

# Gaussian Naive Bayes
def gaussian_naive_bayes(X_train, y_train, X_test):
    class_counts = np.bincount(y_train)
    n_classes = len(class_counts)
    n_features = X_train.shape[1]
    class_means = np.zeros((n_classes, n_features))
    class_variances = np.zeros((n_classes, n_features))

    for label in range(n_classes):
        X_label = X_train[y_train == label]
        class_means[label] = X_label.mean(axis=0)
        class_variances[label] = X_label.var(axis=0)

    class_log_priors = np.log(class_counts / len(y_train))

    log_likelihoods = np.zeros((X_test.shape[0], n_classes))
    for label in range(n_classes):
        class_mean = class_means[label]
        class_variance = class_variances[label]
        class_log_prior = class_log_priors[label]
        log_likelihoods[:, label] = -0.5 * np.sum(np.log(2 * np.pi * class_variance))
        log_likelihoods[:, label] -= 0.5 * np.sum(((X_test - class_mean) ** 2) / class_variance, axis=1)
        log_likelihoods[:, label] += class_log_prior

    return np.argmax(log_likelihoods, axis=1)

# Logistic Regression
def logistic_regression(X_train, y_train, X_test, threshold=0.5):
    X_train = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
    weights = np.zeros(X_train.shape[1])
    learning_rate = 0.01
    num_iterations = 1000
    for _ in range(num_iterations):
        logits = np.dot(X_train, weights)
        y_pred = 1 / (1 + np.exp(-logits))
        error = y_pred - y_train
        gradient = np.dot(X_train.T, error)
        weights -= learning_rate * gradient
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
    logits_test = np.dot(X_test, weights)
    y_pred_test = (1 / (1 + np.exp(-logits_test))) >= threshold
    return y_pred_test.astype(int)

# Decision Tree
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes_ = len(np.unique(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]
        best_entropy = -1.0
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                entropy_left = self._calculate_entropy(num_left)
                entropy_right = self._calculate_entropy(num_right)
                entropy = (i * entropy_left + (m - i) * entropy_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if best_entropy < 0 or entropy < best_entropy:
                    best_entropy = entropy
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = {'predicted_class': predicted_class}
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node['index'] = idx
                node['threshold'] = thr
                node['left'] = self._grow_tree(X_left, y_left, depth + 1)
                node['right'] = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        node = self.tree_
        while 'threshold' in node:
            if inputs[node['index']] < node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node['predicted_class']
    
    def _calculate_entropy(self, class_counts):
        class_counts = np.array(class_counts)
        p = class_counts / np.sum(class_counts)
        return -np.sum(p * np.log2(p + 1e-12))

# Majority Voting Classifier
def majority_voting(predictions):
    ensemble_predictions = np.mean(predictions, axis=0) >= 0.5
    return ensemble_predictions.astype(int)

# Confusion Matrix
def confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[tn, fp], [fn, tp]])

# Load data
file_path = 'data.csv'
data = load_data(file_path)
X = data.drop('Target', axis=1).values
y = data['Target'].values

# Split data into train and validation sets
np.random.seed(0)
indices = np.random.permutation(len(X))
split_point = int(0.67 * len(X))
train_indices, val_indices = indices[:split_point], indices[split_point:]
X_train, X_val = X[train_indices], X[val_indices]
y_train, y_val = y[train_indices], y[val_indices]

# Make predictions with individual classifiers
nb_predictions = gaussian_naive_bayes(X_train, y_train, X_val)
lr_predictions = logistic_regression(X_train, y_train, X_val, threshold=0.7)  
dt = DecisionTree(max_depth=3)
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_val)

# Create ensemble predictions
ensemble_predictions = majority_voting([nb_predictions, lr_predictions, dt_predictions])

# Calculate confusion matrix
ensemble_conf_matrix = confusion_matrix(y_val, ensemble_predictions)

# Calculate metrics
ensemble_accuracy = np.mean(ensemble_predictions == y_val)
ensemble_recall = ensemble_conf_matrix[1][1] / (ensemble_conf_matrix[1][1] + ensemble_conf_matrix[1][0])
ensemble_precision = ensemble_conf_matrix[1][1] / (ensemble_conf_matrix[1][1] + ensemble_conf_matrix[0][1])
ensemble_f1_score = 2 * (ensemble_precision * ensemble_recall) / (ensemble_precision + ensemble_recall)



# Print results for ensemble
print("Ensemble Confusion Matrix:")
print(ensemble_conf_matrix)
print("Ensemble Accuracy:", ensemble_accuracy)
print("Ensemble Recall:", ensemble_recall)
print("Ensemble Precision:", ensemble_precision)
print("Ensemble F1 Score:", ensemble_f1_score)

plt.figure(figsize=(6, 4))
sns.heatmap(ensemble_conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=True)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()