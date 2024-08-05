import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv', sep=';')

data = data[data['Target'] != 'Enrolled']

data['Target'] = (data['Target'] == 'Dropout').astype(int)
# print(data['Target'].head)
data.drop(['International', 'Nacionality', "Mother's qualification", "Curricular units 1st sem (evaluations)",
           "Mother's occupation", "Father's occupation", "Father's qualification", 'Unemployment rate',
           "Application order", "GDP", 'Inflation rate'], axis=1, inplace=True)

X = data.drop('Target', axis=1).values
y = data['Target'].values

def calculate_statistics(X, y):
    statistics = {}
    classes = np.unique(y)
    for class_ in classes:
        X_class = X[y == class_]
        mean = np.mean(X_class, axis=0)
        std_dev = np.std(X_class, axis=0)
        statistics[class_] = {'mean': mean, 'std_dev': std_dev}
    return statistics

def calculate_priors(y):
    priors = {}
    classes, counts = np.unique(y, return_counts=True)
    total_samples = len(y)
    for class_, count in zip(classes, counts):
        priors[class_] = count / total_samples
    return priors

class NaiveBayesClassifier:
    def __init__(self):
        self.statistics = None
        self.priors = None

    def fit(self, X, y):
        self.statistics = calculate_statistics(X, y)
        self.priors = calculate_priors(y)

    def predict(self, X):
        class_probabilities = np.zeros((X.shape[0], len(self.statistics)))
        for class_, class_statistics in self.statistics.items():
            mean = class_statistics['mean']
            std_dev = class_statistics['std_dev']
            class_prior = self.priors[class_]
            std_dev += 1e-10
            likelihood = np.prod((1 / (np.sqrt(2 * np.pi) * std_dev)) * np.exp(-(X - mean) ** 2 / (2 * std_dev ** 2)), axis=1)
            class_probabilities[:, class_] = likelihood * class_prior
        return np.argmax(class_probabilities, axis=1)

def calculate_confusion_matrix(actual, predicted):
    classes = np.unique(np.concatenate((actual, predicted)))
    conf_matrix = np.zeros((len(classes), len(classes)))
    for i, class_ in enumerate(classes):
        for j, class_prime in enumerate(classes):
            conf_matrix[i, j] = np.sum((actual == class_) & (predicted == class_prime))
    return conf_matrix

def precision(true_positives, false_positives):
    return true_positives / (true_positives + false_positives)

def recall(true_positives, false_negatives):
    return true_positives / (true_positives + false_negatives)

def f_measure(precision, recall):
    return 2 * (precision * recall) / (precision + recall)

def accuracy(true_positives, true_negatives, total_samples):
    return (true_positives + true_negatives) / total_samples

np.random.seed(0)
indices = np.random.permutation(len(y))
split = int(0.67 * len(y))
train_indices, test_indices = indices[:split], indices[split:]
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

naive_bayes = NaiveBayesClassifier()
naive_bayes.fit(X_train, y_train)

train_predictions = naive_bayes.predict(X_train)
test_predictions = naive_bayes.predict(X_test)

train_conf_matrix = calculate_confusion_matrix(y_train, train_predictions)
test_conf_matrix = calculate_confusion_matrix(y_test, test_predictions)

train_true_positives = np.sum((y_train == 1) & (train_predictions == 1))
train_false_positives = np.sum((y_train == 0) & (train_predictions == 1))
train_false_negatives = np.sum((y_train == 1) & (train_predictions == 0))
train_true_negatives = np.sum((y_train == 0) & (train_predictions == 0))

test_true_positives = np.sum((y_test == 1) & (test_predictions == 1))
test_false_positives = np.sum((y_test == 0) & (test_predictions == 1))
test_false_negatives = np.sum((y_test == 1) & (test_predictions == 0))
test_true_negatives = np.sum((y_test == 0) & (test_predictions == 0))

train_precision = precision(train_true_positives, train_false_positives)
train_recall = recall(train_true_positives, train_false_negatives)
train_f_measure = f_measure(train_precision, train_recall)
train_accuracy = accuracy(train_true_positives, train_true_negatives, len(y_train))

test_precision = precision(test_true_positives, test_false_positives)
test_recall = recall(test_true_positives, test_false_negatives)
test_f_measure = f_measure(test_precision, test_recall)
test_accuracy = accuracy(test_true_positives, test_true_negatives, len(y_test))

print("Training Precision:", train_precision)
print("Testing Accuracy:", test_accuracy)
print()
print("Testing Precision:", test_precision)
print("Testing Recall:", test_recall)
print("Testing F-measure:", test_f_measure)

plt.figure(figsize=(6, 4))
sns.heatmap(test_conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=True)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Testing Confusion Matrix')
plt.tight_layout()
plt.show()
