#Author : Manish Chandrashekar
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Logistic_Regression_Model:
    def __init__(self, data_file):
        self.data = pd.read_csv(data_file, sep=';')
        
    def preprocess_data(self):
        self.data = self.data[self.data['Target'] != 'Enrolled']
        self.data['Target'] = (self.data['Target'] == 'Dropout').astype(int)
        self.data.drop(['International', 'Nacionality', "Mother's qualification", "Curricular units 1st sem (evaluations)", 
                        "Mother's occupation", "Father's occupation", "Father's qualification", 'Unemployment rate',
                        "Application order", "GDP", 'Inflation rate'], axis=1, inplace=True)
        np.random.seed(0)
        self.data = self.data.sample(frac=1).reset_index(drop=True)
        
        train_size = int(2/3 * len(self.data))
        self.trainingData = self.data.iloc[:train_size]
        self.validationData = self.data.iloc[train_size:]
        
        self.trainingX = self.trainingData.iloc[:, :-1].values
        self.trainingY = self.trainingData.iloc[:, -1].values
        self.validationX = self.validationData.iloc[:, :-1].values
        self.validationY = self.validationData.iloc[:, -1].values
        
        self.mean = np.mean(self.trainingX, axis=0)
        self.std = np.std(self.trainingX, axis=0)
        self.trainingX = (self.trainingX - self.mean) / self.std
        self.validationX = (self.validationX - self.mean) / self.std
    
    def train_model(self, learning_rate=0.1, epochs=1000, tolerance=1e-6):
        self.learningRate = learning_rate
        self.epsilon = 1e-16
        self.epochs = epochs
        self.weights = np.zeros(self.trainingX.shape[1])
        self.tolerance = tolerance
        self.trainingLogloss = []
        self.validationLogloss = []

        for epoch in range(self.epochs):
            z = np.dot(self.trainingX, self.weights)
            sigmoid = 1 / (1 + np.exp(-z))
            loss = -np.mean(self.trainingY * np.log(sigmoid + self.epsilon) + (1 - self.trainingY) * np.log(1 - sigmoid + self.epsilon))
            self.trainingLogloss.append(loss)

            gradientWeights = np.dot(self.trainingX.T, (sigmoid - self.trainingY)) / len(self.trainingY)
            self.weights -= self.learningRate * gradientWeights

            zValidation = np.dot(self.validationX, self.weights)
            sigmoidValidation = 1 / (1 + np.exp(-zValidation))
            lossValidation = -np.mean(self.validationY * np.log(sigmoidValidation + self.epsilon) + (1 - self.validationY) * np.log(1 - sigmoidValidation + self.epsilon))
            self.validationLogloss.append(lossValidation)

            if epoch > 1 and abs(self.trainingLogloss[epoch] - self.trainingLogloss[epoch - 1]) < self.tolerance:
                break

    def evaluate_model(self):
        zValidation = np.dot(self.validationX, self.weights)
        sigmoidValidation = 1 / (1 + np.exp(-zValidation))
        self.yValidationPrediction = (sigmoidValidation >= 0.5).astype(int)

        self.confusion_matrix = np.zeros((2, 2))
        for true_label, predicted_label in zip(self.validationY, self.yValidationPrediction):
            self.confusion_matrix[true_label][predicted_label] += 1
        
        self.truePositive = self.confusion_matrix[1][1]
        self.falsePositive = self.confusion_matrix[0][1]
        self.falseNegative = self.confusion_matrix[1][0]
        self.trueNegative = self.confusion_matrix[0][0]
        self.precision = self.truePositive / (self.truePositive + self.falsePositive)
        self.recall = self.truePositive / (self.truePositive + self.falseNegative)
        self.f_measure = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        self.accuracy = (self.truePositive + self.trueNegative) / np.sum(self.confusion_matrix)
        # Print evaluation metrics
        print("\nPrecision:", self.precision)
        print("Recall:", self.recall)
        print("F-measure:", self.f_measure)
        print("Accuracy:", self.accuracy)
        
    def plot_loss(self):
        plt.plot(range(len(self.trainingLogloss)), self.trainingLogloss, label='Training Loss')
        plt.plot(range(len(self.validationLogloss)), self.validationLogloss, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Log Loss')
        plt.title('Training and Validation Loss vs Epochs')
        plt.legend()
        plt.show()
        
    def plot_confusion_matrix(self):
        plt.figure(figsize=(6, 4))
        sns.heatmap(self.confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=True)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()
    
    def get_predicted_values(self):
        return self.yValidationPrediction
    
Logistic_Regression = Logistic_Regression_Model("datanew.csv")
Logistic_Regression.preprocess_data()
Logistic_Regression.train_model(learning_rate=0.1)
Logistic_Regression.evaluate_model()
Logistic_Regression.plot_loss()
Logistic_Regression.plot_confusion_matrix()

predicted_values = Logistic_Regression.get_predicted_values()
print("Predicted Values:", predicted_values)
