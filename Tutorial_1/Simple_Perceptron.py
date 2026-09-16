import numpy as np


class Perceptron(object):

    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def weighted_sum(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        return np.where(self.sigmoid(self.weighted_sum(X)) >= 0.5, 1, 0)


    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        print("Weights:", self.w_)

        for _ in range(self.n_iter):
            error = 0

            for xi, yi in zip(X, y):
                y_pred = self.predict(xi)

                update = self.eta * (yi - y_pred)

                self.w_[1:] = self.w_[1:] + update * xi

                self.w_[0] = self.w_[0] + update

                error += int(update != 0.0)

            self.errors_.append(error)
            print("Updated Weights:", self.w_[1:])

        return self


import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
    header=None,
)

df = shuffle(df)

print(df.head())

X = df.iloc[:, 0:4].values
y = df.iloc[:, 4].values

print("X sample:", X[0:5])
print("y sample:", y[0:5])

from sklearn.model_selection import train_test_split

train_data, test_data, train_labels, test_labels = train_test_split(
    X, y, test_size=0.25
)

train_labels = np.where(train_labels == "Iris-setosa", 1, 0)
test_labels = np.where(test_labels == "Iris-setosa", 1, 0)

print("Train data:", train_data[0:2])
print("Train labels:", train_labels[0:2])
print("Test data:", test_data[0:2])
print("Test labels:", test_labels[0:2])

perceptron = Perceptron(eta=0.1, n_iter=10)
perceptron.fit(train_data, train_labels)

test_preds = perceptron.predict(test_data)
print("Predictions on test data:", test_preds)

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(test_labels, test_preds)
print("Accuracy:", round(accuracy, 2) * 100, "%")


print("\nEnter flower measurements to predict the species:")
sepal_length = float(input("Sepal length (cm): "))
sepal_width = float(input("Sepal width (cm): "))
petal_length = float(input("Petal length (cm): "))
petal_width = float(input("Petal width (cm): "))

manual_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
manual_pred = perceptron.predict(manual_input)

if manual_pred[0] == 1:
    print("Prediction: Iris-setosa")
else:
    print("Prediction: Not Iris-setosa")
