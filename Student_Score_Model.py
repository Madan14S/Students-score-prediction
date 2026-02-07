
"""
Linear Regression From Scratch (Math Only)
Author: Madan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/student_scores.csv")

X = data["Hours"].values
y = data["Score"].values

# Normalize feature
X = (X - np.mean(X)) / np.std(X)

# Train-test split
split = int(0.8 * len(X))
X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# Initialize parameters
m = 0.0
b = 0.0
learning_rate = 0.01
epochs = 1000
n = len(X_train)

costs = []

# Gradient Descent
for i in range(epochs):
    y_pred = m * X_train + b
    cost = (1 / n) * np.sum((y_pred - y_train) ** 2)
    costs.append(cost)

    dm = (2 / n) * np.sum((y_pred - y_train) * X_train)
    db = (2 / n) * np.sum(y_pred - y_train)

    m -= learning_rate * dm
    b -= learning_rate * db

# Test predictions
y_test_pred = m * X_test + b

# Evaluation
mse = np.mean((y_test - y_test_pred) ** 2)
ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
ss_res = np.sum((y_test - y_test_pred) ** 2)
r2 = 1 - (ss_res / ss_total)

print("Slope (m):", m)
print("Intercept (b):", b)
print("MSE:", mse)
print("R2 Score:", r2)

# Plot regression line
plt.scatter(X, y)
plt.plot(X, m * X + b)
plt.xlabel("Study Hours (normalized)")
plt.ylabel("Score")
plt.title("Linear Regression From Scratch")
plt.show()

# Plot cost curve
plt.plot(costs)
plt.xlabel("Epochs")
plt.ylabel("Cost (MSE)")
plt.title("Cost Function Curve")
plt.show()
