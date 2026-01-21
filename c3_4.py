import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Dataset
X = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([0, 0, 1, 1], dtype=float)

w = 0
b = 0
alpha = 0.1
epochs = 1000

for _ in range(epochs):
    z = w * X + b
    y_pred = sigmoid(z)

    dw = np.mean((y_pred - y) * X)
    db = np.mean(y_pred - y)

    w -= alpha * dw
    b -= alpha * db

print("Weight:", w)
print("Bias:", b)
