import numpy as np

# dataset
X = np.array([[1, 1],
              [2, 3],
              [3, 2],
              [4, 5]], dtype=float)

y = np.array([4, 7, 6, 10], dtype=float)

# Initialize
m = np.zeros(2)
c = 0
alpha = 0.01
epochs = 1500
n = len(y)

for _ in range(epochs):
    y_pred = X @ m + c

    dm = (-2/n) * (X.T @ (y - y_pred))
    dc = (-2/n) * sum(y - y_pred)

    m -= alpha * dm
    c -= alpha * dc

print("Weights:", m)
print("Bias:", c)
