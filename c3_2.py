import numpy as np

# Input data
X = np.array([1, 2, 3, 4, 5], dtype=float)
Y = np.array([3, 4, 2, 4, 5], dtype=float)

m = 0
c = 0
alpha = 0.01
epochs = 1000
n = len(X)

for i in range(epochs):
    Y_pred = m*X + c

    # Compute gradients
    dm = (-2/n) * sum(X * (Y - Y_pred))
    dc = (-2/n) * sum(Y - Y_pred)

    m -= alpha * dm
    c -= alpha * dc

print("Slope (m):", m)
print("Intercept (c):", c)
