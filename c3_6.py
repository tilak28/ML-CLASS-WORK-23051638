import numpy as np

# Fake dataset
X = np.random.rand(100, 1)
y = 3*X[:,0] + 4 + np.random.randn(100)*0.1

w = 0
b = 0
alpha = 0.05
batch_size = 10
epochs = 50

for epoch in range(epochs):
    idx = np.random.permutation(len(X))
    X = X[idx]
    y = y[idx]

    for i in range(0, len(X), batch_size):
        X_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        y_pred = w * X_batch[:,0] + b

        dw = np.mean((y_pred - y_batch) * X_batch[:,0])
        db = np.mean(y_pred - y_batch)

        w -= alpha * dw
        b -= alpha * db

print("w:", w, "b:", b)
