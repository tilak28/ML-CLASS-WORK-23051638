import numpy as np

def multiple_linear_regression(X, y):
    """
    Computes regression coefficients using the Normal Equation:
    beta = (X^T X)^(-1) X^T y
    
    X: Feature matrix (m × n)
    y: Output vector (m × 1)
    """
    

    ones = np.ones((X.shape[0], 1))
    X_b = np.hstack((ones, X))

    beta = np.linalg.inv(X_b.T @ X_b) @ (X_b.T @ y)
    return beta



X = np.array([
    [3, 8],
    [4, 5],
    [5, 7],
    [6, 3],
    [2, 1]
])

y = np.array([-3.7, 3.5, 2.5, 11.5, 5.7])

beta = multiple_linear_regression(X, y)
print("Regression coefficients (including intercept):")
print(beta)
