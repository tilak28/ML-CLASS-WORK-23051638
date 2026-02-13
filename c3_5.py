import numpy as np

# Function: f(w) = w^2
def f_grad(w):
    return 2 * w

w = 10
alpha = 0.1
beta = 0.9
v = 0

for i in range(50):
    grad = f_grad(w)

    v = beta*v + (1 - beta)*grad
    w = w - alpha*v

    print(f"Iter {i+1}: w={w:.4f}")
