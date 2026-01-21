# Gradient Descent for f(x) = x^2

def gradient_descent(alpha=0.1, iterations=50):
    x = 10  # starting value

    for i in range(iterations):
        grad = 2 * x         # derivative of x^2
        x = x - alpha * grad

        print(f"Iter {i+1}: x = {x:.4f}, f(x) = {x**2:.4f}")

gradient_descent()
