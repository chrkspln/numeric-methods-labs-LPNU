import math

# f(x) is the function whose root we are trying to find: f(x) = ln(x) - sqrt(x) / 5
def f(x):
    return math.log(x) - math.sqrt(x) / 5

# f'(x) is the derivative of the function f(x)
# f'(x) = 1/x - 1/(2 * sqrt(5) * sqrt(x))
def f_prime(x):
    return 1 / x - 1 / (2 * math.sqrt(5) * math.sqrt(x))

# Newton-Raphson method to approximate the root of f(x)
# x0: initial guess
# epsilon: tolerance for stopping criterion (small value to control precision)
# max_iter: maximum number of iterations allowed
def newton_method(x0, epsilon=0.00001, max_iter=100):
    # Start with the initial guess for the root
    if 0.1 <= x0 <= 6:
        x = x0
    else:
        print("Початкове наближення x0 не входить в область визначення функції.")
        return None
    for i in range(max_iter):
        # Evaluate the function f(x) and its derivative f'(x) at the current x
        fx = f(x)
        fpx = f_prime(x)

        # Check if the derivative is zero, which would cause division by zero
        if fpx == 0:
            print("Похідна дорівнює нулю. Алгоритм провалився.")
            return None  # Exit the function as we can't proceed

        # Update the estimate of the root using Newton's update formula:
        # x_{k+1} = x_k - f(x_k) / f'(x_k)
        x_new = x - fx / fpx

        # Calculate the relative error between the current and the new estimate
        relative_error = abs((x_new - x) / x_new) * 100

        # Print the current iteration details: new x value and relative error
        print(f"Ітерація {i+1}: x = {x_new:.6f}, відносна похибка = {relative_error:.6f}%")

        # If the relative error is less than the tolerance epsilon, stop the algorithm
        # and return the current estimate of the root
        if relative_error < epsilon:
            return x_new

        # If the stopping criterion isn't met, update x to the new estimate and continue
        x = x_new

    # If the maximum number of iterations is reached and the root is not found
    # within the required tolerance, return None
    print("Не вдалося знайти корінь з необхідною точністю.")
    return None


if __name__ == "__main__":
    # Set an initial guess for the root
    x0 = 1
    print("Початкове наближення: x0 =", x0)
    # Run the Newton-Raphson method to find the root starting from the initial guess
    root = newton_method(x0)

    # If a root was found, print the result and the value of the function at the root
    if root:
        print(f"\nЗнайдений корінь: x = {root:.6f}")
        print(f"Значення функції f(x) в точці {root:.6f}: f(x) = {f(root):.6f}")
    else:
        print("Корінь не знайдений.")
