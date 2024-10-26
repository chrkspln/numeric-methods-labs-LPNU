import numpy as np
import sympy as sp
from matplotlib import pyplot as plt

# Кількість підінтервалів
n = 30
# Ширина кожного підінтервалу
h = 1 / n

x_values = np.linspace(0, 1, n + 1)
x = sp.symbols('x')

def integral_f(x):
    return x / sp.sqrt(0.5 + 2*x)

def F(x):
    return - (2 - 4*x) / 12 * sp.sqrt(0.5 + 2*x)

def midpoint(f, h, n):
    """
    Метод середніх прямокутників для наближеного обчислення інтегралів
    :param f: Функція, яку потрібно інтегрувати
    :param h: Ширина кожного підінтервалу
    :param n: Кількість підінтервалів
    :return: Наближене значення інтегралу
    """
    I_approx = 0

    for i in range(1, n + 1):
        """
        Формула для методу середніх прямокутників:
        h * (f(x_1_star) + f(x_2_star) + ... + f(x_n_star))
        = h * сума(f(x_i_star), i = 1, 2, ..., n)

        Алгоритм:
        1. Для i = 1, 2, ..., n:
            a. Обчислюємо середню точку x_i_star = (2 * i - 1) / (2 * n)
            b. Обчислюємо значення функції в цій точці f(x_i_star)
            c. Оновлюємо наближення для інтегралу I_approx += f(x_i_star)
        2. Множимо кінцеве значення на ширину h
        """
        # Середня точка
        x_i_star = (2 * i - 1) / (2 * n)
        # Значення функції в середній точці
        f_x_i_star = f(x_i_star)
        I_approx += f_x_i_star
    I_approx *= h

    return I_approx

# Обчислюємо похідну
F_prime = sp.diff(integral_f(x), x)
F_prime_simplified = sp.simplify(F_prime)

# Обчислюємо F(1) та F(0)
F_1 = (2 / 12) * sp.sqrt(2.5)  # F(1)
F_0 = - (1 / 6) * sp.sqrt(0.5)  # F(0)

f_vals = [integral_f(x_val) for x_val in x_values]
F_vals = [F(x_val).evalf() for x_val in x_values]

I_approx = midpoint(integral_f, h, n)
# Обчислюємо точне значення I
I_exact = F_1 - F_0
I_exact.evalf()  

print(I_approx)
print(I_exact)

# Побудова графіків
plt.figure(figsize=(10, 6))
plt.plot(x_values, f_vals, label='f(x) = x / sqrt(0.5 + 2x)')
plt.plot(x_values, F_vals, label='F(x) = -(2 - 4x) / 12 * sqrt(0.5 + 2x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Графіки f(x) і g(x)')
plt.legend()
plt.grid(True)
plt.show()
