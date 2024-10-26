import math
import matplotlib.pyplot as plt
import numpy as np


# Вхідні дані
U_max = 100
f = 50
R1 = 5
R2 = 4
R3 = 7
R4 = 2
L1 = 0.01
L2 = 0.02
L3 = 0.015
C1 = 300 * pow(10, -6)
C2 = 150 * pow(10, -6)
C3 = 200 * pow(10, -6)
t0 = 0
tn = 0.2
x0 = np.array([0, 0, 0])
h = pow(10, -5)
amplifier = 10e5


def f_system(t, x):
    U1 = U_max * math.sin(2 * math.pi * f * t)
    dx0dt = (U1 - x[0] - x[2] * (R2 - R3) - x[1]) / R1 * C1
    dx1dt = (U1 - x[0] - x[2] * (R2 - R3) - x[1] - x[2] * R1) / R1 * C1
    dx2dt = (x[2] * (R2 - R3) + x[1]) / L1
    return np.array([dx0dt, dx1dt, dx2dt])

def eulers_method(f_system, x0, t0, tn, h):
    """
    Метод Ейлера для розв'язання системи диференціальних рівнянь
    :param f_system: Функція f(t, x), яка визначає систему
    :param x0: Початкове значення системи розв'язків (000)
    :param t0: Початковий час
    :param tn: Кінцевий час
    :param h: Крок інтегрування
    :return: Масиви значень часу та значень розв'язку [t, tn] та
    матриця значень розв'язку на кожному кроці
    """
    # Кількість підінтервалів
    n = math.ceil((tn - t0) / h)
    # Ініціалізація розв'язку
    t_values = np.linspace(t0, tn, n + 1)
    x_values = np.zeros((len(t_values), len(x0)))

    # Цикл по часу
    for i in range(n):
        x = x_values[i]
        t = t_values[i]
        # Метод Ейлера
        x_next = x + h * f_system(t, x)
        x_values[i + 1] = x_next
    return t_values, x_values


t_values, x_values = eulers_method(f_system, x0, t0, tn, h)
u1_values = [U_max * math.sin(2 * math.pi * f * t) for t in t_values]
u2_values = [amplifier * x[2] * (R2 - R3) + x[2] * R1 for x, t in zip(x_values, t_values)]

for u1, u2, t in zip(u1_values, u2_values, t_values):
    print(f't = {t:.6f}, u1 = {u1}, u2 = {u2}')

plt.figure(figsize=(10, 6))
plt.plot(t_values, u1_values, label='U1')
plt.plot(t_values, u2_values, label='U2')

plt.xlabel('time (s)')
plt.ylabel('values')
plt.legend()
plt.title('System of Differential Equations')
plt.grid(True)
plt.show()
