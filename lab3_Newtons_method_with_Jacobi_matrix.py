import numpy as np


# Описані функції F1 і F2
def F1(x1, x2):
    return 4 * pow(x1, 2) + pow(x2, 2) - 4


def F2(x1, x2):
    return x1 - pow(x2, 2)


# Вектор функцій
def F(x):
    x1, x2 = x
    return np.array([F1(x1, x2), F2(x1, x2)])


# Обчислення матриці Якобі методом кінцевих різниць
def jacobian(x, h=1e-5):
    n = len(x)
    J = np.zeros((n, n))
    F_x = F(x)
    for j in range(n):
        x_forward = np.copy(x)
        x_forward[j] += h
        F_x_forward = F(x_forward)
        J[:, j] = (F_x_forward - F_x) / h
    return J


def gauss_solve(A, b):
    n = len(b)
    # Прямий хід
    for i in range(n):
        if abs(A[i, i]) == 0:
            raise ValueError("Матриця вироджена.")
        # Пошук головного елемента
        max_row = np.argmax(np.abs(A[i:n, i])) + i
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]]
            b[i], b[max_row] = b[max_row], b[i]

        # Прямий хід
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]

    # Зворотній хід
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]

    return x


# Метод Ньютона
def newton_method(x0, tol=1e-6, max_iter=100):
    x_old = np.copy(x0)
    for k in range(max_iter):
        # Обчислюємо значення функцій
        F_val = F(x_old)
        norm_F_val = np.linalg.norm(F_val, ord=2)

        # Перевірка на збіжність
        if norm_F_val < tol:
            print(f"Збіжність досягнута за {k} ітерацій.")
            return x_old

        # Обчислюємо матрицю Якобі
        J = jacobian(x_old)

        # Розв'язуємо систему J * delta_x = -F
        delta_x = gauss_solve(J, -F_val)

        # Оновлюємо значення x
        x_new = x_old + delta_x

        # Перевіряємо умову збіжності для зміни x
        if np.linalg.norm(delta_x, ord=2) < tol:
            print(f"Збіжність досягнута за {k} ітерацій.")
            return x_new

        print(f"Ітерація {k}: x = {x_new}, F(x) = {norm_F_val}")
        # Оновлюємо x
        x_old = x_new

    print("Максимальна кількість ітерацій досягнута.")
    return x_old


# Початкове наближення
x0 = np.array([0.5, 0.5])

# Запуск методу Ньютона
solution = newton_method(x0)

# Виведення результату
print("Розв'язок:", solution)

# Перевірка результату
print("Перевірка F(solution):", F(solution))
