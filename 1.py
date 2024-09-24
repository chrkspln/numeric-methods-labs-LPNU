import numpy as np
from copy import deepcopy

k = 26             
p = 22             
s = 0.02 * k       
b = 0.02 * p       

A = np.array([
    [8.3, 2.62 + s, 4.1, 1.9],
    [3.92, 8.45, 7.78 - s, 2.46],
    [3.77, 7.21 + s, 8.04, 2.28],
    [2.21, 3.65 - s, 1.69, 6.69]
])


def find_max_in_col(mat_p, i, n):
    max_val = abs(mat_p[i][i])
    max_idx = i
    for m in range(i + 1, n):
        if abs(mat_p[m][i]) > max_val:
            max_val = abs(mat_p[m][i])
            max_idx = m
    return max_idx


def e_matrix(length):
    matrix = [[0 for _ in range(length)] for _ in range(length)]
    for i in range(length):
        for j in range(length):
            if i == j:
                matrix[i][j] = 1
    return matrix


def rotated_matrix(matrix):
    n = len(matrix)
    matrix_V = deepcopy(matrix)
    inverse = e_matrix(n)

    for i in range(n):
        max_el_idx = find_max_in_col(matrix_V, i, n)
        for h in range(n):
            matrix_V[h][i], matrix_V[h][max_el_idx] = matrix_V[h][max_el_idx], matrix_V[h][i]
            inverse[h][i], inverse[h][max_el_idx] = inverse[h][max_el_idx], inverse[h][i]

        factor = matrix_V[i][i]

        for h in range(n):
            matrix_V[i][h] /= factor
            inverse[i][h] /= factor

        for g in range(n):
            if i != g:
                factor = matrix_V[g][i]
                for m in range(n):
                    matrix_V[g][m] -= factor * matrix_V[i][m]
                    inverse[g][m] -= factor * inverse[i][m]


if __name__ == "__main__":
    rotated_matrix(A)
