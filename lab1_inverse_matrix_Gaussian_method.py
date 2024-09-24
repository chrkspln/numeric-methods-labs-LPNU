import numpy as np
from copy import deepcopy

k = 26            # 26 is the position in the group list  
p = 22            # 22 is the group number                
s = 0.02 * k      # constant expression for s             
b = 0.02 * p      # constant expression for b  

A = np.array([
    [8.3, 2.62 + s, 4.1, 1.9],
    [3.92, 8.45, 7.78 - s, 2.46],
    [3.77, 7.21 + s, 8.04, 2.28],
    [2.21, 3.65 - s, 1.69, 6.69]
])


def find_max_in_col(mat_p, i, n):
    # Find the maximum element in the i-th column of the matrix
    # and return the index of the row where the maximum element is located
    # so that the row with the maximum element can be swapped with the current row
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
        # The function find_max_in_col looks for the largest value in the current column starting from row i.
        # This is done to swap rows later if needed, improving numerical stability.
        # Larger numbers on the diagonal make the calculations more reliable.
        max_el_idx = find_max_in_col(matrix_V, i, n)
        for h in range(n):
            # If the largest element is not already in the current row (i), this part swaps the rows
            # so that the largest number moves to the diagonal (the "pivot" position).
            # It does the swap in both matrix_V (the working matrix) and inverse matrix.
            matrix_V[h][i], matrix_V[h][max_el_idx] = matrix_V[h][max_el_idx], matrix_V[h][i]
            inverse[h][i], inverse[h][max_el_idx] = inverse[h][max_el_idx], inverse[h][i]

        factor = matrix_V[i][i]

        for h in range(n):
            # The largest value is on the diagonal (position matrix_V[i][i]), so it divides the entire row
            # by that diagonal value. This makes the diagonal value equal to 1.
            matrix_V[i][h] /= factor
            inverse[i][h] /= factor

        for g in range(n):
            # Now that the diagonal value is 1, the function eliminates all the values above and below it
            # in the current column (so all the other values in that column become zero).
            # It does this by subtracting multiples of the current row from the other rows.
            # This step is like "cleaning up" the column, so only the diagonal element remains 1 and the rest become 0s.
            # (in matrix_V only, inverse is becoming the inverse matrix of matrix_V as it originally was the identity matrix)
            if i != g:
                # This factor represents how much of the current row i you need to subtract from row g
                # to "zero out" the value in column i of row g.
                factor = matrix_V[g][i]
                for m in range(n):
                    matrix_V[g][m] -= factor * matrix_V[i][m]
                    inverse[g][m] -= factor * inverse[i][m]

    print("\nMatrix A:\n", A)
    inverse = np.array(inverse)
    print("\nMatrix_V:\n", matrix_V)
    print("\nInverse:\n", inverse)
    identity_check = np.dot(A, inverse)
    print("\nA * A^-1 =\n", identity_check)


if __name__ == "__main__":
    rotated_matrix(A)
