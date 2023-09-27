import numpy as np
import matplotlib.pyplot as plt
import time


def gaussian_upper_triangle(a, b):
    def exchange_rows(a, i, j):
        for k in range(len(a)):
            tmp = a[i][k]
            a[i][k] = a[j][k]
            a[j][k] = tmp
        return a

    C = np.column_stack((a, b))

    for i in range(len(C)):
        if C[i][i] == 0:
            if i < len(C) - 1:
                exchange_rows(C, i, i + 1)
        else:
            diag = C[i][i]
            for k in range(i + 1, len(C)):
                C[k] = C[k] - C[i] * C[k][i]/diag# * diag - C[i] * C[k][i]
    return C


def back_substitution(A, fixed=None):
    N = len(A)
    x = np.zeros(N)

    x[N - 1] = A[N - 1][N] / A[N - 1][N - 1] # invailid value encounder in scalar divide

    if fixed != None:
        for index in fixed:
            if index == N - 1:
                x[N - 1] = fixed[index]

    for i in range(N - 2, -1, -1):
        s = 0
        for j in range(i + 1, N):
            s += A[i][j] * x[j]
        s = A[i][N] - s
        s /= A[i][i] #
        x[i] = s

        if fixed != None:
            for index in fixed:
                if index == i:
                    x[i] = fixed[index]
    return x


def gaussian_elimination(A, b, fixed=None):
    C = gaussian_upper_triangle(A, b)
    return back_substitution(C, fixed)


def jacobi(A, b, epsilon=1e-8, maxiter=5000, omega=1, fixed=None, debug=False):
    # matrix, vector
    # avoid if inside for loops
    diag = np.diag(np.diag(A))  # diag twice?
    LU = A - diag
    x = np.zeros(len(b))
    diag_inverse = np.diag(1 / np.diag(diag))
    if fixed == None:
        for i in range(maxiter):
            x_new = omega * np.dot(diag_inverse, b - np.dot(LU, x)) + (1 - omega) * x
            res = np.linalg.norm(x_new - x)
            if debug:  # remove later
                print(f'residue = {res}')
            if res < epsilon:
                return x_new
            x = x_new.copy()
        raise Exception(f'Jacobi did not converge in {maxiter} steps')
        # return x
    else:
        for i in range(maxiter):
            for index in fixed:
                x[index] = fixed[index]
            x_new = omega * np.dot(diag_inverse, b - np.dot(LU, x)) + (1 - omega) * x
            for index in fixed:
                x_new[index] = fixed[index]
            res = np.linalg.norm(x_new - x)
            if debug:
                print(f'residue = {res}')
            if res < epsilon:
                return x_new
            x = x_new.copy()
        raise Exception(f'Jacobi did not converge in {maxiter} steps')
        # return x


if __name__ == "__main__":
    a1 = np.array([[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]])
    a2 = np.array([[2.0, 1.0, -1.0], [-3.0, -2.0, 2.0], [-0.5, 1.0, 2.0]])
    b = np.array([8, -11, -3])
    x = gaussian_elimination(a2, b)
    test = np.linalg.solve(a2, b)
    print(x, test)
    print(a2.dot(x), b)
    try:
        x_j = jacobi(a2, b)
        print(x_j)
    except Exception as e:
        print(e)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    x = gaussian_elimination(a1, b)
    test = np.linalg.solve(a1, b)
    print(x, test)
    print(a1.dot(x), b)
    try:
        x_j = jacobi(a1, b)
        print(x_j)
    except Exception as e:
        print(e)
    try:
        x_j = jacobi(a1, b, omega=0.5)
        print(x_j)
    except Exception as e:
        print(e)