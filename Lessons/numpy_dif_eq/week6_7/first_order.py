import numpy as np

def first_order_2pt(a, b, N, rhs, bc):
    delta = (b - a) / (N - 1)
    A = np.zeros((N, N))
    b = np.zeros(N)
    x = np.zeros(N)

    for i in range(N):
        A[i][i] = -1
        try:
            A[i][i + 1] = 1
        except Exception as e:
            pass
        b[i] = delta * rhs(i * delta)
        x[i] = a + delta * i
    indices = list(bc.keys())
    # Note: first order equation has only one BC
    index = indices[0]
    bc_val = bc[index]
    A[index][index] = b[index] / bc_val
    try:
        A[index][index + 1] = 0
    except Exception as e:
        pass
    begin_time = time.time()
    C = gaussian_upper_triangle(A, b)
    y = back_substitution(C)
    end_time = time.time()
    compute_time = (end_time - begin_time) * 1000

    return x, y, compute_time