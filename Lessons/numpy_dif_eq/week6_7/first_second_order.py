import numpy as np
import time
from linear_solver import gaussian_upper_triangle, back_substitution, jacobi
import matplotlib.pyplot as plt
from math import gamma

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


def second_order_midpt(a, b, N, rhs, bc, methods='jacobi'):
    delta = (b - a) / (N - 1)
    A = np.zeros((N, N))
    b = np.zeros(N)
    x = np.zeros(N)
    for i in range(N):
        A[i][i] = -2
        try:
            A[i][i + 1] = 1
        except Exception as e:
            pass
        if i > 0:
            A[i][i - 1] = 1
        #b[i] = delta * delta * rhs(i * delta)
        b[i] = delta * delta * rhs(i * delta, 1, 1)  # bessel x, y, yp
        x[i] = a + delta * i

    if methods == 'jacobi':
        begin_time = time.time()
        y = jacobi(A, b, fixed=bc, maxiter=30000)
        end_time = time.time()
        compute_time = (end_time - begin_time) * 1000
    elif methods == 'gaussian_elimination':
        from linear_solver import gaussian_elimination
        begin_time = time.time()
        y = gaussian_elimination(A, b, fixed=bc)
        end_time = time.time()
        compute_time = (end_time - begin_time) * 1000
    return x, y, compute_time


if __name__ == "__main__":
    # first order
    """
    N = []
    t = []
    rhs = lambda x: x * x
    a = 0.0
    b = 2.0

    x = [a + (b - a) / 99 * i for i in range(100)]
    y = [(xx ** 3 - 2.0) / 3.0 for xx in x]

    plt_list = [3, 10, 20, 50, 100, 500]

    xx = [np.zeros(10) for i in range(len(plt_list))]
    yy = [np.zeros(10) for i in range(len(plt_list))]
    tt = [0 for i in range(len(plt_list))]

    count = 0
    for i in plt_list:
        bc = {i - 1: 2.0}
        xx[count], yy[count], tt[count] = first_order_2pt(a, b, i, rhs, bc)
        plt.plot(xx[count], yy[count], label=f"N = {i}")
        N.append(i)
        t.append(tt[count])
        count += 1
        print(f"N = {N} is done :D")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(x, y, label='y(x)')
    plt.legend()
    plt.show()

    plt.xlabel('N')
    plt.ylabel('Time (ms)')
    plt.yscale('log')
    plt.plot(N, t)
    plt.show()
    """


    # second order
    # y'' = ...
    rhs = lambda x: x
    bessel = lambda x, y, yp: -1/x * yp - y  # a = 0
    # x0 domain
    a = 0.0
    bes0 = 0
    # xf domain
    b = 2
    besf = 20
    # iterations
    N = 100
    besN = 1000
    # boundary
    bc = {N-1: -1.0, 0: 1.0}
    besbc = {N-1: 0, 0: 0}

    x_val, y_val, t = second_order_midpt(bes0, besf, besN, bessel, besbc, methods='jacobi')



    #x_val, y_val, t = second_order_midpt(a, b, N, rhs, bc, methods='jacobi')
    plt.scatter(x_val[::10], y_val[::10], color='red')
    # known solution
    #f = [x**3/6 - 5*x/3 + 1 for x in x_val]
    m_l = [0, 1, 2, 3]
    J = []
    for m in m_l:
        Jb = [((-1)**m *(x/2)**(2*m))/(np.factorial(m)*gamma(m+1)) for x in x_val]
        J.append(Jb)
    plt.plot(x_val, J)
    #plt.plot(x_val, f)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid()
    plt.show()

