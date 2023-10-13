from general_linear_second_order import general_linear_second_ordered
import numpy as np
import matplotlib.pyplot as plt


def Euler_Cauchy(a, b):
    no_pts = 100
    bounds = {no_pts - 1: 1.0, 0: 1e-9}
    staring_bound = 1e-9
    ending_bound = 1

    coef1 = lambda x: x**2
    coef2 = lambda x: -a*x
    coef3 = lambda x: b
    non_homo_func = lambda x: 0.0

    x, y, t = general_linear_second_ordered(staring_bound, ending_bound,
                                            no_pts, coef1, coef2, coef3, non_homo_func, bounds)
    return x, y



def Legendre(n_value):
    # (1-x^2)*ypp - 2x*yp + n(n+1)y = 0
    no_pts = 100
    bounds = {no_pts-1:1.0} # {99: 1.0}
    if n_value%2 == 0: # if the remainder is 0
        bounds.update({0:1.0})
    else:
        bounds.update({0:-1.0})

    starting_bounds = -1
    ending_bounds = 1

    coef1 = lambda x: 1 - x**2
    coef2 = lambda x: -2*x
    coef3 = lambda x: n_value*(n_value+1)
    non_homo_func = lambda x: 0.0

    x, y, t = general_linear_second_ordered(starting_bounds, ending_bounds,
                                            no_pts, coef1, coef2, coef3, non_homo_func, bounds)
    return x, y

def Bessel(a_value):
    # x^2*ypp + x*yp + (x^2 - a^2)y = 0
    no_pts = 100
    bounds = {no_pts - 1: 20.0}  # {99: 1.0}
    #if a_value % 2 == 0:  # if the remainder is 0
    #    bounds.update({0: 1.0})
    #else:
    #    bounds.update({0: -1.0})

    starting_bounds = 0
    ending_bounds = 20

    coef1 = lambda x: x ** 2
    coef2 = lambda x: x
    coef3 = lambda x: (x ** 2 - a_value ** 2)
    non_homo_func = lambda x: 0.0

    x, y, t = general_linear_second_ordered(starting_bounds, ending_bounds,
                                            no_pts, coef1, coef2, coef3, non_homo_func, bounds)
    return x, y


def Laguerre(n_value, a_value=0):
    # x*ypp + (a + 1 - x)*yp + n*y = 0
    no_pts = 100
    bounds = {no_pts - 1: 10.0}  # {99: 1.0}

    starting_bounds = -1
    ending_bounds = 10

    coef1 = lambda x: x
    coef2 = lambda x: (a_value + 1 - x)
    coef3 = lambda x: n_value
    non_homo_func = lambda x: 0.0

    x, y, t = general_linear_second_ordered(starting_bounds, ending_bounds,
                                            no_pts, coef1, coef2, coef3, non_homo_func, bounds)
    return x, y


def Hermite(l): # WIP
    # xypp - 2x*yp + 2l*y = 0
    no_pts = 100
    bounds = {no_pts - 1: 10.0}  # {99: 1.0}

    starting_bounds = -5
    ending_bounds = 5

    coef1 = lambda x: 1
    coef2 = lambda x: 2*x
    coef3 = lambda x: l
    non_homo_func = lambda x: 0.0

    x, y, t = general_linear_second_ordered(starting_bounds, ending_bounds,
                                            no_pts, coef1, coef2, coef3, non_homo_func, bounds)
    return x, y


def AssociatedLegendre(n, m):
    #import scipy
    #scipy.special.lpmv(m, n)
    epsillon = 1e-9
    pts = 100
    bc = {pts - 1: 1.0}
    if n % 2 == 0:
        bc.update({0: 1.0})
    else:
        bc.update({0: -1.0})

    starting_bound = -1.0 #+ epsillon
    ending_bound = 1.0 #- epsillon

    px = lambda x: 1 - x ** 2
    qx = lambda x: -2 * x * (m+1)
    rx = lambda x: (n*(n + 1)) - (m*(m + 1))
    sx = lambda x: 0
    x_values, y_values, t = general_linear_second_ordered(starting_bound, ending_bound, pts, px, qx, rx, sx, bc)
    return x_values, y_values

# results P1, P2, P3, ... n = 1, 2, 3
if __name__ == '__main__':
    """
    # Euler_Cauchy Polynomials
    e_list = []
    values = [[-1, -.1], [-2, .25], [-1, .1]]
    ec_list = []
    for i in values:
        x, ec = Euler_Cauchy(i[0], i[1])
        ec_list.append(ec)
    for i in ec_list:
        plt.plot(x, i)
    plt.show()"""

    # Legendre Polynomials
    p_list = []
    for i in np.linspace(1, 4, 4):
        #x, p = Legendre(i)
        x, p = AssociatedLegendre(2, 1)
        p_list.append(p)
    for p in p_list:
        plt.plot(x, p)
    plt.show()
"""
    # Bessel Polynomials
    j_list = []
    for i in np.linspace(0, 3, 3):
        x, j = Bessel(i)
        j_list.append(j)
    for j in j_list:
        plt.plot(x, j)
    plt.show()

    # Laguerre Polynomials
    g_list = []
    for i in np.linspace(0, 5, 5):
        x, g = Laguerre(i)
        g_list.append(g)
    for g in g_list:
        plt.plot(x, g)
    plt.show()

    # Hermite Polynomials

"""