#1D quantum Oscillator

import numpy as np
import numpy.linalg as lin
import time
import matplotlib.pyplot as plt
from matplotlib import cm

# perturbation theory, alpha does not depend on x, it just has
# to intersect HO, with more boxes we get better approximation
def Quantum_1D(xmin, xmax, no_pts, potential, param=None):
    domain = (xmax - xmin) / no_pts
    d_squared = domain*domain
    matrix = np.zeros((no_pts, no_pts))
    i = 0
    x = xmin + i*domain
    if param == None:
        matrix[i][i] = 2.0 / d_squared + potential(x)
    else:
        matrix[i][i] = 2.0/d_squared + potential(x, param)
    matrix[i][i+1] = -1.0 / domain**2

    for i in range(1, no_pts - 1):
        x = xmin + i * domain
        matrix[i][i-1] = -1.0 / d_squared
        if param == None:
            matrix[i][i] = 2.0 / d_squared + potential(x)
        else:
            matrix[i][i] = 2.0 / d_squared + potential(x, param)

        matrix[i][i+1] = -1.0 / domain**2

    i = no_pts - 1
    x = xmin + i * domain
    if param == None:
        matrix[i][i] = 2.0 / d_squared + potential(x)
    else:
        matrix[i][i] = 2.0 / d_squared + potential(x, param)

    matrix[i][i - 1] = -1.0 / d_squared

    #eigen
    value, vector = lin.eig(matrix)
    indices = np.argsort(value)

    energy = []
    states = []
    for index in indices:
        try:
            energy.append(value[index])
        except NameError:
            energy = [value[index]]

        try:
            states.append(vector[:, index] / np.sqrt(domain))
        except NameError:
            states = [vector[:, index] / np.sqrt(domain)]
    return energy, states
if __name__ == '__main__':
    x0 = 5
    harmonic = lambda x: x ** 2
    shift = lambda x: (x - x0) ** 2
    absolute = lambda x: a/2 * np.abs(x)
    a = 10
    no_pts = 100
    # eigen
    value, vector = Quantum_1D(-a, a, no_pts, absolute)

    NN = [i for i in range(20)]
    theory = [(2 * i + 1) for i in NN]
    numerical = [value[i] for i in NN]

    fig = plt.figure(figsize=(3, 3), dpi=300)
    plt.plot(NN, theory, marker='o', label='Theoretical')
    plt.plot(NN, numerical, marker='+', label='Numerical')
    plt.xticks(np.arange(0, len(NN), step=2))
    plt.xlabel('n')
    plt.ylabel(r'Energy ($\frac{1}{2}\hbar\omega$)')
    plt.legend()
    plt.gcf().set_dpi(600)
    plt.show()

    x = np.linspace(-a, a, no_pts)
    theory0 = [(np.pi) ** -0.25 * np.exp(-xx * xx / 2) for xx in x]
    theory1 = [(np.pi) ** -0.25 * np.exp(-xx * xx / 2) * (-xx) for xx in x]
    theory2 = [(8 * np.pi) ** -0.25 * np.exp(-xx * xx / 2) * (-2 * xx * xx + 1) for xx in x]
    theory3 = [(8 * 6 * np.pi) ** -0.25 * np.exp(-xx * xx / 2) * (4 * xx * xx * xx - 6 * xx) for xx in x]

    fig = plt.figure(figsize=(8, 9), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)
    (ax0, ax1), (ax2, ax3) = gs.subplots()
    ax0.plot(x, vector[0], label='Numerical')
    ax0.plot(x, theory0, label='Theory')
    ax0.set_xlabel(r'x $\left(\sqrt{\frac{\hbar}{m \omega}}\right)$')
    ax0.set_ylabel(r'$\psi_0$')
    ax0.legend()

    ax1.plot(x, vector[1], label='Numerical')
    ax1.plot(x, theory1, label='Theory')
    ax1.set_xlabel(r'x $\left(\sqrt{\frac{\hbar}{m \omega}}\right)$')
    ax1.set_ylabel(r'$\psi_1$')
    ax1.legend()

    ax2.plot(x, vector[2], label='Numerical')
    ax2.plot(x, theory2, label='Theory')
    ax2.set_xlabel(r'x $\left(\sqrt{\frac{\hbar}{m \omega}}\right)$')
    ax2.set_ylabel(r'$\psi_2$')
    ax2.legend()

    ax3.plot(x, vector[3], label='Numerical')
    ax3.plot(x, theory3, label='Theory')
    ax3.set_xlabel(r'x $\left(\sqrt{\frac{\hbar}{m \omega}}\right)$')
    ax3.set_ylabel(r'$\psi_3$')
    ax3.legend()

    plt.show()

    # larger orders
    fig = plt.figure(figsize=(8, 9), dpi=300)
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)
    (ax0, ax1), (ax2, ax3) = gs.subplots()

    ax0.plot(x, vector[10], label='Numerical')
    ax0.set_xlabel(r'x $\left(\sqrt{\frac{\hbar}{m \omega}}\right)$')
    ax0.set_ylabel(r'$\psi_10$')

    ax1.plot(x, vector[11], label='Numerical')
    ax1.set_xlabel(r'x $\left(\sqrt{\frac{\hbar}{m \omega}}\right)$')
    ax1.set_ylabel(r'$\psi_11$')

    ax2.plot(x, vector[12], label='Numerical')
    ax2.set_xlabel(r'x $\left(\sqrt{\frac{\hbar}{m \omega}}\right)$')
    ax2.set_ylabel(r'$\psi_12$')

    ax3.plot(x, vector[13], label='Numerical')
    ax3.set_xlabel(r'x $\left(\sqrt{\frac{\hbar}{m \omega}}\right)$')
    ax3.set_ylabel(r'$\psi_13$')

    plt.show()