from quantum_oscillator import Quantum_1D
import numpy as np
import matplotlib.pyplot as plt
import time
import numpy.linalg as lin
from matplotlib import cm

# tensor flow from google

# Finite
Inf = 1e10
def finite_potential(x):
    if x >= 0 and x < 1.0:
        return 0.0
    return Inf


a = 1.5
no_pts = 500
value, vector = Quantum_1D(-a, a, no_pts, finite_potential)

NN = [i for i in range(20)]
theory = [i**2*np.pi**2 for i in NN]
numerical = [value[i] for i in NN]

fig = plt.figure(figsize=(3, 3), dpi=300)
plt.plot(NN, theory, marker='o', label='Theoretical')
plt.plot(NN, numerical, marker='+', label='Numerical')
plt.xticks(np.arange(0, len(NN), step=2))
plt.xlabel('n')
plt.ylabel(r'Energy $\left(\frac{\hbar^2}{2mL^2}\right)$')
plt.legend()
plt.gcf().set_dpi(600)
plt.show()

x = np.linspace(-a, a, no_pts)
psi = lambda x, n: np.sqrt(2) * np.sin(np.pi*x*n)

theory0 = [-psi(xx, 1) if xx >= 0 and xx < 1.0 else 0 for xx in x]
theory1 = [psi(xx, 2) if xx >= 0 and xx < 1.0 else 0 for xx in x]
theory2 = [psi(xx, 3) if xx >= 0 and xx < 1.0 else 0 for xx in x]
theory3 = [psi(xx, 4) if xx >= 0 and xx < 1.0 else 0 for xx in x]

fig = plt.figure(figsize=(8, 9), dpi=300)
gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.28)
(ax0, ax1), (ax2, ax3) = gs.subplots()

ax0.set_xlim([-2, 2])
ax0.plot(x, vector[0], label='Numerical')
ax0.plot(x, theory0, label='Theoretical')
ax0.axvline(x=0, color='grey', linewidth=0.5)
ax0.axvline(x=1, color='grey', linewidth=0.5)
ax0.set_xlabel(r'x $\left(L\right)$')
ax0.set_ylabel(r'$\psi_0$ $\left(\frac{1}{\sqrt{L}}\right)$')
ax0.legend()

ax1.set_xlim([-2, 2])
ax1.plot(x, vector[1], label='Numerical')
ax1.plot(x, theory1, label='Theoretical')
ax1.axvline(x=0, color='grey', linewidth=0.5)
ax1.axvline(x=1, color='grey', linewidth=0.5)
ax1.set_xlabel(r'x $\left(L\right)$')
ax1.set_ylabel(r'$\psi_1$ $\left(\frac{1}{\sqrt{L}}\right)$')
ax1.legend()

ax2.set_xlim([-2, 2])
ax2.plot(x, vector[2], label='Numerical')
ax2.plot(x, theory2, label='Theoretical')
ax2.axvline(x=0, color='grey', linewidth=0.5)
ax2.axvline(x=1, color='grey', linewidth=0.5)
ax2.set_xlabel(r'x $\left(L\right)$')
ax2.set_ylabel(r'$\psi_2$ $\left(\frac{1}{\sqrt{L}}\right)$')
ax2.legend()

ax3.set_xlim([-2, 2])
ax3.plot(x, vector[3], label='Numerical')
ax3.plot(x, theory3, label='Theoretical')
ax3.axvline(x=0, color='grey', linewidth=0.5)
ax3.axvline(x=1, color='grey', linewidth=0.5)
ax3.set_xlabel(r'x $\left(L\right)$')
ax3.set_ylabel(r'$\psi_3$ $\left(\frac{1}{\sqrt{L}}\right)$')
ax3.legend()

plt.show()
