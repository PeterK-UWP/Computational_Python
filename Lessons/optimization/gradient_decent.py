import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt
from matplotlib import cm

def gradient_descent(x, V, dV, param, delta=1e-20, epsilon=1e-20):
    """
    Crude implementation
    :param x: x values
    :param V: potential
    :param dV: change in potential
    :param param:
    :param delta:
    :param epsilon:
    :return:
    """

    x_old = np.zeros_like(x)
    dV_old = np.zeros_like(x)
    dV_current = dV(x, param)
    ddV = dV_current - dV_old
    dx = x - x_old
    f_norm = np.inner(ddV, ddV) # inner product
    x_norm = lin.norm(dx)

    step_count = 0
    while x_norm > delta and f_norm > epsilon*epsilon:
        gamma = np.abs(np.inner(dx, ddV)) / f_norm
        x_old = np.array(x)
        dV_old = np.array(dV_current)
        x = x - gamma*dV_current

        dV_current = dV(x, param)

        ddV = dV_current - dV_old
        dx = x - x_old
        f_norm = np.inner(ddV, ddV)
        x_norm = lin.norm(dx)
        step_count += 1
        print(f'{step_count}: {V(x, param)}')
    return x


def V(x, param):
    # (a0*x0 - a1)**2  + (a2*x1 - a3)**3
    return (param[0]*x[0] - param[1])**2 + (param[2]*x[1] - param[3])**2

def dV(x, param):
    x1 = 2*param[0] * (param[0] * x[0] - param[1])
    y1 = 2*param[2] * (param[2] * x[1] - param[3])
    return np.array([x1, y1])




if __name__ == '__main__':
    param = [4.0, 8.0, 3.0, 1.0]
    #param = [20.0, 0.0, 2.0, 6.5]
    x = np.array([1, 1])
    x = gradient_descent(x, V, dV, param)
    print(x)

    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    XX = [[xx for yy in Y] for xx in X]
    XX = [xx for yy in XX for xx in yy]
    YY = [[yy for yy in Y] for xx in X]
    YY = [xx for yy in YY for xx in yy]
    ZZ = [[V((xx, yy), param) for xx in Y] for yy in X]
    ZZ1 = [zz for z in ZZ for zz in z]

    fig = plt.figure(figsize=(8, 4))
    ax = plt.axes(projection="3d")
    ax.view_init(10, -60)
    surface = ax.plot_trisurf(YY, XX, ZZ1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    xp, yp, zp = x[0], x[1], -50
    ax.scatter(xp, yp, zp)
    ax.contourf(X, Y, ZZ, offset=-50.0)
    ax.set_zlim([-50, 1000])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()



