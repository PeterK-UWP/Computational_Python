import numpy as np
from Lessons.eulermethod.evolution import evolve, euler_step
import matplotlib.pyplot as plt

def projectile_2D(x0, y0, v0, theta, g, n, dt):
    def zeta(xi, t=None):
        return np.array([xi[2], xi[3], 0.0, -1.0])


    tau = v0/g
    chi = v0*tau

    init = np.array([
        x0/chi, y0/chi, np.cos(np.radians(theta)), np.sin(np.radians(theta))
    ])

    dt_p = dt/tau
    t = np.zeros(n)
    traj_p = evolve(init, 0.0, dt_p, n, zeta, euler_step)

    traj = traj_p.copy()

    for i in range(n-1):
        t[i] = i*dt
        traj[i][0] = traj_p[i][0]*chi
        traj[i][1] = traj_p[i][1]*chi
        traj[i][2] = traj_p[i][2]*v0
        traj[i][3] = traj_p[i][3]*v0

    return t, traj

t, traj = projectile_2D(0.0, 0.0, 5, 30, 9.8, 520, 0.001)
plt.plot(traj[:, 0], traj[:, 1])
plt.show()

