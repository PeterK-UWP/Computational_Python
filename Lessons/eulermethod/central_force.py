# keplers problem:


# 2 body: m1, m2
# assume to be gravitational (coulumbs is similar)

import numpy as np
import evolution as ev
import matplotlib.pyplot as plt

G_SI = 6.674E-11 # universal gravitational constant
G = 1.9935E-44 # AU^3/Kg/s^2

M_sun = 1.989E30 # Kg
AU = 1.496E11 # m
Year = 31557600 # s

# scale is a and is derived from rp
M_earth = 5.97E24 # Kg
rp_earth = 1.471E11 # m: perihelion distance
e_earth = 0.017 # eccentricity
M_jupiter = 1.898E27 # Kg
rp_jupiter = 7.405E11 # m
e_jupiter = 0.049
M_halley = 2.2E14 # Kg
rp_halley = 0.59278*AU # m
e_halley = 0.96658


def orbit(mass1, mass2, perihelion, eccentricity, no_pts, dt):
    def zeta(xi, t=None, param=None):
        radius = (xi[0] * xi[0] + xi[1] * xi[1])**1.5
        return np.array([xi[2], xi[3], -4*np.pi**2*xi[0]/radius, -4*np.pi**2*xi[1]/radius])

    a = perihelion/(1-eccentricity)
    chi = a

    total_mass = mass1 + mass2
    tau = np.sqrt(
        (4*np.pi**2*a**3)/(G_SI*total_mass)
    )
    velocity = np.sqrt(
        (1+eccentricity)*G_SI*mass1/perihelion
    )
    t0 = 0.0

    scale = np.array([tau, chi, chi, chi/tau, chi/tau]) # 2D
    initial_data_array = np.array([perihelion, 0, 0, velocity])
    time = np.arange(0, no_pts * dt, dt)
    trajectory = ev.evolve(initial_data_array, t0, dt, no_pts, zeta, ev.rk4_step, scale) # implament rk4 in evolution.py

    complete_traj = []
    for function in trajectory:
        path = [mass2/total_mass*function[0]/AU, mass2/total_mass*function[1]/AU,
                -mass1/total_mass*function[0]/AU, -mass1/total_mass*function[1]/AU,
                mass2/total_mass*function[2], mass2/total_mass*function[3],
                -mass1/total_mass*function[2], -mass1/total_mass*function[3]]
        try:
            complete_traj.append(path)
        except NameError:
            complete_traj = [path]
    return time, complete_traj


if __name__ == "__main__":
    dt = 0.01
    dt = dt*Year
    t_earth, traj_earth = orbit(M_sun, M_earth, rp_earth, e_earth, 200, dt)
    x_earth = [frame[2] for frame in traj_earth]
    y_earth = [frame[3] for frame in traj_earth]
    t_jupiter, traj_jupiter = orbit(M_sun, M_jupiter, rp_jupiter, e_jupiter, 1300, dt)
    x_jupiter = [frame[2] for frame in traj_jupiter]
    y_jupiter = [frame[3] for frame in traj_jupiter]
    t_halley, traj_halley = orbit(M_sun, M_halley, rp_halley, e_halley, 7500, dt)
    x_halley = [frame[2] for frame in traj_halley]
    y_halley = [frame[3] for frame in traj_halley]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal')
    ax.plot(x_earth, y_earth, label='Earth')
    ax.plot(x_jupiter, y_jupiter, label='Jupiter')
    ax.plot(x_halley, y_halley, label='Halley\'s comet')
    ax.legend()
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.show()

