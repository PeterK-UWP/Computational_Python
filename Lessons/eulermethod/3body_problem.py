import numpy as np
import matplotlib.pyplot as plt
import evolution as ev
G_SI = 6.674E-11 # Universal Gravitation m^3/kg/s^2
G = 1.9935E-44 # AU^3/kg/s^2

C = 4.0*np.pi**2
M_sun = 1.989E30 # kg
AU = 1.496E11 # s
Year = 31557600 # s

M_earth = 5.97E24 # kg
rp_earth = 1.471E11 # m perihelion distence
e_earth = 0.017

M_jupiter = 1.898E27 # kg
rp_jupiter = 7.405E11 # m
e_jupiter = 0.049

M_halley = 2.2E14 # kg
rp_halley = 0.59278*AU # m
e_halley = 0.96658

M_moon = 0.073E24 # kg
rp_moon = 0.363E9 # m
e_moon = 0.055

M_mars = 6.42E23 # kg
rp_mars = 1.524*AU # m
e_mars = 0.0934
def orbit(mass1, mass2, perihelion, eccentricity, no_pts, dt):
    def zeta(xi, t=None, param=None):
        radius = (xi[0] * xi[0] + xi[1] * xi[1]) ** 1.5
        return np.array([xi[2], xi[3], -4 * np.pi ** 2 * xi[0] / radius, -4 * np.pi ** 2 * xi[1] / radius])

    a = perihelion / (1 - eccentricity)
    chi = a

    total_mass = mass1 + mass2
    tau = np.sqrt(
        (4 * np.pi ** 2 * a ** 3) / (G_SI * total_mass)
    )
    velocity = np.sqrt(
        (1 + eccentricity) * G_SI * mass1 / perihelion
    )
    t0 = 0.0

    scale = np.array([tau, chi, chi, chi / tau, chi / tau])  # 2D
    initial_data_array = np.array([perihelion, 0, 0, velocity])
    time = np.arange(0, no_pts * dt, dt)
    trajectory = ev.evolve(initial_data_array, t0, dt, no_pts, zeta, ev.rk4_step,
                           scale)  # implement rk4 in evolution.py

    complete_traj = []
    for function in trajectory:
        path = [mass2 / total_mass * function[0] / AU, mass2 / total_mass * function[1] / AU,
                -mass1 / total_mass * function[0] / AU, -mass1 / total_mass * function[1] / AU,
                mass2 / total_mass * function[2], mass2 / total_mass * function[3],
                -mass1 / total_mass * function[2], -mass1 / total_mass * function[3]]
        try:
            complete_traj.append(path)
        except NameError:
            complete_traj = [path]
    return time, complete_traj


def three_body(mass1, mass2, mass3, perihelion2, perihelion3, eccentricity2, eccentricity3, no_pts, dt):

    def zeta(xi, t, dimless_mass):
        # r12 distance from r2 to r1
        r12 = (
            (xi[2] - xi[0])*(xi[2] - xi[0]) + (xi[3] - xi[1])*(xi[3] - xi[1])
        )**1.5
        # r13 distance from r3 to r1
        r13 = (
            (xi[4] - xi[0]) * (xi[4] - xi[0]) + (xi[5] - xi[1]) * (xi[5] - xi[1])
        )**1.5
        # r23 distance from r2 to r3
        r23 = (
            (xi[4] - xi[2]) * (xi[4] - xi[2]) + (xi[5] - xi[3]) * (xi[5] - xi[3])
        )**1.5
        return np.array(
            [
                xi[6], xi[7], xi[8], xi[9], xi[10], xi[11], C * dimless_mass[0] * (xi[2] - xi[0]) / r12 +
                                                            C * dimless_mass[1] * (xi[4] - xi[0]) / r13,
                                                            C * dimless_mass[0] * (xi[3] - xi[1]) / r12 +
                                                            C * dimless_mass[1] * (xi[5] - xi[1]) / r13,
                                C * (xi[0] - xi[2]) / r12 + C * dimless_mass[1] * (xi[4] - xi[2]) / r23,
                                C * (xi[1] - xi[3]) / r12 + C * dimless_mass[1] * (xi[5] - xi[3]) / r23,
                                C * (xi[0] - xi[4]) / r13 + C * dimless_mass[0] * (xi[2] - xi[4]) / r23,
                                C * (xi[1] - xi[5]) / r13 + C * dimless_mass[0] * (xi[3] - xi[5]) / r23

            ]
        )

    a2 = perihelion2 / (1 - eccentricity2)
    chi = a2
    tau = np.sqrt(4*np.pi**2 * (a2**3) / (G_SI * mass1))

    v2 = np.sqrt((1 + eccentricity2) * G_SI * mass1 / perihelion2)
    v3 = np.sqrt((1 + eccentricity3) * G_SI * mass1 / perihelion3)
    t0 = 0.0

    param = np.array([mass2/mass1, mass3/mass1])
    scale = np.array([tau,
                      chi, chi, chi, chi, chi, chi,
                      chi/tau, chi/tau, chi/tau, chi/tau, chi/tau, chi/tau])

    initial_data_array = np.array([0, 0, perihelion2, 0, perihelion3, 0, 0, 0, 0, v2, 0, v3])

    t = np.arange(0, no_pts*dt, dt)
    trajectory = ev.evolve(initial_data_array, t0, dt, no_pts, zeta, ev.rk4_step_param, scale, param)

    return t, trajectory


def three_body_moon(mass1, mass2, mass3, perihelion2, perihelion3, eccentricity2, eccentricity3, no_pts, dt):
    def zeta(xi, t, dimless_mass):
        # r12 distance from r2 to r1
        r12 = (
            (xi[2] - xi[0])*(xi[2] - xi[0]) + (xi[3] - xi[1])*(xi[3] - xi[1])
        )**1.5
        # r13 distance from r3 to r1
        r13 = (
            (xi[4] - xi[0]) * (xi[4] - xi[0]) + (xi[5] - xi[1]) * (xi[5] - xi[1])
        )**1.5
        # r23 distance from r2 to r3
        r23 = (
            (xi[4] - xi[2]) * (xi[4] - xi[2]) + (xi[5] - xi[3]) * (xi[5] - xi[3])
        )**1.5
        return np.array(
            [
                xi[6], xi[7], xi[8], xi[9], xi[10], xi[11], C * dimless_mass[0] * (xi[2] - xi[0]) / r12 +
                                                            C * dimless_mass[1] * (xi[4] - xi[0]) / r13,
                                                            C * dimless_mass[0] * (xi[3] - xi[1]) / r12 +
                                                            C * dimless_mass[1] * (xi[5] - xi[1]) / r13,
                                C * (xi[0] - xi[2]) / r12 + C * dimless_mass[1] * (xi[4] - xi[2]) / r23,
                                C * (xi[1] - xi[3]) / r12 + C * dimless_mass[1] * (xi[5] - xi[3]) / r23,
                                C * (xi[0] - xi[4]) / r13 + C * dimless_mass[0] * (xi[2] - xi[4]) / r23,
                                C * (xi[1] - xi[5]) / r13 + C * dimless_mass[0] * (xi[3] - xi[5]) / r23

            ]
        )

    a2 = perihelion2 / (1 - eccentricity2)
    chi = a2
    tau = np.sqrt(
        4 * np.pi ** 2 * (a2 ** 3) / (G_SI * mass1)
    )

    v2 = np.sqrt(
        (1 + eccentricity2) * G_SI * mass1 / perihelion2
    )
    v3 = np.sqrt(
        (1 + eccentricity3) * G_SI * mass2 / perihelion3
    )
    print(f'velocity 2: {v2}, velocity 3: {v3}')

    t0 = 0.0

    param = np.array([mass2 / mass1, mass3 / mass1])
    scale = np.array([tau,
                      chi, chi, chi, chi, chi, chi,
                      chi / tau, chi / tau, chi / tau, chi / tau, chi / tau, chi / tau])

    initial_data_array = np.array(
        [0, 0, perihelion2, 0, perihelion2 + perihelion3, 0, 0, 0, 0, v2, 0, v2 + v3]
    )

    t = np.arange(0, no_pts*dt, dt)
    trajectory = ev.evolve(initial_data_array, t0, dt, no_pts, zeta, ev.rk4_step_param, scale, param)

    return t, trajectory


def transfer_orbit(mass1, mass2, mass3, perihelion2, perihelion3, eccentricity2, eccentricity3, no_pts, dt):
    # Rocket from Earths orbit, to Suns Orbit, to Mars' orbit

    def zeta(xi, t, dimless_mass):
        # r12 distance from r2 to r1
        r12 = (
                      (xi[2] - xi[0]) * (xi[2] - xi[0]) + (xi[3] - xi[1]) * (xi[3] - xi[1])
              ) ** 1.5
        # r13 distance from r3 to r1
        r13 = (
                      (xi[4] - xi[0]) * (xi[4] - xi[0]) + (xi[5] - xi[1]) * (xi[5] - xi[1])
              ) ** 1.5
        # r23 distance from r2 to r3
        r23 = (
                      (xi[4] - xi[2]) * (xi[4] - xi[2]) + (xi[5] - xi[3]) * (xi[5] - xi[3])
              ) ** 1.5
        return np.array(
            [
                xi[6], xi[7], xi[8], xi[9], xi[10], xi[11], C * dimless_mass[0] * (xi[2] - xi[0]) / r12 +
                                                            C * dimless_mass[1] * (xi[4] - xi[0]) / r13,
                                                            C * dimless_mass[0] * (xi[3] - xi[1]) / r12 +
                                                            C * dimless_mass[1] * (xi[5] - xi[1]) / r13,
                                                            C * (xi[0] - xi[2]) / r12 + C * dimless_mass[1] * (
                                                                        xi[4] - xi[2]) / r23,
                                                            C * (xi[1] - xi[3]) / r12 + C * dimless_mass[1] * (
                                                                        xi[5] - xi[3]) / r23,
                                                            C * (xi[0] - xi[4]) / r13 + C * dimless_mass[0] * (
                                                                        xi[2] - xi[4]) / r23,
                                                            C * (xi[1] - xi[5]) / r13 + C * dimless_mass[0] * (
                                                                        xi[3] - xi[5]) / r23

            ]
        )

    a2 = perihelion2 / (1 - eccentricity2)
    chi = a2
    tau = np.sqrt(
        4 * np.pi ** 2 * (a2 ** 3) / (G_SI * mass1)
    )

    v2 = np.sqrt(
        (1 + eccentricity2) * G_SI * mass1 / perihelion2
    )
    v3 = np.sqrt(
        (1 + eccentricity3) * G_SI * mass2 / perihelion3
    )
    print(f'velocity 2: {v2}, velocity 3: {v3}')

    t0 = 0.0

    param = np.array([mass2 / mass1, mass3 / mass1])
    scale = np.array([tau,
                      chi, chi, chi, chi, chi, chi,
                      chi / tau, chi / tau, chi / tau, chi / tau, chi / tau, chi / tau])

    initial_data_array = np.array(
        [0, 0, perihelion2, 0, perihelion2 + perihelion3, 0, 0, 0, 0, v2, 0, v2 + v3]
    )

    t = np.arange(0, no_pts * dt, dt)
    trajectory = ev.evolve(initial_data_array, t0, dt, no_pts, zeta, ev.rk4_step_param, scale, param)

    return t, trajectory
if __name__ == '__main__':
    """
    # Central Force, Sun, Earth, Comet
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


    # Sun Earth Jupiter: Earth Jupiter orbit
    dt = 0.0001 # Yr
    dt = dt * Year

    t, traj = three_body(M_sun, M_earth, M_jupiter, rp_earth, rp_jupiter, e_earth, e_jupiter, 120000, dt)

    x_earth = [frame[2]/AU for frame in traj]
    y_earth = [frame[3]/AU for frame in traj]
    x_jupiter = [frame[4]/AU for frame in traj]
    y_jupiter = [frame[5]/AU for frame in traj]

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal')
    ax.plot(x_earth, y_earth, label='Earth')
    ax.plot(x_jupiter, y_jupiter, label='Jupiter')
    ax.legend()
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.show()
    # Shows difference in earths trajectory Earth Sun vs Earth Sun Jupiter
    t_earth, traj_earth = orbit(M_sun, M_earth, rp_earth, e_earth, 120000, dt)

    x_earth_orbit = [frame[2] for frame in traj_earth]
    y_earth_orbit = [frame[3] for frame in traj_earth]
    t, traj = three_body(M_sun, M_earth, M_jupiter, rp_earth, rp_jupiter, e_earth, e_jupiter, 120000, dt)
    x_earth = [frame[2] for frame in traj]
    y_earth = [frame[3] for frame in traj]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121)
    ax.set_aspect('equal')
    ax.plot(x_earth_orbit, y_earth_orbit, linewidth=0.1)
    plt.title('Two Body Earth')
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')

    ax1 = fig.add_subplot(122)
    ax1.set_aspect('equal')
    ax1.plot(x_earth, y_earth, linewidth=0.1)
    plt.title('Three Body Earth')
    plt.xlabel('x (AU)')
    plt.show()

    # Moon Earth Sun
    dt = 0.00001 # Yr
    dt = dt*Year
    t, traj = three_body_moon(M_sun, M_earth, M_moon, rp_earth, rp_moon, e_earth, e_moon, 100170, dt)

    x_earth = [frame[2]/AU for frame in traj]
    y_earth = [frame[3]/AU for frame in traj]
    x_moon = [frame[4]/AU for frame in traj]
    y_moon = [frame[5]/AU for frame in traj]

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121)
    ax.set_aspect('equal')
    ax.plot(x_earth, y_earth, label='Earth', linewidth=0.5)
    ax.plot(x_moon, y_moon, label='Moon', linewidth=0.5)
    ax.legend()
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')

    ax1 = fig.add_subplot(122)
    ax1.set_xlim([-1.1, -0.90])
    ax1.set_ylim([-0.1, 0.1])
    ax1.set_aspect('equal')
    ax1.plot(x_earth, y_earth, label='Earth', linewidth=0.1)
    ax1.plot(x_moon, y_moon, label='Moon', linewidth=0.1)
    ax.legend()
    plt.xlabel('x (AU)')
    plt.show()
    """

    # Earth, Mars, Sun transfer orbit
    dt = 0.0001  # Yr
    dt = dt * Year

    t, traj = three_body(M_sun, M_earth, M_mars, rp_earth, rp_mars, e_earth, e_mars, 120000, dt)
    #print(len(traj))
    #print(traj)
    x_earth = [frame[2] / AU for frame in traj]
    y_earth = [frame[3] / AU for frame in traj]
    x_mars = [frame[4] / AU for frame in traj]
    y_mars = [frame[5] / AU for frame in traj]
    #rocket_data_X, rocket_data_Y = np.zeros_like(x_earth), np.zeros_like(y_earth)
    #rocket_data_X[0], rocket_data_Y[0] = x_earth[0], y_earth[0]
    #mars_half_index = int((len(x_mars)+1)/2)
    #print(mars_half_index)
    #rocket_data_X[-1], rocket_data_Y[-1] = x_mars[mars_half_index], y_mars[mars_half_index]
    #print(rocket_data_X, rocket_data_Y)
    #if for i in rocket_data_X, rocket_data_Y == 0:
    #    rocket_data_X[i], rocket_data_Y[i] = x_sun[i], y_sun[i]



    #x_rocket = x_earth + x_mars  # [frame[] / AU for frame in traj]
    #y_rocket = y_earth + y_mars #[frame[] / AU for frame in traj]
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_aspect('equal')
    ax.plot(x_earth, y_earth, label='Earth')
    ax.plot(x_mars, y_mars, label='Mars')
    #ax.plot(x_rocket, y_rocket, label='Rocket')

    ax.legend()
    plt.xlabel('x (AU)')
    plt.ylabel('y (AU)')
    plt.show()

