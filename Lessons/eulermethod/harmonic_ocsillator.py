import numpy as np
from Lessons.eulermethod import evolution as ev
import matplotlib.pyplot as plt


def harmonic(amplitude, spring_constant, mass, dt, N, method):
    def zeta(xi, t=None):
        return np.array(xi[1], -xi[0])
    tau = np.sqrt(mass/spring_constant)
    chi = amplitude
    scale = np.array([tau, chi, chi/tau])
    initial_data_array = np.array([amplitude, 0.0])
    t = np.arange(0, (N-1)*dt, dt)
    trajectory = ev.evolve(initial_data_array, 0.0, dt, N, zeta, method, scale)

    # energy values
    energy_values = []
    for i in trajectory:
        energy_values.append(0.5*mass*i[1]*i[1] + 0.5*spring_constant*i[0]*i[0])
    return t, trajectory, np.array(energy_values)




if __name__ == '__main__':
    t, traj_euler, energy_euler = harmonic(1.0, 1.0, 1.0, 0.01, 1000, ev.euler_step)
    plt.plot(t, energy_euler, label='Euler')
    t, traj_verlet, energy_verlet = harmonic(1.0, 1.0, 1.0, 0.01, 1000, ev.verlet_step)
    plt.plot(t, energy_verlet, label='Verlet')
    t, traj_leap, energy_leap = harmonic(1.0, 1.0, 1.0, 0.01, 1000, ev.leap_frog_step)
    plt.plot(t, energy_leap, label='Leap Frog')
    t, traj_rk4, energy_rk4 = harmonic(1.0, 1.0, 1.0, 0.01, 1000, ev.rk4_step)
    plt.plot(t, energy_rk4, label='Runge-Kutta 4')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy (J)')
    plt.legend()
    plt.show()
