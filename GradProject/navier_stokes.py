# https://www.youtube.com/watch?v=BQLvNLgMTQE   # lid driven cavity
# example of navier stokes from youtube not my work!!
"""
pu/pt + (u * grad) = -1/rho * grad_p + nu lap_u + f
incompressibility div u = 0

u:
p: pressure
f: forcing(this case = 0)
nu: kinemetic viscosity
rho: density
t: time
grad: gradient (defining nonlinear convection)
lap: laplace operator

Lid driven senario (2D with top of container has flow driivng cirular flow inside)
dirichlet bounds (u = 0 and v = 0 on edges of container)

splitting method: Chorin's projection method:

1.) solve momentum without pressure gradient for tentative velocity
    pu/pt + (u * grad) =  nu lap_u

2.) solve pressure poisson eq:
    lap_p = rho/dt * grad * u

3.) correct velocities
    u < - u - dt/rho * grad_p

in 2D: in index notation u = [ux, uy]
                            v = [vx, vy]

"""
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

N_POINTS = 41
DOMAIN_SIZE = 1.0
N_ITERATION = 500
TIME_STEP_LENGTH = 0.001
KINEMATC_VISCOSITY = 0.1
DENSITY = 1.0
HORIZONTAL_VELOCITY_TOP = 1.0

N_PRESSURE_POISSON_ITERATIONS = 50

STABILITY_SAFETY_FACTOR = 0.5
def main():
    element_length = DOMAIN_SIZE / (N_POINTS - 1)  # uniform discretizations, -1 because no. includes bc pts
    x = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)
    y = np.linspace(0.0, DOMAIN_SIZE, N_POINTS)

    X, Y = np.meshgrid(x, y)  # returns a list of coord matrices from coord vectors

    # initial condition fluid at rest, u and v = 0, and pressure = 0
    u_prev = np.zeros_like(X) # v in x
    v_prev = np.zeros_like(X) # v in y
    press_prev = np.zeros_like(X) # pressure = 0

    def central_difference_x(field):
        diff = np.zeros_like(field)  # differentiated object = zero like tensor

        # fill in interior points, not on boundary and fill with central diff
        diff[1:-1, 1:-1] = (
            field[1:-1, 2: ]
            -
            field[1:-1, 0:-2]
        ) / (
                2 * element_length
        )  # field[y:y, x:x] => diff in x
        # advanced by 1 - subtract by 1

        return diff

    def central_difference_y(field):
        diff = np.zeros_like(field)  # differentiated object = zero like tensor

        # fill in interior points, not on boundary and fill with central diff
        diff[1:-1, 1:-1] = (field[2: , 1:-1] - field[0:-2, 1:-1]) / (2 * element_length)  # field[y:y, x:x] => diff in x
        # advanced by 1 - subtract by 1

        return diff

    def laplace(field):
        diff = np.zeros_like(field)
        diff[1:-1, 1:-1] = (field[1:-1, 0:-2] + field[0:-2, 1:-1] - 4*field[1:-1, 1:-1] +
                            field[1:-1, 2: ] + field[2: , 1:-1]) / element_length**2
        return diff

    maxiumum_possible_timestep_length = (0.5 * element_length**2/KINEMATC_VISCOSITY)
    # heat eq
    if TIME_STEP_LENGTH > STABILITY_SAFETY_FACTOR * maxiumum_possible_timestep_length:
        raise RuntimeError("stability not guaranteed")

    for _ in tqdm(range(N_ITERATION)):
        d_u_prev__dx = central_difference_x(u_prev)
        d_u_prev__dy = central_difference_y(u_prev)
        d_v_prev__dx = central_difference_x(v_prev)
        d_v_prev__dy = central_difference_y(v_prev)
        laplace__u_prev = laplace(u_prev)
        laplace__v_prev = laplace(v_prev)

        # perfrom tentative step by solving momentum eq w/out pressure grad.
        # forward euler in time
        u_tent = (u_prev + TIME_STEP_LENGTH * (-(u_prev * d_u_prev__dx + v_prev * d_u_prev__dy)
                                        + KINEMATC_VISCOSITY * laplace__u_prev))

        v_tent = (v_prev + TIME_STEP_LENGTH * (-(v_prev * d_v_prev__dx + v_prev * d_v_prev__dy)
                                        + KINEMATC_VISCOSITY * laplace__v_prev))

        # velocity bc: homo dirichlet bc everywhere except the horizontal at the top with is prescribed

        u_tent[0, :] = 0.0
        u_tent[:, 0] = 0.0
        u_tent[:, -1] = 0.0
        u_tent[-1, :] = HORIZONTAL_VELOCITY_TOP

        v_tent[0, :] = 0.0
        v_tent[:, 0] = 0.0
        v_tent[:, -1] = 0.0
        v_tent[-1, :] = 0.0

        d_u_tent__dx = central_difference_x(u_tent)
        d_v_tent__dy = central_difference_y(v_tent)

        # compute pressure correction by solving the pressure poisson eq

        rhs = (DENSITY / TIME_STEP_LENGTH * (d_u_tent__dx + d_v_tent__dy))

        for _ in range(N_PRESSURE_POISSON_ITERATIONS):
            p_next = np.zeros_like(press_prev)
            p_next[1:-1, 1:-1] = 1/4 * (+ press_prev[1:-1, 0:-2] + press_prev[0:-2, 1:-1]
                                    + press_prev[1:-1, 2: ] + press_prev[2: , 1:-1]
                                    - element_length**2*rhs[1:-1, 1:-1])
            # pressure bc: homo neumann bc everywhere except for the top, where it is homo dirichlet bc

            p_next[:, -1] = p_next[:, -2]
            p_next[0, :] = p_next[1, :]
            p_next[:, 0] = p_next[:, 1]
            p_next[-1, :] = 0.0

            press_prev = p_next

        d_p_next__dx = central_difference_x(p_next)
        d_p_next__dy = central_difference_y(p_next)

        # correct velocities such that fluid is incompressible
        u_next = (u_tent - TIME_STEP_LENGTH / DENSITY * d_p_next__dx)
        v_next = (v_tent - TIME_STEP_LENGTH / DENSITY * d_p_next__dy)

        u_next[0, :] = 0.0
        u_next[:, 0] = 0.0
        u_next[:, -1] = 0.0
        u_next[-1, :] = HORIZONTAL_VELOCITY_TOP

        v_next[0, :] = 0.0
        v_next[:, 0] = 0.0
        v_next[:, -1] = 0.0
        v_next[-1, :] = 0.0

        # advance time
        u_prev = u_next
        v_prev = v_next
        p_prev = p_next

    plt.figure()
    plt.contourf(X, Y, p_next)
    plt.colorbar()

    #plt.quiver(X, Y, u_next, v_next, color='black')
    plt.streamplot(X, Y, u_next, v_next, color='black')
    plt.show()

    return

if __name__ == '__main__':
    main()



"""
plt.figure() #makes a fugure
plt.contourf(a, b, iteration)  # makes a contour along inputs
plt.colorbar() # color scaling
plt.quiver(a, b, iterx, itery, color='') #vector plot
plt.streamplot(a, b, xiter, yiter, color) makes a contour line more streamline
"""

