"""
pu/pt + (u * grad)u = -1/ρ * grad_p + nu lap_u + f
incompressibility div u = 0

u:
p: pressure
f: forcing(this case = 0)
nu: kinemetic viscosity
ρ: density
t: time
grad: gradient (defining nonlinear convection)
lap: laplace operator

Lid driven scenario (2D with top of container has flow driving circular flow inside)
dirichlet boundary conditions (u = 0 and v = 0 on edges of container)

splitting method: Chorin's projection method:

1.) solve momentum without pressure gradient for tentative velocity
    pu/pt + (u * grad) =  nu lap_u

2.) solve pressure poisson eq:
    lap_p = rho/dt * grad * u

3.) correct velocities
    u < - u - dt/rho * grad_p

in 2D: in index notation u = [ux, uy], v = [vx, vy]
"""
# Input Commands
N_POINTS = 41                   #
DOMAIN_SIZE_X = 3.0
DOMAIN_SIZE_Y = 1.0
N_ITERATION = 500
TIME_STEP_LENGTH = 0.001
KINEMATC_VISCOSITY = 0.1        # nu
DENSITY = 1.0                   # rho
HORIZONTAL_VELOCITY_TOP = 1.0

N_PRESSURE_POISSON_ITERATIONS = 50
STABILITY_SAFETY_FACTOR = 0.5


import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def Navier_Stokes():
    element_length = DOMAIN_SIZE / (N_POINTS - 1)                # step size of each term
    x_array = np.linspace(0, DOMAIN_SIZE_X, N_POINTS)         # range of points for x values
    y_array = np.linspace(0, DOMAIN_SIZE_Y, N_POINTS)         # range of points for y values

    x_mesh, y_mesh = np.meshgrid(x_array, y_array)

    # initial conditions: fluid is at rest, no pressure, and zero velocities
    u_previous = np.zeros_like(x_mesh)  # velocity in x direction
    v_previous = np.zeros_like(x_mesh)  # velocity in y direction
    pressure_previous = np.zeros_like(x_mesh) # pressure at each coordinate

    # Central Difference Method to solve for the Laplacian can lead to instability in diffusion...

    # scale factor*function - distence: (diagonal dominence)
    return




# derivative



if __name__ == "__main__":
    a = [1, 2, 3,4, 5, 6, 7, 8]
    print(a[1:-1]) # [2, 3, 4, 5, 6, 7] ignores the boundary