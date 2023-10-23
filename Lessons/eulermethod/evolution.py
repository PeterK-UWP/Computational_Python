import numpy as np
"""
we are solving dxi/dt = zeta(xi, t) where xi is an n-dimensional vector and zeta is called an n-dimensional vector 
field or simply RHS, with the initial condition xi0 - xi(t0)"""
def evolve(y0: np.ndarray, t0: float, dt: float, n: int,
           f: callable, method: callable, scale=None, param=None) -> np.ndarray:  # type to expect
    dof = len(y0)
    t=t0
    a = []
    y1 = y0.copy()
    if scale is None:
        if param is None:
            for i in range(1, n):
                a.append(y1)
                y1 = method(y1, t, dt, f)
                t += dt
        else:
            for i in range(1, n):
                a.append(y1)
                y1 = method(y1, t, dt, f, param)
                t += dt
    else:
        if len(scale) != dof + 1:
            raise Exception(f'scale vector must have dim={dof + 1}')
        t_scale = scale[0]
        dt = dt/t_scale
        scal = np.array(scale[1:])
        y = y1/scal
        if param is None:
            for i in range(1, n):
                a.append(y1)
                y = method(y, t, dt, f)
                y1 = y*scal
                t += dt
        else:
            for i in range(1, n):
                a.append(y1)
                y = method(y, t, dt, f, param)
                y1 = y*scal
                t += dt
    return np.array(a)


def euler_step(y:np.ndarray, t:float, dt:float, f:callable)->np.ndarray:
    y = y+dt * f(y, t)
    return y

def euler_step_param(y:np.ndarray, t:float, dt:float, f:callable, param)->np.ndarray:
    y = y + dt * f(y, t, param)
    return y


def verlet_step(y: np.ndarray, t:float, dt:float, f:callable)->np.ndarray:
    dof = len(y) // 2
    for i in range(dof):
        y[2* i + 1] = y[2* i + 1] + dt * f(y, t)[2*i + 1]
        y[2*i] = y[2*i] + dt*f(y, t)[2*i]
    return y


def verlet_step_param(y:np.ndarray, t:float, dt:float, f:callable, param)->np.ndarray:
    dof = len(y) // 2
    for i in range(dof):
        y[2 * i + 1] = y[2 * i + 1] + dt * f(y, t, param)[2 * i + 1]
        y[2 * i] = y[2 * i] + dt * f(y, t, param)[2 * i]
    return y

def leap_frog_step(y:np.ndarray, t:float, dt:float, f:callable)->np.ndarray:
    dof = len(y) // 2
    for i in range(dof):
        y[2 * i] = y[2 * i] + dt * f(y, t)[2 * i]
        y[2 * i + 1] = y[2 * i + 1] + dt * f(y, t)[2 * i + 1]
    return y

def leap_frog_step_param(y:np.ndarray, t:float, dt:float, f:callable, param)->np.ndarray:
    dof = len(y) // 2
    for i in range(dof):
        y[2 * i] = y[2 * i] + dt * f(y, t, param)[2 * i]
        y[2 * i + 1] = y[2 * i + 1] + dt * f(y, t, param)[2 * i + 1]
    return y


def rk4_step(y:np.ndarray, t:float, dt:float, f:callable)->np.ndarray:
    k1 = f(y, t)
    k2 = f(y + k1*dt / 2.0, t + dt / 2.0)
    k3 = f(y + k2*dt / 2.0, t + dt / 2.0)
    k4 = f(y + k3*dt, t+dt)
    k = dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    y = y + k
    return y


def rk4_step_param(y:np.ndarray, t:float, dt:float, f:callable, param)->np.ndarray:
    k1 = f(y, t, param)
    k2 = f(y + k1*dt / 2.0, t + dt / 2.0, param)
    k3 = f(y + k2*dt / 2.0, t + dt / 2.0, param)
    k4 = f(y + k3*dt, t+dt, param)
    k = dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0
    y = y + k
    return y
