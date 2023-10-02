import numpy as np
"""
we are solving dxi/dt = zeta(xi, t) where xi is an n-dimensional vecotr and zeta is called an n-dimensional vector 
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



if __name__ == '__main__':
    print(
    'chicken'
    )