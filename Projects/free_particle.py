import numpy as np


def plane_wave(k, x, A, phi):

    return A*np.sin(k*x + phi)