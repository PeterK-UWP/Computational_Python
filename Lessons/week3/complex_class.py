import numpy as np


class complex:
    # The constructor: can be invoked with additional argument
    # "polar" to contract numbers in polar form. However, internally numbers
    #  are managed in cartesian format.
    def __init__(self, x, y, rep=None):
        """The constructor for complex"""
        if rep is None:
            self.x = x
            self.y = y
        elif rep=="polar":
            self.x = x*np.cos(y)
            self.y = y*np.sin(y)



