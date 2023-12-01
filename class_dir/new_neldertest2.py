import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt
from matplotlib import cm

class Nelder_Mead:
    def __init__(self, dim, volume, param=None):
        self.dim = dim
        self.simplex = np.zeros(dim * (dim + 1))
        self.simplex = np.reshape(self.simplex, (dim + 1, dim))
        self.param = param
        self.cost_func = volume
        self.val = np.zeros(dim + 1)
        self.alpha = 1.0
        self.beta = 0.5
        self.gamma = 2.0
        self.delta = 0.5
        print(self.simplex)

    def get(self, index, entry=None):  # entry is an n dimensional array
        if entry is None:
            return self.val[index], self.simplex[index]
        else:
            self.simplex[index] = np.array(entry)
            if self.param is None:
                self.val[index] = self.cost_func(entry)
            else:
                self.val[index] = self.cost_func(entry, self.param)
            return self.val[index], self.simplex[index]

    def max_min(self):
        indices = np.argsort(self.val)
        return indices[-1], indices[0]

    def sorted_index(self, index):
        indices = np.argsort(self.val)
        return indices[index]

    def centroid(self):
        max_val, min_val = self.max_min()
        p_bar = np.zeros(self.dim)
        for i in range(max_val):
            p_bar += self.simplex[i]
        for i in range(max_val + 1, self.dim + 1):
            p_bar += self.simplex[i]

        p_bar = p_bar / self.dim
        return p_bar

    def reflect(self, p_bar, p):
        return (1 + self.alpha) * p_bar - self.alpha * p

    def expand(self, p_bar, p):
        return self.gamma * p + (1 - self.gamma) * p_bar

    def contract(self, p_bar, p):
        return self.beta * p + (1 - self.beta) * p_bar

    def point_shrink(self, pl, p):
        return self.delta * p + (1 - self.delta) * pl

    def rmsd(self):
        average = 0.0
        square_average = 0.0
        for entry in self.val:
            average += entry
            square_average += entry * entry
        average /= self.dim
        square_average /= self.dim
        return np.sqrt(np.abs(square_average - average * average))

    def nelder_mead_step(self):
        max_original, min_original = self.max_min()
        max_second = self.sorted_index(-2)
        p_bar = self.centroid()
        p_r = self.reflect(p_bar, self.simplex[max_original])

        if self.param is None:
            val_r = self.cost_func(p_r)
        else:
            val_r = self.cost_func(p_r, self.param)

        if val_r >= self.val[min_original] and val_r < self.val[max_second]:
            self.get(max_original, p_r)
            return self.val[min_original], min_original, self.rmsd(), 'reflect'

        if val_r < self.val[min_original]:
            p_ex = self.expand(p_bar, p_r)
            if self.param is None:
                val_ex = self.cost_func(p_ex)
            else:
                val_ex = self.cost_func(p_ex, self.param)

            if val_ex < val_r:
                self.get(max_original, p_ex)
                return val_ex, max_original, self.rmsd(), 'expand'
            else:
                self.get(max_original, p_r)
                return val_r, max_original, self.rmsd(), 'reflect'

        if val_r >= self.val[max_second]:
            if val_r < self.val[max_original]:
                p_c = self.contract(p_bar, p_r)
                if self.param is None:
                    val_c = self.cost_func(p_c)
                else:
                    val_c = self.cost_func(p_c, self.param)
                if val_c < val_r:
                    self.get(max_original, p_c)
                    return self.val[min_original], min_original, self.rmsd(), 'contract'
                else:
                    pl = self.simplex[min_original]
                    for i in range(min_original):
                        p_sh = self.point_shrink(pl, self.simplex[i])
                        self.get(i, p_sh)
                    for i in range(min_original + 1, self.dim + 1):
                        p_sh = self.point_shrink(pl, self.simplex[i])
                        self.get(i, p_sh)
                    return self.val[min_original], min_original, self.rmsd(), 'shrink'
            if val_r >= self.val[max_original]:
                p_c = self.contract(p_bar, self.simplex[max_original])
                if self.param is None:
                    val_c = self.cost_func(p_c)
                else:
                    val_c = self.cost_func(p_c, self.param)
                if val_c < self.val[max_original]:
                    self.get(max_original, p_c)
                    return self.val[min_original], min_original, self.rmsd(), 'contract2'
                else:
                    pl = self.simplex[min_original]
                    for i in range(min_original):
                        p_sh = self.point_shrink(pl, self.simplex[i])
                        self.get(i, p_sh)
                    for i in range(min_original + 1, self.dim + 1):
                        p_sh = self.point_shrink(pl, self.simplex[i])
                        self.get(i, p_sh)
                    return self.val[min_original], min_original, self.rmsd(), 'shrink'

    def optimize(self):
        rms = 1.0
        while rms > 1e-4:
            val, index, rms, step_name = self.nelder_mead_step()
            print(val, index, rms, step_name)
        return self.simplex[index]

def volume(x, p):
    return (p[0] * x[0] - p[1]) ** 2 + (p[2] * x[1] - p[3]) ** 2 + (p[4] * x[2] - p[5]) ** 2  # Update for 3D

if __name__ == '__main__':
    param = [4.0, 8.0, 3.0, 1.0, 2.0, 5.0]  # Update parameters for 3D
    nm = Nelder_Mead(3, volume, param)  # Update dimension to 3
    v, s = nm.get(0, [1, 1, 1])  # Initial points in 3D
    v, s = nm.get(1, [6, 0, 0])
    v, s = nm.get(2, [0, 3, 0])
    x = nm.optimize()

    # Visualization (3D scatter plot)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(s[:, 0], s[:, 1], s[:, 2], c='r', marker='o', label='Simplex vertices')
    ax.scatter(x[0], x[1], x[2], c='b', marker='x', s=100, label='Optimum')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()
