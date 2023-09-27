import numpy as np
import time
import matplotlib.pyplot as plt
from linear_solver import jacobi, gaussian_elimination, gaussian_upper_triangle, back_substitution

# solving pypp + qyp + ry = s   on domain [a, b] with BC

def general_linear_second_ordered(starting_bound, ending_bound, no_pts,
                                  coef1, coef2, coef3, non_homo_func, boundary_conditions, method='gaussian_elimination'):
    increment = (ending_bound - starting_bound)/(no_pts-1)
    square_increment = increment*increment

    #Ax = b matrix*x = vector_b
    matrix = np.zeros((no_pts, no_pts))
    vector_b = np.zeros(no_pts)
    vector_x = np.zeros(no_pts)

    for i in range(no_pts):
        # for each element in v assign a value
        vector_x[i] = starting_bound + increment*i
        # for each diagonal in matrix assign a value
        matrix[i][i] = -2*coef1(vector_x[i]) - coef2(vector_x[i])*increment + coef3(vector_x[i]) * square_increment

        # assign value to off diagonal, ignores diagonal
        try:
            matrix[i][i+1] = coef1(vector_x[i]) + coef2(vector_x[i])*increment
        except Exception as e:
            pass

        if i > 0:
            matrix[i][i-1] = coef1(vector_x[i])
        vector_b[i] = non_homo_func(vector_x[i])*square_increment

    if method == 'jacobi':
        begin_time = time.time()
        vector_y = jacobi(matrix, vector_b, fixed=boundary_conditions)
        end_time = time.time()
        compute_time = (end_time - begin_time)*1000
    elif method == 'gaussian_elimination':
        begin_time = time.time()
        vector_y = gaussian_elimination(matrix, vector_b, fixed=boundary_conditions)
        end_time = time.time()
        compute_time = (end_time - begin_time)*1000
    return vector_x, vector_y, compute_time






