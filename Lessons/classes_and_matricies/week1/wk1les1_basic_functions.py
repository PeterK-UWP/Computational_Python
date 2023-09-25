import numpy as np


def square_cube(x):
    square = x ** 2
    cube = x ** 3
    return square, cube


print(f'area of square: {square_cube(3)[0]}, area of cube: {square_cube(3)[1]}')


def sum_squares_cubes(square, cube):
    return square + square, cube + cube


def sums(x, func):
    return func(x) + func(x)


print(f'sum of squares: {sum_squares_cubes(square_cube(3)[0], square_cube(3)[1])[0]},'
      f'sum of cubes: {sum_squares_cubes(square_cube(3)[0], square_cube(3)[1])[1]},')
print(f'sum of functions: {sums(3, square_cube)}')


def newton_sqrt(x):
    def sqrt_recur(x, y):
        def good_enough(x, y):
            epsilon = 1e-8
            return np.abs(x - y ** 2) <= epsilon

        def improved(x, y):
            return 0.5 * (y + x / y)

        return y if good_enough(x, y) else sqrt_recur(x, improved(x, y))

    trial = 100
    return sqrt_recur(x, trial)


print(newton_sqrt(2), np.sqrt(2))
