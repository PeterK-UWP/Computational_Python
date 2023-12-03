"""import numpy as np

print(np.zeros([3, 12]))
print(np.zeros([4,3]))
print(np.zeros([6]))
print(np.zeros([3,2]))

print(np.zeros(3, 3))
def volume(x, p, number_dolphins): # not what needs to be changed...
    N = len(x)
    pair_distance = np.zeros(N, N)
    for i in range(1, N):
        pair_distance[i, i:N] = number_dolphins[i:N] - number_dolphins[0:N - i]
        print(i, number_dolphins[i:N])"""
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def cost_function(hunters):
    cost = 0.0
    for i in range(len(hunters)):
        for j in range(i + 1, len(hunters)):
            distance = np.linalg.norm(hunters[i] - hunters[j])
            if distance < 0.1:
                cost += 1 / distance  # Too close, inverse square
            else:
                cost += distance ** 2  # Loosely spread, square
    return cost


def visualize_hunters(hunters):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    hunters = np.array(hunters)
    ax.scatter(hunters[:, 0], hunters[:, 1], hunters[:, 2], c='r', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Hunter Positions')

    plt.show()


def nelder_mead_optimization(dim, num_hunters):
    initial_guess = np.random.rand(num_hunters, dim)  # Initial random positions of hunters
    result = minimize(cost_function, initial_guess, method='Nelder-Mead')

    optimal_positions = result.x
    optimal_cost = result.fun

    print("Optimal Hunter Positions:", optimal_positions)
    print("Optimal Cost:", optimal_cost)

    visualize_hunters(optimal_positions)


if __name__ == "__main__":
    dimension = 3  # 3D space
    num_hunters = 5  # You can adjust the number of hunters as needed

    #nelder_mead_optimization(dimension, num_hunters)
    a = [1, 2, 3]
    b = [1, 2, 3]
    print(np.array(a) - np.array(b))
    c = [[1, 2],[2, 3]]
    print(range(c))


