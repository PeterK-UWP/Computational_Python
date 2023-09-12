import numpy as np

# use recursive functions you can write any mathematical function
def sum_func(x_value, initial_index, final_index, summed_function):
    """
    use in line if statement for faster speeds

    if final_index == initial_index:
        summed_function(x_value, initial_index)
    else:
        summed_function(x_value, final_index) + sum_func(x_value, initial_index, final_index - 1, summed_function)
    return
    """
    return summed_function(x_value, initial_index) if final_index == initial_index \
        else summed_function(x_value, final_index) + sum_func(x_value, initial_index, final_index - 1, summed_function)


def factorial(final_index):
    """
    if final_index == 0:
        return 1
    else:
        final_index*factorial(final_index - 1)
    """
    return 1 if final_index == 0 else final_index * factorial(final_index - 1)


my_exp = lambda x: sum_func(x, 0, 10, lambda x, i: x**i / factorial(i))
my_sin = lambda x: sum_func(x, 0, 10, lambda x, i: (-1)**i * x**(2*i+1)/factorial(2*i+1))
my_cos = lambda x: sum_func(x, 0, 10, lambda x, i: (-1)**i * x**(2*i)/factorial(2*i))

if __name__ == '__main__':
    print(my_exp(1.0), np.exp(1.0))
    print(my_sin(np.pi/2.0), np.sin(np.pi/2.0))
    print(my_cos(np.pi), np.cos(np.pi))

    # also y = x.copy(): stores a lot of data
    x = [1, 2, 3]
    y = [4, 5, 6]
    print(f'original x, y: {x}, {y}')
    y = x
    z = x.copy()
    print(f'copy x, y: {y}, {z}')
    # much faster and simply swaps the names
    x = [1, 2, 3]
    y = [4, 5, 6]
    x, y = y, x
    print(f'name swap x, y: {x}, {y}')

    #Sets
    A = {1, 2, 3, 4, 5, 1, 3}
    B = {'a', 'a', 'b', 'c', 4}
    union = A|B
    intersect = A&B
    print(union, intersect)

    #Dictionary
    value1 = 30
    value2 = 40
    diction = {'key1': value1, 'key2': value2}
    print(diction['key1'], diction['key2'])

    #Iterators: use this inside for loops
    # access object that takes in an object and outputs iter(), next()
    ll = [1, 2, 3, 4, 5]
    list_iterator = iter(ll)
    x = next(list_iterator)
    x = next(list_iterator)
    print(x, end=' ')





