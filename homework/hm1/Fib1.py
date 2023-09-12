import numpy as np

def fibonacci(n):
    return 0 if n==0 else 1 if n==1 else fibonacci(n-1) + fibonacci(n-2)

def recure_fib(n):
    terms = []
    for i in np.linspace(n-n, n, n+1):
        terms.append(fibonacci(i))
    return terms


if __name__ == "__main__":
    # print(fibonacci(eval(input("input a value n = 0, 1, ... to get F_n of the Fibonacci Sequence:"))))
    print(recure_fib(eval(input("Get the first n+1 terms:"))))