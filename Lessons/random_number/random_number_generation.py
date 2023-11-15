import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt
from matplotlib import cm
import time

# random is the ignorance of not knowing.
# situation where you do not know everything.


# LINEAR CONGRUENCE NUMBER GENERATION
# discrete time series: (real life problems) range of motion and consistent sampling time.
# make a very long period, so it does not repeat after a long time. (CPU pseudo long periods)
# goal: get equation that is deterministic with 'folding' 'reflecting'.

# x_n+1 = ( a*x_n + b ) % c  : % is the folding where x_n+1 < c  # def next()
def is_power_of_two(x):
    return x and (not(x & (x - 1)))


def is_prime(n):
    for i in range(2, int(np.sqrt(n)) + 1):
        if (n % i) == 0:
            return False
    return True

def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a

def coprime(a, b):
    return gcd(a, b) == 1

# c != 0
# m = 2^b ; period m
# c&m are relative primes
#  (a - 1) % 4 == 0

# c = 0
# m 2^b ; period m/4
# seed has to be odd number
# (a - 3) % 8 == 0 or (a - 5) % 8 == 0

# c = 0
# m is prime; period m - 1
# (a^k - 1) % m == 0



class lcg_random:
    def __init__(self, a, c, m, seed):
        self.seed = seed
        self.rand = seed
        self.a = a
        self.c = c
        self.m = m
        self.max = 0

        if self.c == 0:
            if is_power_of_two(self.m) and self.rand % 2 != 0:
                if (self.a - 3) % 8 == 0 and (self.a - 5) % 8 == 0:
                    self.max=self.m / 4
                else:
                    raise Exception('Wrong combination of a, c, m')
            elif is_prime(self.m):
                correct = False
                for k in range(self.m - 1, 0, -1):
                    if int(self.a ** k - 1) % self.m == 0:
                        correct = True
                    if correct:
                        self.max = self.m - 1
                    else:
                        raise Exception('Wrong combination of a, c, m')
                else:
                    raise Exception('Wrong combination of a, c, m')
            else:
                if coprime(self.c, self.m) and (self.a - 1) % 4 == 0:
                    self.max = self.m
                else:
                    raise Exception('Wrong combination of a, c, m')

    def next(self):
        self.rand = (self.a * self.rand + self.c) % self.m
        
