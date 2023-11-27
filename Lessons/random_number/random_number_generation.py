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
        return self.rand


    def max_rand(self):
        return self.max


    def random(self, range=1.0):
        return self.next() / self.max_rand() * range


def pi_test(rand, N):
    count1 = 0
    count2 = 0
    val = []
    for i in range(N):
        x = rand.random(2) - 1
        y = rand.random(2) - 1
        if x * x + y * y < 1:
            count1 += 1
        count2 += 1
        val.append(count1/count2 * 4)
    return val

if __name__ == '__main__':
    small_random = lcg_random(3, 0, 7, time.time_ns())
    print(small_random.max_rand())
    for i in range(10):
        print(small_random.random(), end=' ')
    print(' ')

    med_random = lcg_random(17, 21, 128, time.time_ns())
    print(med_random.max_rand())
    for i in range(10):
        print(med_random.random(), end=' ')
    print(' ')

    high_random = lcg_random(1140671485, 128201163, 2**24, time.time_ns())
    print(high_random.max_rand())
    for i in range(10):
        print(high_random.random(), end=' ')
    print(' ')

    pi_small_test = pi_test(small_random, 1000)
    pi_med_test = pi_test(med_random, 1000)
    pi_high_test = pi_test(high_random, 1000)

    plt.plot(pi_small_test, color='blue')
    plt.plot(pi_med_test, color='green')
    plt.plot(pi_high_test, color='orange')
    plt.axhline(y = np.pi, color='red', linestyle='dotted')
    plt.xlabel('step')
    plt.ylabel(r'Value of $\pi$')
    plt.show()
    print(pi_small_test[-1], pi_med_test[-1], pi_high_test[-1], np.pi)
    # how to check how good your random numbers are? Guy Squared test
    # linear congruence. folding method
    # statistically independent
    # one output doesn't depend on the previous output.
    # generate a time series, and make an auto correlation function
    # c_n = 1/N sum from i to N of x sub i times x sum i plus n
    # - 1/N^2 sum from i to N of x sub i times sum from k to N of x sub k (all divided by varience) ie (deltax sub i)^2
    # exponential decay when implemented, should be more like a delta function at 0


    # make a sample of x values x1, x2, x3, ..., xn and put them into the correlation function
    # A e^-alpha*n, n>1/alpha, alpha depends on a b c (a, c, m above)
    # check distribution, periodicity (get from Fouier Transform somehow),
    # FT: x tilda (k/N) = sum from n of x sub n e^(-i2pi*k/N * n), k = 0, 1, ..., N-1 discrete form
    # plot |xtilda|^2 vs k. compare with numpy random
    # computing chi^2 = sum i to m bins of (N_i - P_i)^2/Sum from i to m of P_i
    # ie. binning from computational.

