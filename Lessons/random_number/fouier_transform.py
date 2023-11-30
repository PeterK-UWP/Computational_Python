import numpy as np
from numpy import linalg as lin
import matplotlib.pyplot as plt
from matplotlib import cm
import time
import random


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


def auto_correlate(data):
    mean = np.mean(data)
    var = np.var(data)

    ndata = data - mean

    acorr = np.correlate(ndata, ndata, 'full')[len(ndata) - 1:]
    accor = abs(acorr / var / len(ndata))
    return accor


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


    #~~~~~~~~~~~~~~~~~~~~~~~~~~#
    med_ran = np.zeros(1024)
    ran = np.zeros(1024)
    sys_ran = np.zeros(1024)
    x_axis = range(1024)
    for i in x_axis:
        med_ran[i] = med_random.random()
        ran[i] = high_random.random()
        sys_ran[i] = random.random()
    ac = auto_correlate(ran)
    med_ac = auto_correlate(med_ran)
    sys_ac = auto_correlate(sys_ran)

    plt.plot(x_axis, med_ac)
    plt.plot(x_axis, ac)
    plt.plot(x_axis, sys_ac)
    plt.xlabel(r'$n$')
    plt.ylabel(r'${C_n}$')
    plt.show()

    
    med_fft = np.fft.fft(med_ran, 1024)
    high_fft = np.fft.fft(ran, 1024)
    sys_fft = np.fft.fft(sys_ran, 1024)
    plt.plot(x_axis[1:], abs(med_fft[1:]))
    plt.plot(x_axis[1:], abs(high_fft[1:]))
    plt.plot(x_axis[1:], abs(sys_fft[1:]))
    plt.xlabel(r'$k$')
    plt.ylabel(r'$\tilde{x}_k$')
    plt.show()



