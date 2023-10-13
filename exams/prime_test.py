# Python program to print all
# primes smaller than or equal to
# n using Sieve of Eratosthenes

#https://www.geeksforgeeks.org/sieve-of-eratosthenes/#
def SieveOfEratosthenes(n):
    # Create a boolean array
    # "prime[0..n]" and initialize
    #  all entries it as true.
    # A value in prime[i] will
    # finally be false if i is
    # Not a prime, else true.
    prime = [True for i in range(n + 1)]
    print(prime)
    p = 2
    while (p * p <= n):

        # If prime[p] is not
        # changed, then it is a prime
        if (prime[p] == True):

            # Update all multiples of p
            for i in range(p * p, n + 1, p):
                prime[i] = False
        p += 1

    # Print all prime numbers
    p_numbers = []
    for p in range(2, n + 1):
        if prime[p]:
            p_numbers.append(p)
            #print(p)
    print(p_numbers)
    return p_numbers
        # Driver code








import matplotlib.pyplot as plt
import time as time
def primes(n):
    """
    The goal to make primes run to 10million is to exclude the use of division.
    :param n: integer
    :return: list of floats
    """
    # makes list from 2 to n
    integer_list = []
    for i in range(2, n+1):
        integer_list.append(i)
    # sets initial p
    p0 = 2
    #mult = []
    prime_values = integer_list.copy()
    #print(f'init {prime_values}')
    while p0**2 <= n:
        if prime_values[p0] == prime_values[p0]: # checks if the element in question is the same (prime)
            #print(f'init {prime_values}')
            for j in range(p0**2, n+1, p0): # makes a range of values in incraments of p0. (multplies of the vcalues)
                # #mult.append(j)
                #print(j)
                try:
                    enu = prime_values.index(j) # gets index of j in original list
                    prime_values.pop(enu) # removes indexed term from list
                except Exception as e:
                    pass
                #print(prime_values)
        p0 += 1 # updates to the next p
    print(f'final {prime_values}')
    return prime_values





if __name__ == '__main__':
    n = [100, 1000, 10000, 100000, 1000000, 10000000]
    for i in n:
        print("Following are the prime numbers smaller"),
        print("than or equal to", n)
        b_time = time.time()
        SieveOfEratosthenes(n)

        primes(n)
        e_time = time.time()
        print(compute_time)
        print('ms')
    """
    n = [10, 100, 1000, 10_000, 100_000]  # , 1_000_000, 10_000_000]     # B. run command(s)
    times = []
    for i in n:

        print(f'calculating primes up to {i}')
        b_time = time.time()
        primes(i)
        e_time = time.time()
        compute_time = (e_time - b_time) * 1000
        times.append(compute_time)
        # print(type(prime_time_list))
        # print(prime_time_list)
        # print(prime_time_list[0], prime_time_list[1])
    plt.scatter(n, times)
    plt.xlabel(f'Primes up to {n[-1]}')
    plt.ylabel(f'Computation Time (ms)')
    plt.show()


"""



