""" MA3.py

Student: Alicia Roth
Mail: alicia.roth.4540@student.uu.se
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
from numba import njit

# Exc1
def approximate_pi(n):
    # n is the number of points
    # Write your code here
    x_inside = []
    y_inside = []

    x_outside = []
    y_outside = []

    for i in range(n):
        x = random.uniform(-1,1)
        y = random.uniform(-1,1)
         
        if x**2 + y**2 <= 1:
            x_inside.append(x)
            y_inside.append(y)

        else:
            x_outside.append(x)
            y_outside.append(y)

    n_c = len(x_inside)
    pi_approx = (n_c/n)*4

    #plt.figure(figsize = (6,6))
    plt.scatter(x_inside, y_inside, color ='red', s = 2, label="Innanför cirkeln")
    plt.scatter(x_outside, y_outside, color ='blue', s = 2, label="Utanför cirkeln")

    plt.xlim(-1,1)
    plt.ylim(-1,1)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Monte Carlo-approximation av pi, n = {n}')
    plt.legend()
    plt.savefig(f"monte_carlo_pi_{n}.png", dpi=300)
    plt.show()



    return pi_approx

# Exc2, approximation
def sphere_volume(n, d): 
    # n is the number of points
    # d is the number of dimensions of the sphere 
    circles = [[random.uniform(-1,1) for _ in range(d)]for _ in range(n)] #skapar n punkter i d dimensioner
        
    distances = list(map(lambda point: sum(map(lambda x: x**2, point)), circles)) #ny lista m. ^2
        
    inside_sphere = list(filter(lambda x: x <= 1, distances)) #lista med de punkter som är innanför sfären
        
    volume = (len(inside_sphere)/n) * (2**d) #beräkninga av volymen

    return volume

#Exc2, real value
def hypersphere_exact(n, d):
    # n is the number of points
    # d is the number of dimensions of the sphere 
    V_d = m.pi**(d/2)/(m.gamma(d/2+1))
    return V_d

#Exc3: numba version
@njit
def sphere_volume_numba(n:int, d:int)->float:
    # n is the number of points
    # d is the number of dimensions of the sphere
    #np is the number of processes
    n_c = 0
    for _ in range(n):
        distance = 0.0
        for _ in range(d):
            x = random.uniform(-1,1)
            distance = x**2
            if distance <= 1:
                n_c += 1
    
    volume = (n_c/n) * (2.0**d)
    return volume




#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel2(n, d, np=10):
    # n is the number of points
    # d is the number of dimensions of the sphere
    # np is the number of processes
    with future.ProcessPoolExecutor() as ex: 
        results = ex.map(sphere_volume, [n//np]*np, [d]*np)
        
    return mean(results)
    
def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        print(approximate_pi(n))

    # Exc2
    n = 100000
    d = 2
    print(sphere_volume(n, d))
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    print(sphere_volume(n, d))
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11

    for i in range(3):
        start = pc()
        sphere_volume(n, d)
        stop = pc()

        start_2 = pc()
        sphere_volume_numba(n, d)
        stop_2 = pc()

        print(f"Exc3: {i+1} Sequential time of {d} and {n}: {stop-start}")
        print(f"Exc3: {i+1} numba time of {d} and {n}: {stop_2-start_2}")

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()

    start_2 = pc()
    sphere_volume_parallel2(n, d)
    stop_2 = pc()

    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    print(f"Exc4: Parallell time of {d} and {n}: {stop_2-start_2}")

    
    

if __name__ == '__main__':
	main()
