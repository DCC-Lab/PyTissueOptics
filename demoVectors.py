import numpy as np
try:
    import cupy as cp
    from cupyx import optimizing
except:
    pass

import random
from pytissueoptics import *
from time import time_ns


def print_to_stdout(*a):

    print(*a, file=sys.stdout)

def getScatteringDistance(N, mu_s):
    scalar = [] 
    for i in range(N):
        scalar.append(random.random())
    return scalar

def getScatteringAngle(N):
    theta = [] 
    phi = [] 
    for i in range(N):
        theta.append(random.random() * 2 * np.pi)
        phi.append(random.random() * 2 * np.pi)
    return theta, phi


time0 = time_ns()
batches = 10
N = 5000
position = Vectors(N=N)

# direction = Vectors([zHat]*N)
# er = Vectors([xHat]*N)
# weight = Scalars([1.0]*N)

direction = Vectors([[0, 0, 1]]*N)
# direction = Vectors([zHat]*N)
er = Vectors([[1, 0, 0]]*N)
# er = Vectors([xHat]*N)
weight = Scalars([1.0]*N)

isAlive = True
time1 = time_ns()
print(f"Initialization   ::  {(time1-time0)/1000000000}s")

time2 = time_ns()
for i in range(batches):
    time3 = time_ns()
    count = 0
    while isAlive:
        count += 1
        theta = Scalars.random(N=N) * 2 * np.pi
        phi = Scalars.random(N=N) * 2 * np.pi

        d = Scalars.random(N=N)

        er.rotateAround(direction, phi)
        direction.rotateAround(er, theta)
        position = position + direction*d
        weight *= 0.9
        isAlive = (weight.v > 0.001).any()

    weight = Scalars([1.0] * N)
    isAlive = True
    time4 = time_ns()
    print(f"Batch #{i}   ::  {(time4-time3)/1000000000}s ::  {(i+1)*N/1000000}M photons/{N*batches/1000000}M    ::  ({int((i+1)*N*100/(N*batches))}%)")

time5 = time_ns()
print(f"Calculations Only    ::  {N*batches/1000000}M photons    ::  {(time5-time2)/1000000000}s")
print(f"Complete Process    ::  {N*batches/1000000}M photons    ::  {(time5-time0)/1000000000}s")
