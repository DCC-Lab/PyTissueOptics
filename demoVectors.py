import numpy as np
import cupy as cp
import random
from pytissueoptics import *
from time import time_ns


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

batches = 10
N = 5000000

position = Vectors(N=N)
direction = Vectors([[0, 0, 1]]*N)
# direction = Vectors([zHat]*N)
er = Vectors([[1, 0, 0]]*N)
# er = Vectors([xHat]*N)
weight = Scalars([1.0]*N)

isAlive = True
print(cp.cuda.runtime.getDeviceCount())

timea = time_ns()
for _ in range(batches):
    while isAlive:
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

timeb = time_ns()

print(f"{N} photons per batch, {batches} batches, in {(timeb-timea)/1e9} seconds.")
