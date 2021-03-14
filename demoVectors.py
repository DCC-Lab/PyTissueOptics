import numpy as np
import cupy as cp
import random
from pytissueoptics import *
from cupyx import optimizing
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

batches = 2
N = 8000000

position = Vectors(N=N)
direction = Vectors([[0, 0, 1]]*N)
# direction = Vectors([zHat]*N)
er = Vectors([[1, 0, 0]]*N)
# er = Vectors([xHat]*N)
weight = Scalars([1.0]*N)

isAlive = True

# cuda stuff
print(cp.cuda.runtime.getDeviceCount())
mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()

timea = time_ns()
for i in range(batches):
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

        if not count % 10:
            print(f"Pool Available:{mempool.total_bytes()/1000000}MB, Allocated:{mempool.used_bytes()/1000000}MB, Pool Blocks:{pinned_mempool.n_free_blocks()}")  # 512

    weight = Scalars([1.0] * N)
    isAlive = True

timeb = time_ns()

print(f"{N} photons per batch, {batches} batches, in {(timeb-timea)/1e9} seconds.")
