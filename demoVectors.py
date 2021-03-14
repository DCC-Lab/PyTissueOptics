import numpy as np
import random
from pytissueoptics import *


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


N = 100000


position = Vectors(N=N)
direction = Vectors([[0, 0, 1]]*N)
er = Vectors([[1, 0, 0]]*N)
weight = NumpyScalars(np.array([1.0]*N))

isAlive = True
while isAlive:
    theta = NumpyScalars.random(N=N) * 2 * np.pi
    phi = NumpyScalars.random(N=N) * 2 * np.pi

    d = NumpyScalars(getScatteringDistance(N, mu_s=30))

    er.rotateAround(direction, phi)
    direction.rotateAround(er, theta)
    position = position + direction*d
    weight *= 0.9
    isAlive = (weight.v > 0.001).any()
