import numpy as np
import random
from pytissueoptics import *

def getScatteringDistance(N, mu_s):
    scalar = [] 
    for i in range(N):
        scalar.append( random.random())
    return scalar

def getScatteringAngle(N):
    theta = [] 
    phi = [] 
    for i in range(N):
        theta.append( random.random()*2*np.pi)
        phi.append( random.random()*2*np.pi)
    return theta, phi


N = 10000
position = Vectors(N=N)
direction = Vectors([zHat]*N)
er = Vectors([xHat]*N)
weight = Scalars([1.0]*N)

isAlive = True
while isAlive:
    theta, phi = getScatteringAngle(N)
    d = getScatteringDistance(N, mu_s=30)

    er.rotateAround(direction, phi)
    direction.rotateAround(er, theta)
    position = position + direction*d
    weight *= 0.9
    isAlive = (weight > 0.001).any()
