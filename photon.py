import numpy as np
from vector import *
from material import *

class Photon:
    def __init__(self):
        self.reset()

    def reset(self):
        self.r = Vector(0,0,0)
        self.ez = UnitVector(0,0,1) # Propagation direction vector
        self.er = UnitVector(0,1,0) # Vector perpendicular to scattering plane and ez
        # We don't need to keep el, because it is obtainable from ez and er
        self.weight = 1.0
        self.uniqueId = np.random.randint(1<<31) # This is dumb but it works for now.

    @property
    def el(self) -> UnitVector:
        return self.ez.cross(self.er) 

    @property
    def isAlive(self) -> bool :
        return self.weight > 0

    def moveBy(self, d):
        self.r.addScaled(self.ez, d)

    def scatterBy(self, theta, phi):
        self.er.rotateAround(self.ez, phi)
        self.ez.rotateAround(self.er, theta)

    def decreaseWeightBy(self, delta):
        self.weight -= delta
        if self.weight < 0:
            self.weight = 0

    def roulette(self):
        chance = 0.1
        if self.weight >= 1e-4 or self.weight == 0:
            return
        elif np.random.random() < chance:
            self.weight /= chance
        else:
            self.weight = 0
