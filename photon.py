import numpy as np
from vector import *
from material import *
import time
import warnings

class Photon:
    def __init__(self):
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

    def transformToLocalCoordinates(self, origin):
        self.r = self.r - origin

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

class LaserSource:
    def __init__(self, position, direction, maxCount):
        self.position = position
        self.direction = direction
        self.maxCount = maxCount
        self.iteration = 0
        self._photons = []

    def __iter__(self):
        self.iteration = 0
        return self

    def __len__(self) -> int:
        return self.maxCount

    def __getitem__(self, item):
        if item < 0:
            # Convert negative index to positive (i.e. -1 == len - 1)
            item += self.maxCount

        if item < 0 or item >= self.maxCount:
            raise IndexError(f"Index {item} out of bound, min = 0, max {self.maxCount}.")

        start = time.monotonic()
        while len(self._photons) <= item:
            self._photons.append(self.randomPhoton())
            if time.monotonic() - start > 3:
                warnings.warn(f"Generating missing photon. This can take a few seconds.", UserWarning)

        return self._photons[item]

    def __next__(self) -> Photon:
        if self.iteration >= self.maxCount:
            raise StopIteration
        # This should be able to know if enough photon. If not enough, generate them
        photon = self[self.iteration]
        self.iteration += 1
        return photon

    def randomPhoton(self) -> Photon:
        p = Photon()
        p.r = self.position
        p.ez = self.direction
        return p
