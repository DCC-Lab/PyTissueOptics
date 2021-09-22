from arrayImplementation.vectors import Vectors
from arrayImplementation.scalars import Scalars
import numpy as np


class Photons:
    def __init__(self, positions=Vectors(N=1000), directions=Vectors([[0, 0, 1]]*1000)):
        self.N = len(positions)
        self.r = positions
        self.ez = directions
        self.er = self.ez.anyPerpendicular()
        self.wavelength = None
        self.weight = Scalars([1]*self.N)
        self.path = None
        self.origin = Vectors([0, 0, 0]*self.N)

    @property
    def localPosition(self):
        return self.r

    @property
    def globalPosition(self):
        return self.r + self.origin

    @property
    def el(self):
        return self.ez.cross(self.er)

    @property
    def isAlive(self):
        return self.weight > 0

    def transformToLocalCoordinates(self, origin):
        self.r = self.r - origin

    def transformFromLocalCoordinates(self, origin):
        self.r = self.r + origin

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
        rouletteMask = (self.weight <= 1e-4)
        photonsKillMask = Scalars().random(self.N) > chance
        photonsKillMask = rouletteMask.logical_and(photonsKillMask)
        self.removePhotonsWeights(photonsKillMask)

    def removePhotonsWeights(self, killMask):
        self.weight = self.weight * ~killMask

    def deflect(self):
        pass

    def reflect(self):
        pass

    def refract(self):
        pass




