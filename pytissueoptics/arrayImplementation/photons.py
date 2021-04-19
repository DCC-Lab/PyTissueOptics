from arrayImplementation.vectors import *
from arrayImplementation.scalars import *


class Photons:
    def __init__(self, positions=Vectors(N=1000), directions=Vectors([[0, 0, 1]]*1000)):
        self.N = len(positions)
        self.r = positions
        self.ez = directions
        self.er = self.ez.anyPerpendicular()
        self.wavelength = None
        self.weight = Scalars([1]*self.N)
        self.path = None

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


