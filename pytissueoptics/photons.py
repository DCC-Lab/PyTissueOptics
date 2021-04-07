from .vectors import *
from .scalars import *

class Photons:
    def __init__(self, positions=Vectors(1000), directions=Vectors([[0, 0, 1]]*1000)):
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

p = Photons()
