import numpy as np
import matplotlib.pyplot as plt
from stats import *
from material import *
from vector import *
from photon import *

class Geometry:
    def __init__(self, material=None, stats=None):
        self.material = material
        self.stats = stats
        self.origin = Vector(0,0,0)

    def propagate(self, photon):
        photon.transformToLocalCoordinates(self.origin)

        while photon.isAlive and self.contains(photon.r):
            d = self.material.getScatteringDistance(photon)
            (theta, phi) = self.material.getScatteringAngles(photon)
            photon.moveBy(d)
            photon.scatterBy(theta, phi)
            self.absorbEnergy(photon)
            photon.roulette()

        photon.transformFromLocalCoordinates(self.origin)

    def propagateMany(self, source, showProgressEvery=100):
        for i, photon in enumerate(source):
            self.propagate(photon)
            self.showProgress(i, maxCount=source.maxCount, steps=showProgressEvery)

    def contains(self, position) -> bool:
        """ This object is infinite. Subclasses override with their 
        specific geometry. """
        return True

    def absorbEnergy(self, photon):
        delta = photon.weight * self.material.albedo
        if self.stats is not None:
            self.stats.score(photon, delta)
        photon.decreaseWeightBy(delta)    

    def showProgress(self, i, maxCount, steps):
        if i  % steps == 0:
            print("Photon {0}/{1}".format(i, maxCount) )
            if self.stats is not None:
                self.stats.show2D(plane='xz', integratedAlong='y', title="{0} photons".format(i)) 

    def report(self):
        if self.stats is not None:
            self.stats.report()
            self.stats.show2D(plane='xz', integratedAlong='y', title="Final photons", realtime=False)
            #stats.show1D(axis='z', integratedAlong='xy', title="{0} photons".format(N), realtime=False)

class Box(Geometry):
    def __init__(self, size, material, stats=None):
        super(Box, self).__init__(material, stats)
        self.size = size

    def contains(self, localPosition) -> bool:
        # We check Z first because ZLayer will benefit from this
        if abs(localPosition.z) > self.size[2]/2:
            return False
        if abs(localPosition.y) > self.size[1]/2:
            return False
        if abs(localPosition.x) > self.size[0]/2:
            return False

        return True

class Cube(Geometry):
    def __init__(self, side, material, stats=None):
        super(Cube, self).__init__(material, stats)
        self.size = (side,side,side)

class ZLayer(Geometry):
    def __init__(self, thickness, material, stats=None):
        super(ZLayer, self).__init__(material, stats)
        self.size = (1e6,1e6,thickness)

class Sphere(Geometry):
    def __init__(self, radius, material, stats=None):
        super(Sphere, self).__init__(material, stats)
        self.radius = radius

    def contains(self, localPosition) -> bool:
        if localPosition.abs() > self.radius:
            return False

        return True

