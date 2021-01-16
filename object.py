import numpy as np
import matplotlib.pyplot as plt
from stats import *
from material import *
from vector import *
from photon import *

class Object:
    def __init__(self, material, stats=None):
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


    def propagateMany(self, source, showProgressEvery=100):
        for i, photon in enumerate(source):
            self.propagate(photon)
            if i  % showProgressEvery == 0:
                print("Photon {0}/{1}".format(i,source.maxCount) )
                if self.stats is not None:
                    self.stats.show2D(plane='xz', integratedAlong='y', title="{0} photons".format(i)) 

    def contains(self, position) -> bool:
        return True

    def absorbEnergy(self, photon):
        delta = photon.weight * self.material.albedo
        if self.stats is not None:
            self.stats.score(photon, delta)
        photon.decreaseWeightBy(delta)    

    def report(self):
        if self.stats is not None:
            self.stats.report()
            self.stats.show2D(plane='xz', integratedAlong='y', title="Final photons", realtime=False)
            #stats.show1D(axis='z', integratedAlong='xy', title="{0} photons".format(N), realtime=False)

class Cube(Object):
    def __init__(self, side, material, stats=None):
        super(Cube, self).__init__(material, stats)
        self.side = side
        self.origin = Vector(0,0,0)

    def contains(self, localPosition) -> bool:
        if abs(localPosition.x) > self.side/2:
            return False
        if abs(localPosition.y) > self.side/2:
            return False
        if abs(localPosition.z) > self.side/2:
            return False

        return True

class Sphere(Object):
    def __init__(self, radius, material, stats=None):
        super(Sphere, self).__init__(material, stats)
        self.radius = radius
        self.origin = Vector(0,0,0)

    def contains(self, localPosition) -> bool:
        if localPosition.abs() > self.radius:
            return False

        return True

