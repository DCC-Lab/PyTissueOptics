import numpy as np
import matplotlib.pyplot as plt
from stats import *
from material import *
from vector import *
from photon import *
from surface import *

class Geometry:
    def __init__(self, material=None, stats=None):
        self.material = material
        self.stats = stats
        self.origin = Vector(0,0,0)
        self.surfaces = []

    def propagate(self, photon):
        photon.transformToLocalCoordinates(self.origin)

        while photon.isAlive and self.contains(photon.r):
            # Pick to scattering point
            d = self.material.getScatteringDistance(photon)
            scatteringPoint = photon.r + photon.ez * d

            if self.contains(scatteringPoint):
                # If the scatteringPoint is still inside, we simply move
                photon.moveBy(d)
 
                # Interact with volume
                delta = self.absorbEnergy(photon)
                self.scoreStepping(photon, delta)

                # Scatter within volume
                theta, phi = self.material.getScatteringAngles(photon)
                photon.scatterBy(theta, phi)
            else:
                # If the scatteringPoint is outside, we move to the surface
                (d, surface) = self.intersects(photon.r, d)
                photon.moveBy(d)

                # then we neglect reflections (for now), score, and leave
                self.scoreLeaving(photon, surface)
                break
            
            # And go again    
            photon.roulette()

        photon.transformFromLocalCoordinates(self.origin)

    def propagateMany(self, source, showProgressEvery=100):
        startTime = time.time()
        N = source.maxCount

        for i, photon in enumerate(source):
            self.propagate(photon)
            self.showProgress(i, maxCount=N , steps=showProgressEvery)

        elapsed = time.time() - startTime
        print('{0:.1f} s for {2} photons, {1:.1f} ms per photon'.format(elapsed, elapsed/N*1000, N))

    def intersects(self, position, distance) -> (float, Surface):
        return distance, None

    def contains(self, position) -> bool:
        """ This object is infinite. Subclasses override with their 
        specific geometry. """
        return True

    def absorbEnergy(self, photon) -> float:
        delta = photon.weight * self.material.albedo
        photon.decreaseWeightBy(delta)
        return delta

    def scoreStepping(self, photon, delta):
        if self.stats is not None:
            self.stats.score(photon, delta)

    def scoreLeaving(self, photon, surface):
        if surface is not None:
            surface.score(photon)

    def showProgress(self, i, maxCount, steps):
        if i  % steps == 0:
            print("Photon {0}/{1}".format(i, maxCount) )
            if self.stats is not None:
                self.stats.show2D(plane='xz', integratedAlong='y', title="{0} photons".format(i)) 

    def report(self):
        if self.stats is not None:
            self.stats.show2D(plane='xz', integratedAlong='y', title="Final photons", realtime=False)
            #stats.show1D(axis='z', integratedAlong='xy', title="{0} photons".format(N), realtime=False)

class Box(Geometry):
    def __init__(self, size, material, stats=None):
        super(Box, self).__init__(material, stats)
        self.size = size

    def contains(self, localPosition) -> bool:
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

class Layer(Geometry):
    def __init__(self, thickness, material, stats=None):
        super(Layer, self).__init__(material, stats)
        self.size = (1e6,1e6,thickness)

    def contains(self, localPosition) -> bool:
        if localPosition.z > self.size[2] or localPosition.z < 0:
            return False
        if abs(localPosition.y) > self.size[1]/2:
            return False
        if abs(localPosition.x) > self.size[0]/2:
            return False

        return True

class Sphere(Geometry):
    def __init__(self, radius, material, stats=None):
        super(Sphere, self).__init__(material, stats)
        self.radius = radius

    def contains(self, localPosition) -> bool:
        if localPosition.abs() > self.radius:
            return False

        return True

