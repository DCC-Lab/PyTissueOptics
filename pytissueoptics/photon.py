import numpy as np
import time
import warnings
from .vector import *
from .vectors import *
from math import acos, asin, cos, sin, atan, tan, sqrt, pi
import random


class Photon:
    def __init__(self, position=None, direction=None, weight=None):
        if position is not None:
            self.r = Vector(position) # local coordinate position
        else:
            self.r = Vector(0, 0, 0)

        if direction is not None:
            self.ez = UnitVector(direction)  # Propagation direction vector
        else:
            self.ez = UnitVector(zHat)  # Propagation direction vector

        self.er = UnitVector(0, 1, 0)

        if not self.er.isPerpendicularTo(self.ez):
            self.er = self.ez.anyPerpendicular()

        # We don't need to keep el, because it is obtainable from ez and er

        if weight is None:
            self.weight = 1.0
        else:
            self.weight = weight

        self.wavelength = None
        self.path = None
        self.origin = Vector(0, 0, 0) # The global coordinates of the local origin

    @property
    def localPosition(self):
        return self.r

    @property
    def globalPosition(self):
        return self.r + self.origin
    
    @property
    def el(self) -> UnitVector:
        return self.ez.cross(self.er) 

    @property
    def isAlive(self) -> bool :
        return self.weight > 0

    @property
    def isDead(self) -> bool :
        return self.weight == 0

    def keepPathStatistics(self):
        self.path = [Vector(self.r)]  # Will continue every move

    def transformToLocalCoordinates(self, origin):
        self.r = self.r - origin
        self.origin = origin

    def transformFromLocalCoordinates(self, origin):
        self.r = self.r + origin
        self.origin = Vector(0,0,0)

    def moveBy(self, d):
        self.r.addScaled(self.ez, d)
        
        if self.path is not None:
            self.path.append(Vector(self.r))  # We must make a copy

    def scatterBy(self, theta, phi):
        self.er.rotateAround(self.ez, phi)
        self.ez.rotateAround(self.er, theta)

    def decreaseWeightBy(self, delta):
        self.weight -= delta
        if self.weight < 0:
            self.weight = 0

    def deflect(self, deflectionAngle, incidencePlane):
        self.ez.rotateAround(incidencePlane, deflectionAngle)

    def reflect(self, intersection):
        self.ez.rotateAround(intersection.incidencePlane, intersection.reflectionDeflection)

    def refract(self, intersection):
        """ Refract the photon when going through surface.  The surface
        normal in the class Surface always points outward for the object.
        Hence, to simplify the math, we always flip the normal to have 
        angles between -90 and 90.

        Since having n1 == n2 is not that rare, if that is the case we 
        know there is no refraction, and we simply return.
        """

        self.ez.rotateAround(intersection.incidencePlane, intersection.refractionDeflection)

    def roulette(self):
        chance = 0.1
        if self.weight >= 1e-4 or self.weight == 0:
            return
        elif np.random.random() < chance:
            self.weight /= chance
        else:
            self.weight = 0


"""
Photons is essentially an abstract class, but a basic implementation is provided.  It offers a model for
all children classes.

"""
class Photons:
    def __init__(self, array=None, N=0, position=None, direction=None):
        self.iteration = None
        self._photons = []
        if array is not None:
            self._photons = array
        else:
            for i in range(N):
                self._photons.append(Photon(position=position, direction=direction))

    def __getitem__(self, item):
        return self._photons[item]

    def __len__(self):
        return len(self._photons)

    def __iter__(self):
        self.iteration = 0
        return self

    def __next__(self) -> Photon:
        if self._photons is None:
            raise StopIteration

        if self.iteration < len(self):  # We really want to use len(self) to be compatible with CompactRays
            photon = self[self.iteration]  # Again we want to use __getitem__ for self for CompactRays
            self.iteration += 1
            return photon

        raise StopIteration

    def append(self, photon):
        self._photons.append(photon)

    @property
    def areAllDead(self) -> bool:
        return np.array([ photon.isDead for photon in self._photons]).all()

    def areReflected(self, interfaces):
        areReflected = [ photon.isReflected(interface) for photon, interface in zip(self._photons, interfaces) ]

        reflectedPhotons = Photons([ photon for photon, isReflected in zip(self._photons, areReflected) if isReflected ])
        transmittedPhotons = Photons([ photon for photon, isReflected in zip(self._photons, areReflected) if not isReflected ])
        return (reflectedPhotons, transmittedPhotons)

    def transformToLocalCoordinates(self, origin):
        map(lambda photon: photon.transformToLocalCoordinates(origin), self._photons)

    def transformFromLocalCoordinates(self, origin):
        map(lambda photon: photon.transformFromLocalCoordinates(origin), self._photons)

    def moveBy(self, d):
        map(lambda photon: photon.moveBy(d), self._photons)

    def scatterBy(self, thetas, phis):
        map(lambda photon, theta, phi: photon.scatterBy(theta, phi), self._photons, thetas, phis)

    def decreaseWeight(self, albedo):
        map(lambda photon, delta: photon.decreaseWeightBy(albedo*photon.weight()), self._photons)

    def decreaseWeightBy(self, deltas):
        map(lambda photon, delta: photon.decreaseWeightBy(delta), self._photons, deltas)

    def deflect(self):
        pass

    def reflect(self, interfaces):
        map(lambda photon, interface: photon.reflect(intersect), self._photons, interfaces)

    def refract(self, interfaces):
        map(lambda photon, interface: photon.refract(intersect), self._photons, interfaces)

    def roulette(self):
        map(lambda photon: photon.roulette(), self._photons)


class NativePhotons(Photons):
    def __init__(self, positions=Vectors(N=1000), directions=Vectors([zHat] * 1000)):
        self.N = len(positions)
        self.r = positions
        self.ez = directions
        self.er = self.ez.anyPerpendicular()
        self.wavelength = None
        self.weight = Scalars([1] * self.N)
        self.path = None
        self.origin = Vectors(N=self.N)
        self._iteration = 0


    def __len__(self):
        return self.N


    def __getitem__(self, index):
        return Photon(position=self.r[index], direction=self.ez[index], weight=self.weight[index])


    def __setitem__(self, index, photon):
        self.r[index] = photon.r
        self.z[index] = photon.z
        self.ez[index] = photon.ez
        self.er[index] = photon.er
        self.weight[index] = photon.weight
        self.origin[index] = photon.origin


    def __iter__(self):
        self._iteration = 0
        return self


    def __next__(self):
        if self._iteration < self.N:
            result = self[self._iteration]
            self._iteration += 1
            return result
        else:
            raise StopIteration

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
    def areAllDead(self):
        return all(self.weight == 0)

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

    def decreaseWeightBy(self, deltas):
        self.weight -= deltas
        self.weight.conditional_lt(0, 0, self.weight.v)

    def roulette(self):
        chance = 0.1
        rouletteMask = self.weight <= 1e-4
        photonsKillMask = (Scalars.random(self.N)) > chance
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