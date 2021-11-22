from pytissueoptics import *
from pytissueoptics.vector import Vector, UnitVector, zHat
from pytissueoptics.vectors import Vectors
import numpy as np


class Photon:
    def __init__(self, position=None, direction=None, weight=1.0, origin=Vector(0,0,0), currentGeometry=None):
        if position is not None:
            self.r = Vector(position)  # local coordinate position
        else:
            self.r = Vector(0, 0, 0)

        if direction is not None:
            self.ez = UnitVector(direction)  # Propagation direction vector
        else:
            self.ez = UnitVector(zHat)  # Propagation direction vector

        self.er = UnitVector(0, 1, 0)

        if not self.er.isPerpendicularTo(self.ez):
            self.er = self.ez.anyPerpendicular()

        self.origin = Vector(origin)
        # We don't need to keep el, because it is obtainable from ez and er

        self.weight = weight
        self.wavelength = None
        self.path = None

         # The global coordinates of the local origin
        self.currentGeometry = currentGeometry

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
    def isAlive(self) -> bool:
        return self.weight > 0

    @property
    def isDead(self) -> bool:
        return self.weight == 0

    def keepPathStatistics(self):
        self.path = [Vector(self.r)]  # Will continue every move

    def transformToLocalCoordinates(self, origin):
        self.r = self.r - origin
        self.origin = origin

    def transformFromLocalCoordinates(self, origin):
        self.r = self.r + origin
        self.origin = Vector(0, 0, 0)

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


class NativePhotons:
    def __init__(self, array=None, positions=None, directions=None, N=0):
        self.iteration = None
        self._photons = []
        if array is not None:
            self._photons = array
        elif not None in (positions, directions):
            positions = Vectors(positions)
            directions = Vectors(directions)
            print(len(positions), len(directions))
            for position, direction in zip(positions, directions):
                self._photons.append(Photon(position=position, direction=direction))
        elif N > 0 and isinstance(positions, Vector):
            for i in range(N):
                self._photons.append(Photon(position=positions, direction=directions))

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

    @property
    def isEmpty(self):
        if len(self._photons) == 0:
            return True
        else:
            return False

    @property
    def isRowOptimized(self):
        return True

    @property
    def isColumnOptimized(self):
        return False

    def append(self, photon):
        if isinstance(photon, Photon):
            self._photons.append(photon)
        elif isinstance(photon, NativePhotons):
            for p in photon:
                self._photons.append(p)

    def remove(self, somePhotons):
        for photon in somePhotons:
            self._photons.remove(photon)

    def livePhotonsInGeometry(self, geometry):
        return Photons(
            list(filter(lambda photon: (photon.currentGeometry == geometry) and photon.isAlive, self._photons)))

    def areAllDead(self) -> bool:
        for photon in self._photons:
            if photon.isAlive:
                return False

        return True

    def deadCount(self):
        count = 0
        for photon in self._photons:
            if photon.isDead:
                count += 1
        return count

    def liveCount(self):
        count = 0
        for photon in self._photons:
            if photon.isAlive:
                count += 1
        return count

    def areReflected(self, interfaces):
        areReflected = [interface.isReflected() for photon, interface in zip(self._photons, interfaces)]

        reflectedPhotons = Photons([photon for photon, isReflected in zip(self._photons, areReflected) if isReflected])
        reflectedInterfaces = FresnelIntersects([intersect for intersect in interfaces if intersect.isReflected()])

        transmittedPhotons = Photons(
            [photon for photon, isReflected in zip(self._photons, areReflected) if not isReflected])
        transmittedInterfaces = FresnelIntersects([intersect for intersect in interfaces if not intersect.isReflected()])

        return (reflectedPhotons, reflectedInterfaces),  (transmittedPhotons, transmittedInterfaces)


    def transformToLocalCoordinates(self, origin):
        for photon in self._photons:
            photon.transformToLocalCoordinates(origin)

    def transformFromLocalCoordinates(self, origin):
        for photon in self._photons:
            photon.transformFromLocalCoordinates(origin)

    def moveBy(self, distances):
        try:
            for photon, d in zip(self._photons, distances):
                photon.moveBy(d)
        except TypeError as err:
            for photon in self._photons:
                photon.moveBy(distances)

        # I don't understand why this does not work:
        # map(lambda photon, d: photon.moveBy(d), zip(self._photons, distances))

    def scatterBy(self, thetas, phis):
        for photon, theta, phi in zip(self._photons, thetas, phis):
            photon.scatterBy(theta, phi)

    def decreaseWeight(self, albedo):
        weightLoss = []
        for photon in self._photons:
            delta = albedo * photon.weight
            photon.decreaseWeightBy(delta)
            weightLoss.append(delta)

        return Scalars(weightLoss)

        # I don't understand why this does not work:
        # map(lambda photon : photon.decreaseWeightBy(albedo * photon.weight), self._photons)

    def decreaseWeightBy(self, deltas):
        map(lambda photon, delta: photon.decreaseWeightBy(delta), self._photons, deltas)

    def deflect(self):
        pass

    def reflect(self, interfaces):
        for photon, interface in zip(self._photons, interfaces):
            photon.reflect(interface)
            photon.moveBy(1e-6)

    def refract(self, interfaces):
        for photon, interface in zip(self._photons, interfaces):
            photon.refract(interface)
            photon.moveBy(1e-6)
            photon.currentGeometry = interface.geometry

    def roulette(self):
        for photon in self._photons:
            photon.roulette()


class ArrayPhotons:
    def __init__(self, array=None, positions=None, directions=None,):
        self.r = Vectors(positions)
        self.ez = Vectors(directions)
        if not self.ez.isEmpty:
            self.er = self.ez.anyPerpendicular()
            N = len(self.r)
        else:
            self.er = Vectors()
            N = 0

        self.wavelength = None
        self.weight = Scalars([1] * N)
        self.path = None
        self.origin = Vectors(N=N)

        if array is not None and type(array) == list and isinstance(array[0], Photon):
            for photon in array:
                self.append(photon)

        self._iteration = 0
        self.maskedPhotons = None
        self.mask = None

    def __len__(self):
        if not self.isEmpty:
            return len(self.r)
        else:
            return 0

    def __getitem__(self, index):
        return Photon(position=self.r[index], direction=self.ez[index], weight=self.weight[index], origin=self.origin[index])

    def __setitem__(self, index, photon):
        self.r[index] = photon.r
        self.ez[index] = photon.ez
        self.er[index] = photon.er
        self.weight[index] = photon.weight
        self.origin[index] = photon.origin

    def __iter__(self):
        self._iteration = 0
        return self

    def __next__(self):
        if self._iteration < len(self):
            result = self[self._iteration]
            self._iteration += 1
            return result
        else:
            raise StopIteration

    @property
    def isEmpty(self):
        if len(self.r) == 0:
            return True
        else:
            return False

    @property
    def isRowOptimized(self):
        return False

    @property
    def isColumnOptimized(self):
        return True

    def append(self, photon):
        if isinstance(photon, Photons):
            if photon.isEmpty:
                return

        self.r.append(photon.r)
        self.ez.append(photon.ez)
        self.er.append(photon.er)
        self.weight.append(photon.weight)
        self.origin.append(photon.origin)


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
        self.origin = Vectors([origin]*len(self))

    def transformFromLocalCoordinates(self, origin):
        self.r = self.r + Vectors(origin)
        self.origin = Vectors([0, 0, 0]*len(self))

    def moveBy(self, d):
        if not self.isEmpty:
            self.r.addScaled(self.ez, d)

    def scatterBy(self, theta, phi):
        if not self.isEmpty:
            self.er.rotateAround(self.ez, phi)
            self.ez.rotateAround(self.er, theta)

    def decreaseWeight(self, albedo):
        deltas = []
        if not self.isEmpty:
            deltas = albedo * self.weight
            self.weight -= deltas
            self.weight.conditional_lt(0, 0, self.weight.v)
            # FIXME: Porblem with deltas if it is negative, they wont be accurate anymore.
        return deltas

    def decreaseWeightBy(self, deltas):
        if not self.isEmpty:
            self.weight -= deltas
            self.weight.conditional_lt(0, 0, self.weight.v)

    def roulette(self):
        if not self.isEmpty:
            chance = 0.1
            rouletteMask = self.weight <= 1e-4
            photonsKillMask = (Scalars.random(len(self))) > chance
            photonsKillMask = rouletteMask.logical_and(photonsKillMask)
            self.removePhotonsWeights(photonsKillMask)

    def removePhotonsWeights(self, killMask):
        self.weight = self.weight * ~killMask

    def deflect(self):
        pass

    def reflect(self, interfaces):
        if not self.isEmpty:
            self.ez.rotateAround(interfaces.incidencePlane, interfaces.reflectionDeflection)
            self.moveBy(1e-6)

    def areReflected(self, interfaces):
        reflectedPhotons = Photons()
        reflectedInterfaces = FresnelIntersects()
        transmittedPhotons = Photons()
        transmittedInterfaces = FresnelIntersects()
        for i, p in enumerate(self):
            if interfaces[i].isReflected():
                reflectedPhotons.append(p)
                reflectedInterfaces.append(interfaces[i])
            else:
                transmittedPhotons.append(p)
                transmittedInterfaces.append(interfaces[i])

        return (reflectedPhotons, reflectedInterfaces), (transmittedPhotons, transmittedInterfaces)

    def refract(self, interfaces):
        if not self.isEmpty:
            self.ez.rotateAround(interfaces.incidencePlane, interfaces.refractionDeflection)

    def photonsTemporaryMasking(self, mask):
        self.mask = mask

    def unMask(self, newPhotons):
        pass


Photons = NativePhotons
