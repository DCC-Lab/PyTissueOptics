import numpy as np
import time
import warnings
from .vector import *
from .vectors import *
from math import acos, asin, cos, sin, atan, tan, sqrt, pi
import random


class Photon:
    def __init__(self, position=Vector(0, 0, 0), direction=UnitVector(0, 0, 1)):
        self.r = Vector(position)
        self.ez = UnitVector(direction)  # Propagation direction vector
        self.er = UnitVector(0, 1, 0)

        if not self.er.isPerpendicularTo(self.ez):
            self.er = self.ez.anyPerpendicular()

        self.wavelength = None
        # We don't need to keep el, because it is obtainable from ez and er
        self.weight = 1.0
        self.path = None

    @property
    def el(self) -> UnitVector:
        return self.ez.cross(self.er) 

    @property
    def isAlive(self) -> bool :
        return self.weight > 0

    def keepPathStatistics(self):
        self.path = [Vector(self.r)]  # Will continue every move

    def transformToLocalCoordinates(self, origin):
        self.r = self.r - origin

    def transformFromLocalCoordinates(self, origin):
        self.r = self.r + origin

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


class Photons:
    def __init__(self, position=None, direction=None):
        n = len(position)
        self.r = Vectors(position)
        self.ez = Vectors(direction)  # Propagation direction vector
        self.er = Vectors( [Vector(0, 1, 0)]*n )

        self.er.select(logicalNot(self.er.isPerpendicularTo(self.ez)))
        self.er.replaceSelected(self.ez.anyPerpendicular())
        self.er.selectAll()

        # We don't need to keep el, because it is obtainable from ez and er
        self.weight = Scalars([1.0]*n)
        self.path = None

    @property
    def el(self) -> Vectors:
        return self.ez.cross(self.er)

    @property
    def isAlive(self) -> bool:
        return (self.weight.v > 0).all()

    def transformToLocalCoordinates(self, origin):
        self.r = self.r - [origin]* len(self.r)

    def transformFromLocalCoordinates(self, origin):
        self.r = self.r + [origin]* len(self.r)

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
        n = len(self.position)

        chance = 0.1

        rouletteMask = (self.weight.v < 1e-4)

        live = (Scalars.random(n) > chance)
        die = logicalNot(live)
        factor = dontTouchMask + rouletteMask * live/chance

