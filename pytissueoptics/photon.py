import numpy as np
import time
import warnings
from .vector import *
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

    def angleOfIncidence(self, surface) -> (float, Vector):
        planeOfIncidenceNormal = self.ez.normalizedCrossProduct(surface.normal)
        return self.ez.angleWith(surface.normal, righthand=planeOfIncidenceNormal), planeOfIncidenceNormal

    def fresnelCoefficient(self, surface):
        n1 = surface.indexInside
        n2 = surface.indexOutside

        thetaIn, planeOfIncidenceNormal = self.angleOfIncidence(surface)
        if math.sin(thetaIn)*n1/n2 > 1:
            return 1

        R = (n2-n1)/(n1+n2)
        return R*R

    def reflect(self, surface):
        planeOfIncidenceNormal = self.ez.normalizedCrossProduct(surface.normal)
        thetaIn = self.ez.angleWith(surface.normal, righthand=planeOfIncidenceNormal)

        self.ez.rotateAround(planeOfIncidenceNormal, 2*thetaIn-np.pi)

    def refract(self, surface):
        planeOfIncidenceNormal = self.ez.normalizedCrossProduct(surface.normal)
        if planeOfIncidenceNormal.norm() == 0:
            # Normal incidence
            return

        thetaIn = self.ez.angleWith(surface.normal, righthand=planeOfIncidenceNormal)

        if self.ez.dot(surface.normal) > 0:
            n1 = surface.indexInside
            n2 = surface.indexOutside
        else:
            n1 = surface.indexOutside
            n2 = surface.indexInside

        thetaOut = math.asin(n1*math.sin(thetaIn)/n2)

        self.ez.rotateAround(planeOfIncidenceNormal, thetaOut-thetaIn)

    def roulette(self):
        chance = 0.1
        if self.weight >= 1e-4 or self.weight == 0:
            return
        elif np.random.random() < chance:
            self.weight /= chance
        else:
            self.weight = 0
