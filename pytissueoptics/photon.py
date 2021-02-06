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

    def fresnelCoefficient(self, surface):
        """ Fresnel reflection coefficient, directly from MCML code in 
        Wang, L-H, S.L. Jacques, L-Q Zheng: 
        MCML - Monte Carlo modeling of photon transport in multi-layered
        tissues. Computer Methods and Programs in Biomedicine 47:131-146, 1995. 

        """
        if self.ez.dot(surface.normal) > 0:
            normal = surface.normal
            n1 = surface.indexInside
            n2 = surface.indexOutside
        else:
            normal = -surface.normal
            n1 = surface.indexOutside
            n2 = surface.indexInside

        if n1 == n2:
            return 0

        planeOfIncidenceNormal = self.ez.normalizedCrossProduct(normal)
        if planeOfIncidenceNormal.isNull:
            R = (n2-n1)/(n2+n1)
            return R*R

        thetaIn = self.ez.angleWith(normal, axis=planeOfIncidenceNormal)

        sa1 = math.sin(thetaIn)
        if sa1*n1/n2 > 1:
            return 1
        sa2 = sa1*n1/n2
        ca1 = math.sqrt(1-sa1*sa1)
        ca2 = math.sqrt(1-sa2*sa2)

        cap = ca1*ca2 - sa1*sa2 # c+ = cc - ss.
        cam = ca1*ca2 + sa1*sa2 # c- = cc + ss. 
        sap = sa1*ca2 + ca1*sa2 # s+ = sc + cs. 
        sam = sa1*ca2 - ca1*sa2 # s- = sc - cs. 
        r = 0.5*sam*sam*(cam*cam+cap*cap)/(sap*sap*cam*cam); 
        return r

    def reflect(self, surface):
        planeOfIncidenceNormal = self.ez.normalizedCrossProduct(surface.normal)
        thetaIn = self.ez.angleWith(surface.normal, axis=planeOfIncidenceNormal)

        self.ez.rotateAround(planeOfIncidenceNormal, 2*thetaIn-np.pi)

    def refract(self, surface):
        """ Refract the photon when going through surface.  The surface
        normal in the class Surface always points outward for the object.
        Hence, to simplify the math, we always flip the normal to have 
        angles between -90 and 90.

        Since having n1 == n2 is not that rare, if that is the case we 
        know there is no refraction, and we simply return.
        """

        if surface.indexInside == surface.indexOutside:
            return

        normal = None

        if self.ez.dot(surface.normal) > 0:
            # Going out
            n1 = surface.indexInside
            n2 = surface.indexOutside
            normal = surface.normal
        else:
            # Going in, we flip normal
            n1 = surface.indexOutside
            n2 = surface.indexInside
            normal = -surface.normal

        planeOfIncidenceNormal = self.ez.normalizedCrossProduct(normal)
        if planeOfIncidenceNormal.norm() == 0:
            # Normal incidence
            return

        thetaIn = self.ez.angleWith(normal, axis=planeOfIncidenceNormal)
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
