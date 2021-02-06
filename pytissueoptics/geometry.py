import numpy as np
import matplotlib.pyplot as plt
import signal
import sys
import time
from .surface import *
from .photon import *
from .source import *

class Geometry:
    def __init__(self, material=None, stats=None, label=""):
        self.material = material
        self.origin = Vector(0,0,0)
        self.stats = stats
        self.surfaces = []
        self.label = label
        self.inputWeight = 0

        self.epsilon = 1e-5
        self.startTime = None # We are not calculating anything

    def propagate(self, photon):
        photon.transformToLocalCoordinates(self.origin)
        self.scoreWhenStarting(photon)
        d = 0
        while photon.isAlive and self.contains(photon.r):
            # Pick distance to scattering point
            if d <= 0:
                d = self.material.getScatteringDistance(photon)
                
            distToPropagate, surface = self.nextExitInterface(photon.r, photon.ez, d)

            if surface is None:
                # If the scattering point is still inside, we simply move
                # Default is simply photon.moveBy(d) but other things 
                # would be here. Create a new material for other behaviour
                self.material.move(photon, d=d)
                d = 0
                # Interact with volume: default is absorption only
                # Default is simply absorb energy. Create a Material
                # for other behaviour
                delta = self.material.interactWith(photon)
                self.scoreInVolume(photon, delta)

                # Scatter within volume
                theta, phi = self.material.getScatteringAngles(photon)
                photon.scatterBy(theta, phi)
            else:
                # If the photon crosses an interface, we move to the surface
                self.material.move(photon, d=distToPropagate)

                # Determine if reflected or not with Fresnel coefficients
                if self.isReflected(photon, surface): 
                    # reflect photon and keep propagating
                    photon.reflect(surface)
                    photon.moveBy(d=1e-3) # Move away from surface
                    d -= distToPropagate
                else:
                    # transmit, score, and leave
                    photon.refract(surface)
                    self.scoreWhenCrossing(photon, surface)
                    photon.moveBy(d=1e-3) # We make sure we are out
                    break

            # And go again    
            photon.roulette()

        # Because the code will not typically calculate millions of photons, it is
        # inexpensive to keep all the propagated photons.  This allows users
        # to go through the list after the fact for a calculation of their choice
        self.scoreWhenFinal(photon)
        photon.transformFromLocalCoordinates(self.origin)
    
    def contains(self, position) -> bool:
        """ The base object is infinite. Subclasses override this method
        with their specific geometry. 

        It is important that this function be efficient: it is called
        very frequently. See implementations for Box, Sphere and Layer
        """
        return True

    def nextExitInterface(self, position, direction, distance) -> (float, Surface): 
        """ Is this line segment from position to distance*direction leaving
        the object through any surface elements? Valid only from inside the object.
        
        This function is a very general function
        to find if a photon will leave the object.  `contains` is called
        repeatedly, is geometry-specific, and must be high performance. 
        It may be possible to write a specialized version for a subclass,
        but this version will work by default for all objects and is 
        surprisingly efficient.
        """

        finalPosition = Vector.fromScaledSum(position, direction, distance)
        if self.contains(finalPosition):
            return distance, None

        wasInside = True
        finalPosition = Vector(position) # Copy
        delta = 0.5*distance

        while abs(delta) > 0.00001:
            finalPosition += delta * direction
            isInside = self.contains(finalPosition)
            
            if isInside != wasInside:
                delta = -delta * 0.5

            wasInside = isInside

        for surface in self.surfaces:
            if surface.normal.dot(direction) > 0:
                if surface.contains(finalPosition):
                    return (finalPosition-position).abs(), surface

        return distance, None 


    def nextEntranceInterface(self, position, direction, distance) -> (float, Surface):
        """ Is this line segment from position to distance*direction crossing
        any surface elements of this object? Valid from outside the object.

        This will be very slow: going through all elements to check for
        an intersection is abysmally slow
        and increases linearly with the number of surface elements
        There are tons of strategies to improve this (axis-aligned boxes,
        oriented boxes but most importantly KDTree and OCTrees).
        It is not done here, we are already very slow: what's more slowdown
        amongst friends? """

        minDistance = distance
        intersectSurface = None
        for surface in self.surfaces:
            if direction.dot(surface.normal) >= 0:
                # Parallel or outward, does not apply
                continue
            # Going inward, against surface normal
            isIntersecting, distanceToSurface = surface.intersection(position, direction, distance)
            if isIntersecting and distanceToSurface < minDistance:
                intersectSurface = surface
                minDistance = distanceToSurface

        return minDistance, intersectSurface

    def isReflected(self, photon, surface) -> bool:
        R = photon.fresnelCoefficient(surface)
        if np.random.random() < R:
            return True
        return False

    def scoreWhenStarting(self, photon):
        if self.stats is not None:
            self.stats.scoreWhenStarting(photon)

    def scoreInVolume(self, photon, delta):
        if self.stats is not None:
            self.stats.scoreInVolume(photon, delta)

    def scoreWhenCrossing(self, photon, surface):
        if self.stats is not None:
            self.stats.scoreWhenCrossing(photon, surface)

    def scoreWhenFinal(self, photon):
        if self.stats is not None:
            self.stats.scoreWhenFinal(photon)

    def report(self):
        print("Geometry and material")
        print("=====================")
        print(self)

        print("\nPhysical quantities")
        print("=====================")
        if self.stats is not None:
            totalWeightAcrossAllSurfaces = 0
            for i, surface in enumerate(self.surfaces):
                totalWeight = self.stats.totalWeightCrossingPlane(surface)
                print("Transmittance [{0}] : {1:.1f}% ".format(surface, 100*totalWeight/self.stats.inputWeight))
                print("Transmittance [{0}] : {1:.1f}% of total power".format(surface, 100*totalWeight/self.stats.photonCount))
                totalWeightAcrossAllSurfaces += totalWeight

            print("Absorbance : {0:.1f}%".format(100*self.stats.totalWeightAbsorbed()/self.stats.inputWeight))
            print("Absorbance : {0:.1f}%".format(100*self.stats.totalWeightAbsorbed()/self.stats.photonCount))

            totalCheck = totalWeightAcrossAllSurfaces + self.stats.totalWeightAbsorbed()
            print("Absorbance + Transmittance = {0:.1f}%".format(100*totalCheck/self.stats.inputWeight))

            self.stats.showEnergy2D(plane='xz', integratedAlong='y', title="Final photons", realtime=False)
            if len(self.surfaces) != 0:
                self.stats.showSurfaceIntensities(self.surfaces)

    def __repr__(self):
        return "{0}".format(self)

    def __str__(self):
        string = "'{0}' {2} with surfaces {1}\n".format(self.label, self.surfaces, self.origin)
        string += "{0}".format(self.material)
        return string

class Box(Geometry):
    def __init__(self, size, material, stats=None, label="Box"):
        super(Box, self).__init__(material, stats, label)
        self.size = size
        self.surfaces = [-XYPlane(atZ=-self.size[2]/2, description="Front"),
                          XYPlane(atZ= self.size[2]/2, description="Back"),
                         -YZPlane(atX=-self.size[0]/2, description="Left"),
                          YZPlane(atX= self.size[0]/2, description="Right"), 
                         -ZXPlane(atY=-self.size[1]/2, description="Bottom"),
                          ZXPlane(atY= self.size[1]/2, description="Top")]

    def contains(self, localPosition) -> bool:
        if abs(localPosition.z) > self.size[2]/2 + self.epsilon:
            return False
        if abs(localPosition.y) > self.size[1]/2 + self.epsilon:
            return False
        if abs(localPosition.x) > self.size[0]/2 + self.epsilon:
            return False

        return True

class Cube(Box):
    def __init__(self, side, material, stats=None, label="Cube"):
        super(Cube, self).__init__((side,side,side), material, stats, label)

class Layer(Geometry):
    def __init__(self, thickness, material, stats=None, label="Layer"):
        super(Layer, self).__init__(material, stats, label)
        self.thickness = thickness
        self.surfaces = [-XYPlane(atZ= 0, description="Front"),
                          XYPlane(atZ= self.thickness, description="Back")]

    def contains(self, localPosition) -> bool:
        if localPosition.z < -self.epsilon:
            return False
        if localPosition.z > self.thickness + self.epsilon:
            return False

        return True

    def nextExitInterface(self, position, direction, distance) -> (float, Surface): 
        finalPosition = Vector.fromScaledSum(position, direction, distance)
        if self.contains(finalPosition):
            return distance, None

        if direction.z > 0:
            d = (self.thickness - position.z)/direction.z
            return d, self.surfaces[0]
        elif direction.z < 0:
            d = - position.z/direction.z
            return d, self.surfaces[1]

        return distance, None

    def stack(self, layer):
        return

class SemiInfiniteLayer(Geometry):
    """ This class is actually a bad idea: the photons don't exit
    on the other side and will just take a long time to propagate.
    It is better to use a finite layer with a thickness a bit larger
    than what you are interested in."""

    def __init__(self, material, stats=None, label="Semi-infinite layer"):
        super(SemiInfiniteLayer, self).__init__(material, stats, label)
        self.surfaces = [ -XYPlane(atZ= 0, description="Front")]

    def contains(self, localPosition) -> bool:
        if localPosition.z < -self.epsilon:
            return False

        return True

    def nextExitInterface(self, position, direction, distance) -> (float, Surface): 
        finalPosition = position + distance*direction
        if self.contains(finalPosition):
            return distance, None

        if direction.z < 0:
            d = - position.z/direction.z
            return d, self.surfaces[0]

        return distance, None


# class Sphere(Geometry):
#     def __init__(self, radius, material, stats=None, label="Sphere"):
#         super(Sphere, self).__init__(position, material, stats, label)
#         self.radius = radius

#     def contains(self, localPosition) -> bool:
#         if localPosition.abs() > self.radius + self.epsilon:
#             return False

#         return True

class KleinBottle(Geometry):
    def __init__(self, position, material, stats=None):
        super(KleinBottle, self).__init__(position, material, stats)

    def contains(self, localPosition) -> bool:
        raise NotImplementedError()
