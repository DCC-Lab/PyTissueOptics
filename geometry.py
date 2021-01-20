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
        self.origin = Vector(0,0,0)
        self.stats = stats

    @property
    def surfaces(self):
        return []

    def propagate(self, photon):
        photon.transformToLocalCoordinates(self.origin)

        while photon.isAlive and self.contains(photon.r):
            # Pick distance to scattering point
            d = self.material.getScatteringDistance(photon)
            isIntersecting, d = self.intersection(photon.r, photon.ez, d)

            if not isIntersecting:
                # If the scatteringPoint is still inside, we simply move
                # Default is simply photon.moveBy(d) but other things 
                # would be here. Create a new material for other behaviour
                self.material.move(photon, d=d)
 
                # Interact with volume: default is absorption only
                # Default is simply absorb energy. Create a Material
                # for other behaviour
                delta = self.material.interactWith(photon)
                self.scoreInVolume(photon, delta)

                # Scatter within volume
                theta, phi = self.material.getScatteringAngles(photon)
                photon.scatterBy(theta, phi)

                # And go again    
                photon.roulette()
            else:
                # If the scatteringPoint is outside, we move to the surface
                self.material.move(photon, d=d)

                # then we neglect reflections (for now), and score
                # (this can have a significant effect: it will overestimate
                # transmittance)
                self.scoreWhenCrossing(photon)

                # and leave
                break

        self.scoreFinal(photon)
        photon.transformFromLocalCoordinates(self.origin)

    def propagateMany(self, source):
        startTime = time.time()
        N = source.maxCount

        for i, photon in enumerate(source):
            self.propagate(photon)
            self.showProgress(i+1, maxCount=N)

        elapsed = time.time() - startTime
        print('{0:.1f} s for {2} photons, {1:.1f} ms per photon'.format(elapsed, elapsed/N*1000, N))

    def contains(self, position) -> bool:
        """ The base object is infinite. Subclasses override this method
        with their specific geometry. 

        It is important that this function be efficient: it is called
        very frequently. See implementations for Box, Sphere and Layer
        """
        return True

    def intersection(self, position, direction, distance) -> (bool, float): 
        """ This function is a very general function, somewhat efficient
        to find if a photon will leave the object.  `contains` is called
        repeatedly and must be high performance. It is possible to write 
        a specialized version for a subclass, but this version will work
        by default for all objects.
        """

        finalPosition = position + distance*direction
        if self.contains(finalPosition):
            return False, distance

        wasInside = True
        finalPosition = position
        delta = 0.5*distance

        while ( abs(delta) > 0.0001):
            finalPosition += delta * direction
            isInside = self.contains(finalPosition)
            
            if isInside != wasInside:
                delta = -delta / 2.0
            else:
                delta = delta * 1.5

            if delta >= 2*distance:
                return False, distance
            elif delta <= -2*distance:
                return False, distance

            wasInside = isInside

        return True, (finalPosition-position).abs()

    def scoreInVolume(self, photon, delta):
        if self.stats is not None:
            self.stats.scoreInVolume(photon, delta)

    def scoreWhenCrossing(self, photon):
        if self.stats is not None:
            self.stats.scoreWhenCrossing(photon)

    def scoreFinal(self, photon):
        if self.stats is not None:
            self.stats.scoreWhenFinal(photon)

    def showProgress(self, i, maxCount):
        steps = 100
        while steps < i:
            steps *= 10

        if i  % steps == 0:
            print("{2} Photon {0}/{1}".format(i, maxCount, time.ctime()) )
            if self.stats is not None:
                self.stats.showEnergy2D(plane='xz', integratedAlong='y', title="{0} photons".format(i)) 

    def report(self):
        if self.stats is not None:
            self.stats.showEnergy2D(plane='xz', integratedAlong='y', title="Final photons", realtime=False)
            self.stats.showSurfaceIntensities(self.surfaces)

            for i, surface in enumerate(self.surfaces):
                totalWeight = self.stats.totalWeightCrossingPlane(surface)
                print("Transmittance [{0}] : {1:.1f}%".format(surface, 100*totalWeight/self.stats.photonCount))

            print("Absorbance : {0:.1f}%".format(100*self.stats.totalWeightAbsorbed()/self.stats.photonCount))

class Box(Geometry):
    def __init__(self, size, material, stats=None):
        super(Box, self).__init__(material, stats)
        self.size = size

    @property
    def surfaces(self):
        return [ XYPlane(atZ= self.size[2]/2),
                -XYPlane(atZ=-self.size[2]/2),
                 YZPlane(atX= self.size[0]/2),
                -YZPlane(atX=-self.size[0]/2),
                 ZXPlane(atY= self.size[1]/2),
                -ZXPlane(atY=-self.size[1]/2)]

    def contains(self, localPosition) -> bool:
        if abs(localPosition.z) > self.size[2]/2:
            return False
        if abs(localPosition.y) > self.size[1]/2:
            return False
        if abs(localPosition.x) > self.size[0]/2:
            return False

        return True


class Cube(Box):
    def __init__(self, side, material, stats=None):
        super(Cube, self).__init__(material, stats)
        self.size = (side,side,side)

class Layer(Geometry):
    def __init__(self, thickness, material, stats=None):
        super(Layer, self).__init__(material, stats)
        self.thickness = thickness

    @property
    def surfaces(self):
        return [ XYPlane(atZ= self.thickness, description="Back"),
                -XYPlane(atZ= 0, description="Front")]

    def contains(self, localPosition) -> bool:
        if localPosition.z < 0:
            return False
        if localPosition.z > self.thickness:
            return False

        return True

    def intersection(self, position, direction, distance) -> (bool, float): 
        finalPosition = position + distance*direction
        if self.contains(finalPosition):
            return False, distance

        if direction.z > 0:
            d = (self.thickness - position.z)/direction.z
            return True, d
        elif direction.z < 0:
            d = - position.z/direction.z
            return True, d
        return False, distance

class SemiInfiniteLayer(Geometry):
    """ This class is actually a bad idea: the photons don't exit
    on the other side and will just take a long time to propagate.
    It is better to use a finite layer with a thickness a bit larger
    than what you are interested in."""

    def __init__(self, material, stats=None):
        super(SemiInfiniteLayer, self).__init__(material, stats)

    @property
    def surfaces(self):
        return [ -XYPlane(atZ= 0)]

    def contains(self, localPosition) -> bool:
        if localPosition.z < 0:
            return False

        return True

    def intersection(self, position, direction, distance) -> (bool, float): 
        finalPosition = position + distance*direction
        if self.contains(finalPosition):
            return False, distance

        if direction.z < 0:
            d = - position.z/direction.z
            return True, d
        return False, distance

class Sphere(Geometry):
    def __init__(self, radius, material, stats=None):
        super(Sphere, self).__init__(material, stats)
        self.radius = radius

    def contains(self, localPosition) -> bool:
        if localPosition.abs() > self.radius:
            return False

        return True

class KleinBottle(Geometry):
    def __init__(self, material, stats=None):
        super(KleinBottle, self).__init__(material, stats)

    def contains(self, localPosition) -> bool:
        raise NotImplementedError()

class World:
    def __init__(self):
        self.sources = []
        self.geometries = []

    def place(self, geometry, position):
        geometry.origin = position
        self.geometries.append(geometry)
