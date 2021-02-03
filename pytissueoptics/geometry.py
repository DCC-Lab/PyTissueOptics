import numpy as np
import matplotlib.pyplot as plt
import signal
import sys
import time
from .surface import *

class Geometry:
    verbose = False

    def __init__(self, material=None, stats=None, label=""):
        self.material = material
        self.origin = Vector(0,0,0)
        self.stats = stats
        self.surfaces = []
        self.label = label

        self.epsilon = 1e-5
        self.startTime = None # We are not calculating anything

    def propagate(self, photon):
        photon.transformToLocalCoordinates(self.origin)

        d = 0
        while photon.isAlive and self.contains(photon.r):
            # Pick distance to scattering point
            if d <= 0:
                d = self.material.getScatteringDistance(photon)
                
            isIntersecting, distToPropagate, surface = self.intersection(photon.r, photon.ez, d)

            if not isIntersecting:
                # If the scattering point is still inside, we simply move
                # Default is simply photon.moveBy(d) but other things 
                # would be here. Create a new material for other behaviour
                self.material.move(photon, d=distToPropagate)

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
                else:
                    # transmit, score, and leave
                    photon.refract(surface)
                    self.scoreWhenCrossing(photon, surface)
                    break

            d -= distToPropagate

            # And go again    
            photon.roulette()

        # Because the code will not typically calculate millions of photons, it is
        # inexpensive to keep all the propagated photons.  This allows users
        # to go through the list after the fact for a calculation of their choice
        self.scoreFinal(photon)
        photon.transformFromLocalCoordinates(self.origin)
    def propagateMany(self, source, graphs=True):
        self.startCalculation()

        N = source.maxCount

        for i, photon in enumerate(source):
            self.propagate(photon)
            self.showProgress(i+1, maxCount=N, graphs=graphs)

        self.completeCalculation()

    def contains(self, position) -> bool:
        """ The base object is infinite. Subclasses override this method
        with their specific geometry. 

        It is important that this function be efficient: it is called
        very frequently. See implementations for Box, Sphere and Layer
        """
        return True

    def intersection(self, position, direction, distance) -> (bool, float, Surface): 
        """ This function is a very general function
        to find if a photon will leave the object.  `contains` is called
        repeatedly, is geometry-specific, and must be high performance. 
        It may be possible to write a specialized version for a subclass,
        but this version will work by default for all objects and is 
        surprisingly efficient.
        """

        finalPosition = position + distance*direction
        if self.contains(finalPosition):
            return False, distance, None

        wasInside = True
        finalPosition = Vector(position) # Copy
        delta = 0.5*distance

        while abs(delta) > 0.00001:
            finalPosition += delta * direction
            isInside = self.contains(finalPosition)
            
            if isInside != wasInside:
                delta = -delta * 0.5

            wasInside = isInside

        surface = None
        for surface in self.surfaces:
            if surface.contains(finalPosition):
                break

        return True, (finalPosition-position).abs(), surface


    def setupSurfaces(self):
        for s in self.surfaces:
            s.indexInside = self.material.index
            s.indexOutside = 1.0 # FIXME

    def isReflected(self, photon, surface) -> bool:
        R = photon.fresnelCoefficient(surface)
        if np.random.random() < R:
            return True
        return False

    def scoreInVolume(self, photon, delta):
        if self.stats is not None:
            self.stats.scoreInVolume(photon, delta)

    def scoreWhenCrossing(self, photon, surface):
        if self.stats is not None:
            self.stats.scoreWhenCrossing(photon, surface)

    def scoreFinal(self, photon):
        if self.stats is not None:
            self.stats.scoreWhenFinal(photon)

    def startCalculation(self):
        self.setupSurfaces()

        if 'SIGUSR1' in dir(signal) and 'SIGUSR2' in dir(signal):
            # Trick to send a signal to code as it is running on Unix and derivatives
            # In the shell, use `kill -USR1 processID` to get more feedback
            # use `kill -USR2 processID` to force a save
            signal.signal(signal.SIGUSR1, self.processSignal)
            signal.signal(signal.SIGUSR2, self.processSignal)

        self.startTime = time.time()

    def completeCalculation(self) -> float:
        if 'SIGUSR1' in dir(signal) and 'SIGUSR2' in dir(signal):
            signal.signal(signal.SIGUSR1, signal.SIG_DFL)
            signal.signal(signal.SIGUSR2, signal.SIG_DFL)

        elapsed = time.time() - self.startTime
        self.startTime = None
        return elapsed

    def processSignal(self, signum, frame):
        if signum == signal.SIGUSR1:
            Geometry.verbose = not Geometry.verbose
            print('Toggling verbose to {0}'.format(Geometry.verbose))
        elif signum == signal.SIGUSR2:
            print("Requesting save (not implemented)")

    def showProgress(self, i, maxCount, graphs=False):
        steps = 100

        if not Geometry.verbose:
            while steps < i:
                steps *= 10

        if i  % steps == 0:
            print("{2} Photon {0}/{1}".format(i, maxCount, time.ctime()) )
            if graphs and self.stats is not None:
                self.stats.showEnergy2D(plane='xz', integratedAlong='y', title="{0} photons".format(i)) 

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
                print("Transmittance [{0}] : {1:.1f}%".format(surface, 100*totalWeight/self.stats.photonCount))
                totalWeightAcrossAllSurfaces += totalWeight

            print("Absorbance : {0:.1f}%".format(100*self.stats.totalWeightAbsorbed()/self.stats.photonCount))

            totalCheck = totalWeightAcrossAllSurfaces + self.stats.totalWeightAbsorbed()
            print("Absorbance + Transmittance = {0:.1f}%".format(100*totalCheck/self.stats.photonCount))

            self.stats.showEnergy2D(plane='xz', integratedAlong='y', title="Final photons", realtime=False)
            self.stats.showSurfaceIntensities(self.surfaces)


    def __repr__(self):
        return "{0}".format(self)

    def __str__(self):
        string = "'{0}' with surfaces {1}\n".format(self.label, self.surfaces)
        string += "{0}".format(self.material)
        return string


class Box(Geometry):
    def __init__(self, size, material, stats=None, label="Box"):
        super(Box, self).__init__(material, stats, label)
        self.size = size
        self.surfaces = [ XYPlane(atZ= self.size[2]/2, description="Back"),
                         -XYPlane(atZ=-self.size[2]/2, description="Front"),
                          YZPlane(atX= self.size[0]/2, description="Right"), 
                         -YZPlane(atX=-self.size[0]/2, description="Left"),
                          ZXPlane(atY= self.size[1]/2, description="Top"),
                         -ZXPlane(atY=-self.size[1]/2, description="Bottom")]

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
        super(Cube, self).__init__(material, stats, label)
        self.size = (side,side,side)

class Layer(Geometry):
    def __init__(self, thickness, material, stats=None, label="Box"):
        super(Layer, self).__init__(material, stats, label)
        self.thickness = thickness
        self.surfaces = [ XYPlane(atZ= self.thickness, description="Back"),
                         -XYPlane(atZ= 0, description="Front")]

    def contains(self, localPosition) -> bool:
        if localPosition.z < -self.epsilon:
            return False
        if localPosition.z > self.thickness + self.epsilon:
            return False

        return True

    def intersection(self, position, direction, distance) -> (bool, float, Surface): 
        finalPosition = position + distance*direction
        if self.contains(finalPosition):
            return False, distance, None

        if direction.z > 0:
            d = (self.thickness - position.z)/direction.z
            return True, d, self.surfaces[0]
        elif direction.z < 0:
            d = - position.z/direction.z
            return True, d, self.surfaces[1]

        return False, distance, None

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

    def intersection(self, position, direction, distance) -> (bool, float, Surface): 
        finalPosition = position + distance*direction
        if self.contains(finalPosition):
            return False, distance, None

        if direction.z < 0:
            d = - position.z/direction.z
            return True, d, self.surfaces[0]

        return False, distance, None

class Sphere(Geometry):
    def __init__(self, radius, material, stats=None, label="Sphere"):
        super(Sphere, self).__init__(material, stats, label)
        self.radius = radius

    def contains(self, localPosition) -> bool:
        if localPosition.abs() > self.radius + self.epsilon:
            return False

        return True

class KleinBottle(Geometry):
    def __init__(self, material, stats=None):
        super(KleinBottle, self).__init__(material, stats)

    def contains(self, localPosition) -> bool:
        raise NotImplementedError()

# class World:
#     def __init__(self):
#         self.sources = []
#         self.geometries = []

#     def place(self, geometry, position):
#         geometry.origin = position
#         self.geometries.append(geometry)

