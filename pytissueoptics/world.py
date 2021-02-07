import numpy as np
import matplotlib.pyplot as plt
import signal
import sys
import time
from .surface import *
from .photon import *
from .source import *
from .geometry import *
from .detector import *

class World:
    geometries = set()
    sources = set()
    verbose = False

    @classmethod
    def totalSourcePhotons(cls) -> float:
        total = 0
        for source in cls.sources:
            total += source.maxCount
        return total

    @classmethod
    def compute(self, graphs):
        World.startCalculation()
        N = 0
        for source in World.sources:
            N += source.maxCount

            for i, photon in enumerate(source):
                while photon.isAlive:
                    currentGeometry = World.contains(photon.r)
                    if currentGeometry is not None:
                        currentGeometry.propagate(photon)
                    else:
                        distance, surface, nextGeometry = World.nextObstacle(photon)
                        if surface is not None:
                            # Moving to next object in air
                            photon.moveBy(distance)
                            R = photon.fresnelCoefficient(surface)
                            photon.refract(surface)
                            photon.decreaseWeightBy(R*photon.weight)
                            photon.moveBy(1e-4)
                        else:
                            photon.weight = 0

                World.showProgress(i+1, maxCount=source.maxCount, graphs=graphs)

        duration = World.completeCalculation()
        print("{0:.1f} ms per photon\n".format(duration*1000/N))

    @classmethod
    def place(cls, anObject, position):
        if isinstance(anObject, Geometry) or isinstance(anObject, Detector):
            anObject.origin = position
            World.geometries.add(anObject)
        elif isinstance(anObject, Source):
            anObject.origin = position
            World.sources.add(anObject)

    @classmethod
    def contains(cls, worldCoordinates):
        for geometry in World.geometries:
            localCoordinates = worldCoordinates - geometry.origin
            if geometry.contains(localCoordinates):
                return geometry
        return None

    @classmethod
    def nextObstacle(cls, photon):
        distance = 1e7
        intersect = None
        sGeometry = None
        for geometry in World.geometries:
            photon.transformToLocalCoordinates(geometry.origin)
            distanceToSurface, surface = geometry.nextEntranceInterface(photon.r, photon.ez, distance=distance)
            if distanceToSurface < distance:
                distance = distanceToSurface
                intersect = surface
                sGeometry = geometry
            photon.transformFromLocalCoordinates(geometry.origin)

        return distance, intersect, sGeometry

    @classmethod
    def startCalculation(self):
        if 'SIGUSR1' in dir(signal) and 'SIGUSR2' in dir(signal):
            # Trick to send a signal to code as it is running on Unix and derivatives
            # In the shell, use `kill -USR1 processID` to get more feedback
            # use `kill -USR2 processID` to force a save
            signal.signal(signal.SIGUSR1, self.processSignal)
            signal.signal(signal.SIGUSR2, self.processSignal)

        if len(World.geometries) == 0:
            raise LogicalError("No geometries: you must create objects")

        for geometry in World.geometries:
            for surface in geometry.surfaces:
                surface.indexInside = geometry.material.index
                surface.indexOutside = 1.0 # Index outside

        if len(World.sources) == 0:
            raise LogicalError("No sources: you must create sources")

        World.startTime = time.time()

    @classmethod
    def completeCalculation(cls) -> float:
        if 'SIGUSR1' in dir(signal) and 'SIGUSR2' in dir(signal):
            signal.signal(signal.SIGUSR1, signal.SIG_DFL)
            signal.signal(signal.SIGUSR2, signal.SIG_DFL)

        elapsed = time.time() - World.startTime
        World.startTime = None
        return elapsed

    @classmethod
    def processSignal(cls, signum, frame):
        if signum == signal.SIGUSR1:
            World.verbose = not World.verbose
            print('Toggling verbose to {0}'.format(World.verbose))
        elif signum == signal.SIGUSR2:
            print("Requesting save (not implemented)")

    @classmethod
    def showProgress(cls, i, maxCount, graphs=False):
        steps = 100

        if not World.verbose:
            while steps < i:
                steps *= 10

        if i  % steps == 0:
            print("{2} Photon {0}/{1}".format(i, maxCount, time.ctime()) )

            if graphs:
                for geometry in World.geometries:
                    if geometry.stats is not None:
                        geometry.stats.showEnergy2D(plane='xz', integratedAlong='y', title="{0} photons".format(i)) 

    @classmethod
    def report(cls):
        for geometry in World.geometries:
            geometry.report(totalSourcePhotons=World.totalSourcePhotons())
