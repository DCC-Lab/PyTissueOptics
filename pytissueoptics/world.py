import numpy as np
import matplotlib.pyplot as plt
import signal
import sys
import time
from .surface import *
from .photon import *
from .source import *

class World:
    geometries = []
    sources = []
    verbose = False

    @classmethod
    def propagateAll(self, graphs):
        World.startCalculation()

        for source in World.sources:
            for i, photon in enumerate(source):
                while photon.isAlive:
                    currentGeometry = World.contains(photon.r)
                    if currentGeometry is not None:
                        currentGeometry.propagate(photon)
                    else:
                        distanceToSurface, surface, nextGeometry = World.willEnterThroughInterface(photon)
                        if surface is not None:
                            # Moving to next object in air
                            photon.moveBy(distanceToSurface)
                            #photon.refract(surface) #screw reflections!
                            photon.moveBy(1e-4)
                        else:
                            photon.weight = 0

                World.showProgress(i+1, maxCount=source.maxCount, graphs=graphs)

        World.completeCalculation()

    @classmethod
    def contains(cls, worldCoordinates):
        for geometry in World.geometries:
            localCoordinates = worldCoordinates - geometry.origin
            if geometry.contains(localCoordinates):
                return geometry
        return None

    @classmethod
    def willEnterThroughInterface(cls, photon):
        distance = 1e7
        intersect = None
        sGeometry = None
        for geometry in World.geometries:
            photon.transformToLocalCoordinates(geometry.origin)
            distanceToSurface, surface = geometry.mayEnterThroughInterface(photon.r, photon.ez, distance=1e4)
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
            print('Toggling verbose to {0}'.format(Geometry.verbose))
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
            # if graphs and self.stats is not None:
            #     self.stats.showEnergy2D(plane='xz', integratedAlong='y', title="{0} photons".format(i)) 

    @classmethod
    def report(cls):
        for geometry in World.geometries:
            geometry.report()
            