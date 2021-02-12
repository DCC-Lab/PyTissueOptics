import signal
from .detector import *
import datetime as dt


class World:
    geometries = set()
    sources = set()
    verbose = False
    startTime = None
    stopTime = None

    @classmethod
    def totalSourcePhotons(cls) -> float:
        total = 0
        for source in cls.sources:
            total += source.maxCount
        return total

    @classmethod
    def compute(cls, graphs):
        World.startCalculation()
        World.startTime = dt.datetime.now()
        N = 0
        for source in World.sources:
            N += source.maxCount

            for i, photon in enumerate(source):
                currentGeometry = World.contains(photon.r)
                while photon.isAlive:
                    if currentGeometry is not None:
                        # We are in an object, propagate in it
                        currentGeometry.propagate(photon)
                        # Then check if we are in another adjacent object
                        currentGeometry = World.contains(photon.r)
                    else:
                        # We are in free space (World). Find next object
                        distance, surface, nextGeometry = World.nextObstacle(photon)
                        if surface is not None:
                            # We are hitting something, moving to surface
                            photon.moveBy(distance)
                            # At surface, determine if reflected or not 
                            if nextGeometry.isReflected(photon, surface):
                                # reflect photon and keep propagating
                                photon.reflect(surface)
                                # Move away from surface to avoid getting stuck there
                                photon.moveBy(d=1e-3)
                            else:
                                # transmit, score, and enter (at top of this loop)
                                photon.refract(surface)
                                nextGeometry.scoreWhenEntering(photon, surface)
                                # Move away from surface to avoid getting stuck there
                                photon.moveBy(d=1e-3)
                                currentGeometry = nextGeometry
                        else:
                            photon.weight = 0
                World.showProgress(i + 1, maxCount=source.maxCount, graphs=graphs)

    @classmethod
    def place(cls, anObject, position):
        if isinstance(anObject, Geometry) or isinstance(anObject, Detector):
            anObject.origin = position
            World.geometries.add(anObject)
        elif isinstance(anObject, Source):
            anObject.origin = position
            World.sources.add(anObject)
        elif isinstance(anObject, Detector):
            anObject.origin = position
            World.detector.add(anObject)

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
                surface.indexOutside = 1.0  # Index outside

        if len(World.sources) == 0:
            raise LogicalError("No sources: you must create sources")

        World.startTime = time.time()

    @classmethod
    def completeCalculation(cls):
        if 'SIGUSR1' in dir(signal) and 'SIGUSR2' in dir(signal):
            signal.signal(signal.SIGUSR1, signal.SIG_DFL)
            signal.signal(signal.SIGUSR2, signal.SIG_DFL)

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
                steps *= 5

        if i % steps == 0 or i == maxCount:

            deltaTime = (dt.datetime.now()-World.startTime)
            delta_us = deltaTime / dt.timedelta(microseconds=1)
            avgTimeUsPerPhoton = delta_us/i
            remainingSecs = ((maxCount-i)*avgTimeUsPerPhoton)/10**6
            print(f"{i}/{maxCount} Photons\t::\t{avgTimeUsPerPhoton:9.4f} us/photon\t::\t{remainingSecs:9.4f} seconds remaining.")

            if graphs:
                for geometry in World.geometries:
                    if geometry.stats is not None:
                        geometry.stats.showEnergy2D(plane='xz', integratedAlong='y', title="{0} photons".format(i))



    @classmethod
    def report(cls):
        for geometry in World.geometries:
            geometry.report(totalSourcePhotons=World.totalSourcePhotons())
