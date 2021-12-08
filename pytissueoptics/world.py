import signal
from .detector import *


class World:
    def __init__(self):
        self.geometries = set()
        self.sources = set()
        self.verbose = False
        self.countNotSupposedToBeThere = 0

    def totalSourcePhotons(self) -> float:
        total = 0
        for source in self.sources:
            total += source.maxCount
        return total

    def compute(self, graphs, progress=False):
        self.startCalculation()
        N = 0
        for source in self.sources:
            N += source.maxCount

            for i, photon in enumerate(source):
                currentGeometry = self.contains(photon.globalPosition)
                while photon.isAlive:
                    if currentGeometry is not None:
                        # We are in an object, propagate in it
                        currentGeometry.propagate(photon)
                        # Then check if we are in another adjacent object
                        currentGeometry = self.contains(photon.globalPosition)
                    else:
                        # We are in free space (World). Find next object
                        intersection = self.nextObstacle(photon)
                        if intersection is not None:
                            # We are hitting something, moving to surface
                            photon.moveBy(intersection.distance)
                            # At surface, determine if reflected or not 
                            if intersection.isReflected():
                                # reflect photon and keep propagating
                                photon.reflect(intersection)
                                # Move away from surface to avoid getting stuck there
                                photon.moveBy(d=1e-3)
                            else:
                                # transmit, score, and enter (at top of this loop)
                                photon.refract(intersection)
                                intersection.geometry.scoreWhenEntering(photon, intersection.surface)
                                # Move away from surface to avoid getting stuck there
                                photon.moveBy(d=1e-3)
                                currentGeometry = intersection.geometry
                        else:
                            photon.weight = 0
            if progress:
                self.showProgress(i + 1, maxCount=source.maxCount, graphs=graphs)

        duration = self.completeCalculation()
        if progress:
            print("{0:.1f} ms per photon\n".format(duration * 1000 / N))

    def place(self, anObject, position):
        if isinstance(anObject, Geometry) or isinstance(anObject, Detector):
            anObject.origin = position
            self.geometries.add(anObject)
        elif isinstance(anObject, Source):
            anObject.origin = position
            self.sources.add(anObject)

    def contains(self, worldCoordinates):
        for geometry in self.geometries:
            localCoordinates = worldCoordinates - geometry.origin
            if geometry.contains(localCoordinates):
                return geometry
        return self

    def nextObstacle(self, photon):
        if not photon.isAlive:
            return None
        distance = 1e7
        shortestDistance = distance
        closestIntersect = None
        for geometry in self.geometries:
            photon.transformToLocalCoordinates(geometry.origin)
            someIntersection = geometry.nextEntranceInterface(photon.r, photon.ez, distance=shortestDistance)
            if someIntersection is not None:
                if someIntersection.distance < shortestDistance:
                    shortestDistance = someIntersection.distance
                    closestIntersect = someIntersection

            photon.transformFromLocalCoordinates(geometry.origin)

        return closestIntersect

    def allNextObstacles(self, photons):
        impededPhotons = []
        unimpededPhotons = []
        intersects = []

        for photon in photons:
            intersect = self.nextObstacle(photon)
            if intersect is not None:
                intersects.append(intersect)
                impededPhotons.append(photon)
            else:
                unimpededPhotons.append(photon)

        return Photons(unimpededPhotons), (Photons(impededPhotons), FresnelIntersects(intersects))

    def startCalculation(self):
        if 'SIGUSR1' in dir(signal) and 'SIGUSR2' in dir(signal):
            # Trick to send a signal to code as it is running on Unix and derivatives
            # In the shell, use `kill -USR1 processID` to get more feedback
            # use `kill -USR2 processID` to force a save
            signal.signal(signal.SIGUSR1, self.processSignal)
            signal.signal(signal.SIGUSR2, self.processSignal)

        if len(self.geometries) == 0:
            raise SyntaxError("No geometries: you must create objects")

        for geometry in self.geometries:
            for surface in geometry.surfaces:
                surface.indexInside = geometry.material.index
                surface.indexOutside = 1.0  # Index outside
            try:
                geometry.validateGeometrySurfaceNormals()
            except Exception as err:
                print("The geometry {0} appears invalid. Advancing cautiously.".format(geometry, err))

        if len(self.sources) == 0:
            raise SyntaxError("No sources: you must create sources")

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
            self.verbose = not self.verbose
            print('Toggling verbose to {0}'.format(self.verbose))
        elif signum == signal.SIGUSR2:
            print("Requesting save (not implemented)")

    def showProgress(self, i, maxCount, graphs=False):
        steps = 100

        if not self.verbose:
            while steps < i:
                steps *= 10

        if i % steps == 0:
            print("{2} Photon {0}/{1}".format(i, maxCount, time.ctime()))

            if graphs:
                for geometry in self.geometries:
                    if geometry.stats is not None:
                        geometry.stats.showEnergy2D(plane='xz', integratedAlong='y', title="{0} photons".format(i))

    def report(self, graphs=True):
        for geometry in self.geometries:
            geometry.report(totalSourcePhotons=self.totalSourcePhotons(), graphs=graphs)
