from .geometry import *
from .stats import *
from .surface import *
from .material import *

class Detector(Geometry):
    def __init__(self, NA, size=(1,1), label="Detector"):
        stats = Stats()
        stats.isMonitoringEnergy = False
        stats.isMonitoringVolume = False
        super(Detector, self).__init__(material=Material(), stats=stats, label=label)
        self.surfaces = [XYRect(origin=Vector(-size[0]/2,-size[1]/2,0),size=size, description="Detector")]                
        self.NA = NA

    def validateGeometrySurfaceNormals(self):
        return

    def contains(self, position):
        return False

    def propagate(self, photon):
        photon.transformToLocalCoordinates(self.origin)
        photon.r.z = 0  # We force it onto the front surface
        self.scoreWhenEntering(photon, self.surfaces[0])
        photon.weight = 0

    def scoreWhenEntering(self, photon, surface):
        if self.stats is not None:
            isContained, u, v = surface.contains(photon.r)
            if isContained:
                intersect = FresnelIntersect(direction=photon.ez, surface=surface, 
                                 distance=0, geometry=self)
                if abs(sin(intersect.thetaIn)) <= self.NA:
                    self.stats.scoreWhenCrossing(photon, surface)

    def scoreInVolume(self, photon, surface):
        return

    def scoreWhenExiting(self, photon, surface):
        return

    def report(self, totalSourcePhotons):
        detectorSurface = self.surfaces[0]
        print("\nXY Detector at z={0:.1f}".format(self.origin.z))
        print("=================")

        if self.stats is not None:
            totalWeight = self.stats.totalWeightCrossingPlane(detectorSurface)
            print("{0:.1f}% intensity".format(100*totalWeight/totalSourcePhotons))
            self.stats.showSurfaceIntensities(self.surfaces, maxPhotons=totalSourcePhotons, bins=21)
            plt.show()