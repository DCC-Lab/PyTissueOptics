from .geometry import *
from .stats import *
from .surface import *
from .material import *

class Detector(Geometry):
    def __init__(self, NA, label="Detector"):
        stats = Stats()
        super(Detector, self).__init__(material=Material(), stats=stats, label=label)
        self.surfaces = [XYPlane(atZ=0,description="Detector")]                
        self.NA = NA

    def contains(self, position):
        return False

    def propagate(self, photon):
        photon.transformToLocalCoordinates(self.origin)
        photon.z = 0  # We force it onto the front surface
        self.scoreWhenEntering(photon, self.surfaces[0])
        photon.weight = 0

    def scoreWhenEntering(self, photon, surface):
        if self.stats is not None:
            cosPhi = photon.ez.normalizedDotProduct(-self.surfaces[0].normal)
            sinPhi = math.sqrt(1-cosPhi*cosPhi)
            
            if sinPhi <= self.NA:
                self.stats.scoreWhenCrossing(photon, surface)
            else:
                if photon.r.abs() < 0.3:
                    print(photon.r, cosPhi, sinPhi)
    def scoreInVolume(self, photon, surface):
        return

    def scoreWhenExiting(self, photon, surface):
        return

    def report(self, totalSourcePhotons):
        detectorSurface = self.surfaces[0]
        print("\nDetector at z={0:.1f}".format(detectorSurface.origin.z))
        print("=================")

        if self.stats is not None:
            totalWeight = self.stats.totalWeightCrossingPlane(detectorSurface)
            print("{0:.1f}% intensity".format(100*totalWeight/totalSourcePhotons))
            self.stats.showSurfaceIntensities(self.surfaces, maxPhotons=totalSourcePhotons, bins=11)
