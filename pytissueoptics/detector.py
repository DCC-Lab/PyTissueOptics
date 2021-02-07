from .geometry import *
from .stats import *
from .surface import *
from .material import *

class Detector(Geometry):
    def __init__(self, NA, label="Detector"):
        stats = Stats()
        super(Detector, self).__init__(material=Material(), stats=stats, label=label)
        self.surfaces = [XYPlane(atZ=0,description="Detector")]                

    def contains(self, position):
        return False

    def propagate(self, photon):
        photon.transformToLocalCoordinates(self.origin)
        photon.z = 0 # We force it onto the front surface
        self.scoreWhenStarting(photon)
        self.scoreWhenEntering(photon, self.surfaces[0])
        self.scoreWhenFinal(photon)
        photon.weight = 0

    def scoreWhenEntering(self, photon, surface):
        if self.stats is not None:
            # Do the math for NA If angle too large, reject
            self.stats.scoreWhenCrossing(photon, surface)

    def scoreWhenExiting(self, photon, surface):
        return

    def report(self, totalSourcePhotons):
        print("Detector")
        print("=====================")

        if self.stats is not None:
            for i, surface in enumerate(self.surfaces):
                totalWeight = self.stats.totalWeightCrossingPlane(surface)
                print("Detected [{0}] : {1:.1f}% intensity".format(surface, 100*totalWeight/totalSourcePhotons))
                self.stats.showSurfaceIntensities(self.surfaces, bins=51)
