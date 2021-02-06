from .geometry import *
from .stats import *
from .surface import *
from .material import *

class Detector(Layer):
    def __init__(self, NA, label="Detector"):
        stats = Stats(min = (-2,-2,-0.01), max = (2,2,0), size = (50,50,1))
        super(Detector, self).__init__(thickness=0.01, material=Material(), stats=stats, label=label)
        for surface in self.surfaces:
            if surface.normal.dot(zHat) > 0:
                surface.description = "Detector"
                self.surfaces = [surface]

    def propagate(self, photon):
        photon.transformToLocalCoordinates(self.origin)
        photon.z = 0 # We force it onto the front surface
        self.scoreWhenStarting(photon)
        self.scoreWhenCrossing(photon, self.surfaces[0])
        self.scoreWhenFinal(photon)

    def scoreWhenCrossing(self, photon, surface):
        if self.stats is not None:
            # Do the math for NA If angle too large, reject
            self.stats.scoreWhenCrossing(photon, surface)

    def report(self):
        print("Detector")
        print("=====================")

        if self.stats is not None:
            for i, surface in enumerate(self.surfaces):
                totalWeight = self.stats.totalWeightCrossingPlane(surface)
                print("Detected [{0}] : {1:.1f}% photon weight".format(surface, totalWeight))
                self.stats.showSurfaceIntensities(self.surfaces, bins=51)
