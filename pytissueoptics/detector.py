from .geometry import *
from .stats import *
from .surface import *
from .material import *

class Detector(Layer):
    def __init__(self, NA, label="Detector"):
        stats = Stats(min = (-2,-2,-0.01), max = (2,2,0), size = (50,50,1))
        super(Detector, self).__init__(thickness=0.01, material=Material(), stats=stats, label=label)

    def propagate(self, photon):
        photon.transformToLocalCoordinates(self.origin)
        self.scoreWhenStarting(photon)
        self.scoreWhenCrossing(photon, self.surfaces[0])
        self.scoreWhenFinal(photon)

    def report(self):
        print("Detector")
        print("=====================")

        if self.stats is not None:
            totalWeightAcrossAllSurfaces = 0
            for i, surface in enumerate(self.surfaces):
                totalWeight = self.stats.totalWeightCrossingPlane(surface)
                print("Transmittance [{0}] : {1:.1f}% ".format(surface, 100*totalWeight/self.stats.inputWeight))
                print("Transmittance [{0}] : {1:.1f}% of total power".format(surface, 100*totalWeight/self.stats.photonCount))
                totalWeightAcrossAllSurfaces += totalWeight

            if len(self.surfaces) != 0:
                self.stats.showSurfaceIntensities(self.surfaces)
