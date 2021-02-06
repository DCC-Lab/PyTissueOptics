from .geometry import *
from .stats import *
from .surface import *
from .material import *

class Detector(Layer):
    def __init__(self, NA, label="Detector"):
        super(Detector, self).__init__(thickness=0.01, material=Material(), stats=Stats(), label=label)

    def propagate(self, photon):
        print("Entering detector")
        photon.transformToLocalCoordinates(self.origin)
        self.scoreWhenStarting(photon)
        self.scoreWhenCrossing(photon, self.surfaces[0])
        self.scoreWhenFinal(photon)
