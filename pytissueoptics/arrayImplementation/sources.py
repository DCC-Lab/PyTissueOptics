from abc import ABC
from vectors import *
from arrayImplementation.photons import Photons


class Source(ABC):
    def __init__(self, maxCount):
        self.origin = [0, 0, 0]
        self.maxCount = maxCount
        self._photons = []

    def __len__(self) -> int:
        return self.maxCount

    def initializePhotons(self):
        raise NotImplementedError


class IsotropicSource(Source):
    def __init__(self, maxCount):
        super(IsotropicSource, self).__init__(maxCount)

    def initializePhotons(self):
        theta = np.random.rand(self.maxCount) * 2 * np.pi
        phi = np.random.rand(self.maxCount) * np.pi
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        direction = Vectors(np.stack((x, y, z), axis=-1))
        position = Vectors([self.origin]*self.maxCount)

        self._photons = Photons(position, direction)
        return self._photons
