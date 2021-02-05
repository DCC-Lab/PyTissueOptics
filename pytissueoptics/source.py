import numpy as np
import time
import warnings
from .vector import *
from .photon import *

class Source:
    allSources = []

    def __init__(self, position, maxCount):
        self.origin = position
        self.maxCount = maxCount
        self.iteration = 0
        self._photons = []
        Source.allSources.append(self)

    def __iter__(self):
        self.iteration = 0
        return self

    def __len__(self) -> int:
        return self.maxCount

    def __getitem__(self, item):
        if item < 0:
            # Convert negative index to positive (i.e. -1 == len - 1)
            item += self.maxCount

        if item < 0 or item >= self.maxCount:
            raise IndexError(f"Index {item} out of bound, min = 0, max {self.maxCount}.")

        start = time.monotonic()
        while len(self._photons) <= item:
            self._photons.append(self.newPhoton())
            if time.monotonic() - start > 2:
                warnings.warn(f"Generating missing photon. This can take a few seconds.", UserWarning)

        return self._photons[item]

    def __next__(self) -> Photon:
        if self.iteration >= self.maxCount:
            raise StopIteration
        # This should be able to know if enough photon. If not enough, generate them
        photon = self[self.iteration]
        self.iteration += 1
        return photon

    def newPhoton(self) -> Photon:
        raise NotImplementedError()

class IsotropicSource(Source):
    def __init__(self, position, maxCount):
        super(IsotropicSource, self).__init__(position, maxCount)

    def newPhoton(self) -> Photon:
        p = Photon()
        p.r = self.origin

        phi = np.random.random()*2*np.pi
        cost = 2*np.random.random()-1 

        p.scatterBy(np.arccos(cost), phi)
        return p

class PencilSource(Source):
    def __init__(self, position, direction, maxCount):
        super(PencilSource, self).__init__(position, maxCount)
        self.direction = direction

    def newPhoton(self) -> Photon:
        return Photon( Vector(self.origin), Vector(self.direction))
