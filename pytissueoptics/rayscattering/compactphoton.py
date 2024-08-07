import math
import random
from typing import Optional
import numpy as np
from pytissueoptics.rayscattering.photon import Photon
from pytissueoptics.scene.geometry import Vector, CompactVector
from pytissueoptics.rayscattering import Source

WORLD_LABEL = "world"
WEIGHT_THRESHOLD = 1e-4

class CompactPhoton(Photon):
    npStruct = np.dtype([("position", CompactVector.npStruct),
                      ("direction", CompactVector.npStruct),
                      ("er", CompactVector.npStruct),
                      ("weight", np.float32)                      
                      ])

    def __init__(self, compactPhotons, index):
        # HACK: I would prefer None
        super().__init__(position=Vector(0,0,0), direction=Vector(0,0,1))
        self.compactPhotons = compactPhotons
        self.index = index
        self.array = None

        self._position = CompactVector(self.compactPhotons.rawbuffer, index=self.index, offset=0, stride=CompactPhoton.npStruct.itemsize)
        self._direction = CompactVector(self.compactPhotons.rawbuffer, index=self.index, offset=3*4, stride=CompactPhoton.npStruct.itemsize)
        self._er = CompactVector(self.compactPhotons.rawbuffer, index=self.index, offset=6*4, stride=CompactPhoton.npStruct.itemsize)

    @property
    def elementAsStruct(self):
        if self.array is None:
            self.array = np.frombuffer(self.compactPhotons.rawbuffer, dtype=CompactPhoton.npStruct, count=1, offset=CompactPhoton.npStruct.itemsize * self.index)
        return self.array[0]

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, vector):
        self._position.x = vector.x
        self._position.y = vector.y
        self._position.z = vector.z

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, vector):
        self._direction.x = vector.x
        self._direction.y = vector.y
        self._direction.z = vector.z

    @property
    def er(self):
        return self._er

    @er.setter
    def er(self, vector):
        self._er.x = vector.x
        self._er.y = vector.y
        self._er.z = vector.z

    @property
    def weight(self):
        return self.elementAsStruct[3]

    @weight.setter
    def weight(self, value):
        self.elementAsStruct[3] = value



class CompactPhotons:
    def __init__(self, compactPhotonsStructuredBuffer=None, maxCount=None):
        self.maxCount = maxCount
        if maxCount is not None:
            self.photons = np.zeros((maxCount,), dtype=CompactPhoton.npStruct)
            self.rawbuffer = self.photons.data
        else:
            raise ValueError('You must provide a compactPhotonsStructuredBuffer or a maxCount')

        self.iteration = 0

    def __getitem__(self, index):
        return CompactPhoton(self, index)

    def __setitem__(self, index, value):
        CompactPhoton(self, index).assign(value)

    def __iter__(self):
        self.iteration = 0
        return self

    def __len__(self):
        return self.maxCount

    def __next__(self) -> CompactPhoton:

        if self.iteration < len(self):
            photon = self[self.iteration] # Again we want to use __getitem__ for self for CompactPhotons
            self.iteration += 1
            return photon

        raise StopIteration

    def append(self, tuple):
        raise RuntimeError('You can only replace elements from a pre-allocated CompactPhotons')

    def propagateAll(self, environment, ):
        for i, photon in self.photons:
            photon.setContext(self.environment, intersectionFinder=intersectionFinder, logger=logger)
            photon.propagate()

class CompactSource(Source):
    def __init__(self, position: Vector, displaySize: float = 0.1):
        super().__init__(position=position, N=N, useHardwareAcceleration = True, displaySize=displaySize)

