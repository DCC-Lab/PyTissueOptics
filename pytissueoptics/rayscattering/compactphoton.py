import math
import random
from typing import Optional
import numpy as np
from pytissueoptics.rayscattering.photon import Photon, Vector

WORLD_LABEL = "world"
WEIGHT_THRESHOLD = 1e-4

class CompactVector(Vector):
    npVector = np.dtype([("x", np.float32),("y", np.float32),("z", np.float32)])
    def __init__(self, rawBuffer, index=0, offset=0, stride=0):
        super().__init__()
        self._data = np.frombuffer(rawBuffer, dtype=np.float32, count=3, offset=offset+index*stride)

class CompactPhoton(Photon):
    Struct = np.dtype([("position", CompactVector.npVector),
                      ("direction", CompactVector.npVector),
                      ("er", CompactVector.npVector),
                      ("weight", np.float32)                      
                      ])

    def __init__(self, compactPhotons, index):
        # HACK: I would prefer None
        super().__init__(position=Vector(0,0,0), direction=Vector(0,0,1))
        self.compactPhotons = compactPhotons
        self.index = index
        self.array = None

        self._position = CompactVector(self.compactPhotons.rawbuffer, index=self.index, offset=0, stride=CompactPhoton.Struct.itemsize)
        self._direction = CompactVector(self.compactPhotons.rawbuffer, index=self.index, offset=3*4, stride=CompactPhoton.Struct.itemsize)
        self._er = CompactVector(self.compactPhotons.rawbuffer, index=self.index, offset=6*4, stride=CompactPhoton.Struct.itemsize)

    @property
    def elementAsStruct(self):
        if self.array is None:
            self.array = np.frombuffer(self.compactPhotons.rawbuffer, dtype=CompactPhoton.Struct, count=1, offset=CompactPhoton.Struct.itemsize * self.index)
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
            self._photons = np.zeros((maxCount,), dtype=CompactPhoton.Struct)
            self.rawbuffer = self._photons.data
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

