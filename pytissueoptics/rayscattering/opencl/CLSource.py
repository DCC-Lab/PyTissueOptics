from typing import List

from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.scene import Vector
from pytissueoptics.rayscattering.photon import Photon


class CLSource:
    def __init__(self, position: Vector, direction: Vector, photons: List[Photon], worldMaterial: ScatteringMaterial):
        self._position = position
        self._direction = direction
        self._direction.normalize()
        self._photons = photons
        self._worldMaterial = worldMaterial
        for photon in self._photons:
            photon.setContext(self._worldMaterial, intersectionFinder=None, logger=None)

    @property
    def photons(self):
        return self._photons


class CLPencilSource(CLSource):
    def __init__(self, position=Vector(0, 0, 0), direction=Vector(0, 0, 1), N=100, worldMaterial=ScatteringMaterial()):
        photons = []
        for _ in range(N):
            photons.append(Photon(position=position.copy(), direction=direction.copy()))

        super().__init__(position, direction, photons, worldMaterial)
