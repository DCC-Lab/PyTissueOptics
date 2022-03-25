from typing import List

from pytissueoptics.rayscattering.tissues import Tissue
from pytissueoptics.scene import Vector, Material
from pytissueoptics.rayscattering.photon import Photon
from pytissueoptics.scene.intersection import SimpleIntersectionFinder
from pytissueoptics.scene.logger import Logger


class Source:
    def __init__(self, position: Vector, direction: Vector, photons: List[Photon]):
        self._position = position
        self._direction = direction
        self._direction.normalize()

        self._photons = photons

    def propagate(self, tissue: Tissue, worldMaterial: Material = Material(), logger: Logger = None):
        intersectionFinder = SimpleIntersectionFinder(tissue.getSolids())
        tissue.setWorldMaterial(worldMaterial)
        for photon in self._photons:
            photon.setContext(worldMaterial, intersectionFinder=intersectionFinder, logger=logger)
            photon.propagate()

    @property
    def photons(self):
        return self._photons


class PencilSource(Source):
    def __init__(self, position=Vector(0, 0, 0), direction=Vector(0, 0, 1), N=100):
        photons = []
        for _ in range(N):
            photons.append(Photon(position=position.copy(), direction=direction.copy()))

        super().__init__(position, direction, photons)
