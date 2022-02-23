from typing import List

from pytissueoptics.rayscattering.tissues import Tissue
from pytissueoptics.scene import Vector, Material
from pytissueoptics.rayscattering.photon import Photon
from pytissueoptics.scene.intersection import SimpleIntersectionFinder
from pytissueoptics.scene.logger import Logger


class Source:
    def __init__(self, position=Vector(0, 0, 0), direction=Vector(0, 0, 1), nPhotons=1):
        self._position = position
        self._direction = direction
        self._nPhotons = nPhotons

        self._photons: List[Photon] = []
        self._createPhotons()

    def _createPhotons(self):
        self._photons = []
        for _ in range(self._nPhotons):
            self._photons.append(Photon(position=self._position.copy(), direction=self._direction.copy()))

    def propagate(self, tissue: Tissue, worldMaterial: Material = Material(), logger: Logger = None):
        intersectionFinder = SimpleIntersectionFinder(tissue.solids)
        for photon in self._photons:
            photon.setContext(worldMaterial=worldMaterial, intersectionFinder=intersectionFinder, logger=logger)
            photon.propagate()
