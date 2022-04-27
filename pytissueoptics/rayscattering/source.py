from typing import List

from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.rayscattering.photon import Photon
from pytissueoptics.scene.geometry import Vector, Environment
from pytissueoptics.scene.intersection import SimpleIntersectionFinder
from pytissueoptics.scene.logger import Logger


class Source:
    def __init__(self, position: Vector, direction: Vector, photons: List[Photon]):
        self._position = position
        self._direction = direction
        self._direction.normalize()

        self._photons = photons
        self._environment = None

    def propagate(self, scene: RayScatteringScene, logger: Logger = None):
        intersectionFinder = SimpleIntersectionFinder(scene)
        self._environment = scene.getEnvironmentAt(self._position)

        for photon in self._photons:
            photon.setContext(self._environment, intersectionFinder=intersectionFinder, logger=logger)
            photon.propagate()

    @property
    def photons(self):
        return self._photons

    def getPhotonCount(self) -> int:
        return len(self._photons)

    def getPosition(self) -> Vector:
        return self._position

    def getEnvironment(self) -> Environment:
        if self._environment is None:
            return Environment(None)
        return self._environment


class PencilSource(Source):
    def __init__(self, position=Vector(0, 0, 0), direction=Vector(0, 0, 1), N=100):
        photons = []
        for _ in range(N):
            photons.append(Photon(position=position.copy(), direction=direction.copy()))

        super().__init__(position, direction, photons)
