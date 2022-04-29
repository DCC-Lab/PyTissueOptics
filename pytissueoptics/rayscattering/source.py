from typing import List, Union
import numpy as np

from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.rayscattering.photon import Photon
from pytissueoptics.rayscattering.opencl import CLPhotons
from pytissueoptics.scene.solids import Sphere
from pytissueoptics.scene.geometry import Vector, Environment
from pytissueoptics.scene.intersection import FastIntersectionFinder
from pytissueoptics.scene.logger import Logger
from pytissueoptics.scene.viewer import MayaviViewer


class Source:
    def __init__(self, position: Vector, N: int, use_opencl: bool = False):
        self._position = position
        self._N = N
        self._photons: Union[List[Photon], CLPhotons] = []
        self._environment = None
        if use_opencl:
            self._makePhotonsOpenCL()
            self.propagate = self._propagateOpenCL
        else:
            self._makePhotonsCPU()
            self.propagate = self._propagateCPU

    def propagate(self, scene: RayScatteringScene, logger: Logger = None):
        raise NotImplementedError

    def _makePhotonsCPU(self):
        raise NotImplementedError

    def _propagateCPU(self, scene: RayScatteringScene, logger: Logger = None):
        intersectionFinder = FastIntersectionFinder(scene)
        self._environment = scene.getEnvironmentAt(self._position)

        for photon in self._photons:
            photon.setContext(self._environment, intersectionFinder=intersectionFinder, logger=logger)
            photon.propagate()

    def _makePhotonsOpenCL(self):
        self._photons = CLPhotons(self)

    def _propagateOpenCL(self, scene: RayScatteringScene, logger: Logger = None):
        self._photons.prepareAndPropagate(scene, logger)

    @property
    def photons(self):
        return self._photons

    def getPhotonCount(self) -> int:
        return self._N

    def getPosition(self) -> Vector:
        return self._position

    def getEnvironment(self) -> Environment:
        if self._environment is None:
            return Environment(None)
        return self._environment

    def addToViewer(self, viewer: MayaviViewer, size: float = 0.1):
        sphere = Sphere(radius=size/2, position=self._position)
        viewer.add(sphere, representation="surface", colormap="Wistia", opacity=0.8)


class PencilSource(Source):
    def __init__(self, position: Vector, direction: Vector, N: int, use_opencl: bool = False):
        self._direction = direction
        self._direction.normalize()
        super().__init__(position, N, use_opencl)

    def getDirection(self) -> Vector:
        return self._direction

    def _makePhotonsCPU(self):
        for _ in range(self._N):
            self._photons.append(Photon(position=self._position.copy(), direction=self._direction.copy()))


class IsotropicPointSource(Source):
    @staticmethod
    def _getRandomDirection():
        phi = np.random.random() * 2 * np.pi
        theta = np.arccos(2 * np.random.random() - 1)
        return theta, phi

    def _makePhotonsCPU(self):
        for _ in range(self._N):
            p = Photon(position=self._position.copy(), direction=Vector(0, 0, 1))
            theta, phi = self._getRandomDirection()
            p.scatterBy(theta, phi)
            self._photons.append(p)
