import warnings
from typing import List, Union, Optional, Tuple
import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def mock_tqdm(iterable, *args, **kwargs):
        warnings.warn("Package 'tqdm' not found. Progress bar will not be shown.")
        return iterable
    tqdm = mock_tqdm

from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.rayscattering.photon import Photon
from pytissueoptics.rayscattering.opencl import CLPhotons, OPENCL_AVAILABLE
from pytissueoptics.scene.solids import Sphere
from pytissueoptics.scene.geometry import Vector, Environment
from pytissueoptics.scene.intersection import FastIntersectionFinder
from pytissueoptics.scene.logger import Logger
from pytissueoptics.scene.viewer import MayaviViewer


class Source:
    def __init__(self, position: Vector, N: int, useHardwareAcceleration: bool = False):
        self._position = position
        self._N = N
        self._photons: Union[List[Photon], CLPhotons] = []
        self._environment = None

        if useHardwareAcceleration and not OPENCL_AVAILABLE:
            warnings.warn("Hardware acceleration not available. Falling back to CPU. Please install pyopencl.")
            useHardwareAcceleration = False
        self._useHardwareAcceleration = useHardwareAcceleration

        self._loadPhotons()

    def propagate(self, scene: RayScatteringScene, logger: Logger = None):
        if self._useHardwareAcceleration:
            self._propagateOpenCL(scene, logger)
        else:
            self._propagateCPU(scene, logger)

    def _propagateCPU(self, scene: RayScatteringScene, logger: Logger = None):
        intersectionFinder = FastIntersectionFinder(scene)
        self._environment = scene.getEnvironmentAt(self._position)
        self._prepareLogger(logger)

        for i in tqdm(range(self._N), desc="Propagating photons"):
            self._photons[i].setContext(self._environment, intersectionFinder=intersectionFinder, logger=logger)
            self._photons[i].propagate()

    def _propagateOpenCL(self, scene: RayScatteringScene, logger: Logger = None):
        self._photons.prepareAndPropagate(scene, logger)

    def getInitialPositionsAndDirections(self) -> Tuple[np.ndarray, np.ndarray]:
        """ To be implemented by subclasses. Needs to return a tuple containing the
        initial positions and normalized directions of the photons as (N, 3) numpy arrays. """
        raise NotImplementedError

    def _loadPhotons(self):
        if self._useHardwareAcceleration:
            self._loadPhotonsOpenCL()
        else:
            self._loadPhotonsCPU()

    def _loadPhotonsCPU(self):
        positions, directions = self.getInitialPositionsAndDirections()
        for i in range(self._N):
            self._photons.append(Photon(Vector(*positions[i]), Vector(*directions[i])))

    def _loadPhotonsOpenCL(self):
        positions, directions = self.getInitialPositionsAndDirections()
        self._photons = CLPhotons(positions, directions, self._N)

    def _prepareLogger(self, logger: Optional[Logger]):
        if logger is None:
            return
        if "photonCount" not in logger.info:
            logger.info["photonCount"] = 0
        logger.info["photonCount"] += self.getPhotonCount()

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
    def __init__(self, position: Vector, direction: Vector, N: int, useHardwareAcceleration: bool = False):
        self._direction = direction
        self._direction.normalize()
        super().__init__(position=position, N=N, useHardwareAcceleration=useHardwareAcceleration)

    def getInitialPositionsAndDirections(self) -> Tuple[np.ndarray, np.ndarray]:
        positions = np.full((self._N, 3), self._position.array)
        directions = np.full((self._N, 3), self._direction.array)
        return positions, directions


class IsotropicPointSource(Source):
    def getInitialPositionsAndDirections(self) -> Tuple[np.ndarray, np.ndarray]:
        positions = np.full((self._N, 3), self._position.array)
        directions = np.random.randn(self._N, 3) * 2 - 1
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return positions, directions
