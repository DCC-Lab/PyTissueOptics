import hashlib
import time
from typing import List, Union, Optional, Tuple
import numpy as np

from pytissueoptics.rayscattering import utils
from pytissueoptics.rayscattering.energyLogging import EnergyLogger
from pytissueoptics.rayscattering.opencl.CLPhotons import CLPhotons
from pytissueoptics.rayscattering.tissues.scatteringScene import ScatteringScene
from pytissueoptics.rayscattering.photon import Photon
from pytissueoptics.rayscattering.opencl import IPPTable, CONFIG, validateOpenCL, warnings
from pytissueoptics.scene.solids import Sphere
from pytissueoptics.scene.geometry import Vector, Environment
from pytissueoptics.scene.intersection import FastIntersectionFinder
from pytissueoptics.scene.logger import Logger
from pytissueoptics.scene.utils import progressBar
from pytissueoptics.scene.viewer import MayaviViewer


class Source:
    def __init__(self, position: Vector, N: int, useHardwareAcceleration: bool = False):
        self._position = position
        self._N = N
        self._photons: Union[List[Photon], CLPhotons] = []
        self._environment = None

        if useHardwareAcceleration:
            useHardwareAcceleration = validateOpenCL()
        self._useHardwareAcceleration = useHardwareAcceleration

        self._loadPhotons()

    def propagate(self, scene: ScatteringScene, logger: Logger = None, showProgress: bool = True):
        self._environment = scene.getEnvironmentAt(self._position)
        self._prepareLogger(logger)

        if self._useHardwareAcceleration:
            IPP = self._getAverageInteractionsPerPhoton(scene)
            self._propagateOpenCL(IPP, scene, logger, showProgress)
            self._updateIPP(scene, logger)
        else:
            self._propagateCPU(scene, logger, showProgress)

    def _propagateCPU(self, scene: ScatteringScene, logger: Logger = None, showProgress: bool = True):
        intersectionFinder = FastIntersectionFinder(scene)

        for i in progressBar(range(self._N), desc="Propagating photons", disable=not showProgress):
            self._photons[i].setContext(self._environment, intersectionFinder=intersectionFinder, logger=logger)
            self._photons[i].propagate()

    def _getAverageInteractionsPerPhoton(self, scene: ScatteringScene) -> float:
        """
        Returns the average number of interactions per photon (IPP) for a given experiment (scene and source
        combination). This is used to optimize the hardware accelerated kernel (OpenCL).

        If the experiment was already seen, the IPP is loaded from the hash table. Otherwise, the IPP is estimated by
        propagating 1000 photons (using a gross estimate of the IPP by assuming an infinite medium of mean scene
        albedo). The measured IPP is stored in the hash table for future use and updated (cumulative average) after
        each propagation.
        """
        experimentHash = self._getExperimentHash(scene)

        if experimentHash not in IPPTable():
            self._measureIPP(scene)

        return IPPTable().getIPP(experimentHash)

    def _getExperimentHash(self, scene: ScatteringScene) -> int:
        return hash((scene, self))

    def _measureIPP(self, scene: ScatteringScene):
        warnings.warn("WARNING: Could not find the average interactions per photon (IPP) for this experiment. \n... "
                      "[Estimating IPP]")

        t0 = time.time()
        tempN = self._N
        self._N = CONFIG.IPP_TEST_N_PHOTONS
        self._loadPhotons()
        tempLogger = Logger()
        estimatedIPP = scene.getEstimatedIPP(CONFIG.WEIGHT_THRESHOLD)
        self._propagateOpenCL(estimatedIPP, scene, tempLogger, showProgress=False)
        self._updateIPP(scene, tempLogger)

        self._N = tempN
        self._loadPhotons()

        warnings.warn(f"... [IPP test took {time.time() - t0:.2f}s]")

    def _updateIPP(self, scene: ScatteringScene, logger: Logger = None):
        if logger is None:
            return
        measuredIPP = logger.nDataPoints / self._N
        table = IPPTable()
        table.updateIPP(self._getExperimentHash(scene), self._N, measuredIPP)

    def _propagateOpenCL(self, IPP: float, scene: ScatteringScene, logger: Logger = None,
                         showProgress: bool = True):
        self._photons.setContext(scene, self._environment, logger=logger)
        self._photons.propagate(IPP=IPP, verbose=showProgress)

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
        if not isinstance(logger, EnergyLogger):
            utils.warn("WARNING: Logging to the base class `Logger` will not allow for energy visualization. "
                       "Please use `EnergyLogger` instead to unlock all features.")

        if "photonCount" not in logger.info:
            logger.info["photonCount"] = 0
        logger.info["photonCount"] += self.getPhotonCount()

        sourceSolid = self.getEnvironment().solid
        logger.info["sourceSolidLabel"] = sourceSolid.getLabel() if sourceSolid else None

        if "sourceHash" not in logger.info:
            logger.info["sourceHash"] = hash(self)
        else:
            if logger.info["sourceHash"] != hash(self):
                utils.warn("WARNING: The logger was previously used with a different source. This may corrupt "
                           "statistics and visualization. Proceed at your own risk.")

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

    @property
    def _nameHash(self) -> int:
        return int(hashlib.sha256(type(self).__name__.encode('utf-8')).hexdigest(), 16)

    @property
    def _hashComponents(self) -> tuple:
        raise NotImplementedError

    def __hash__(self):
        return hash((self._nameHash, *self._hashComponents))


class DirectionalSource(Source):
    def __init__(self, position: Vector, direction: Vector, diameter: float, N: int,
                 useHardwareAcceleration: bool = False):
        self._diameter = diameter
        self._direction = direction
        self._direction.normalize()
        self._xAxis = self._direction.getAnyOrthogonal()
        self._xAxis.normalize()
        self._yAxis = self._direction.cross(self._xAxis)
        self._yAxis.normalize()
        super().__init__(position=position, N=N, useHardwareAcceleration=useHardwareAcceleration)

    def getInitialPositionsAndDirections(self) -> Tuple[np.ndarray, np.ndarray]:
        positions = self._getInitialPositions()
        directions = self._getInitialDirections()
        return positions, directions

    def _getInitialPositions(self):
        return self._getUniformlySampledDisc(self._diameter) + self._position.array

    def _getUniformlySampledDisc(self, diameter) -> np.ndarray:
        # The square root method was used, since the rejection method was slower in numpy because of index lookup.
        # https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
        r = diameter / 2 * np.sqrt(np.random.random((self._N, 1)))
        theta = np.random.random((self._N, 1)) * 2 * np.pi
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        x = np.tile(x, (1, 3))
        y = np.tile(y, (1, 3))
        xAxisArray = np.full((self._N, 3), self._xAxis.array)
        yAxisArray = np.full((self._N, 3), self._yAxis.array)
        xDifference = np.multiply(x, xAxisArray)
        yDifference = np.multiply(y, yAxisArray)

        discPositions = xDifference + yDifference
        return discPositions

    def _getInitialDirections(self):
        return np.full((self._N, 3), self._direction.array)

    @property
    def _hashComponents(self) -> tuple:
        return self._position, self._direction, self._diameter


class PencilPointSource(DirectionalSource):
    def __init__(self, position: Vector, direction: Vector, N: int, useHardwareAcceleration: bool = False):
        super().__init__(position=position, direction=direction, diameter=0, N=N,
                         useHardwareAcceleration=useHardwareAcceleration)


class IsotropicPointSource(Source):
    def getInitialPositionsAndDirections(self) -> Tuple[np.ndarray, np.ndarray]:
        positions = np.full((self._N, 3), self._position.array)
        directions = np.random.randn(self._N, 3)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return positions, directions

    @property
    def _hashComponents(self) -> tuple:
        return self._position,


class DivergentSource(DirectionalSource):
    def __init__(self, position: Vector, direction: Vector, diameter: float, divergence: float, N: int,
                 useHardwareAcceleration: bool = False):
        self._divergence = divergence

        super().__init__(position=position, direction=direction, diameter=diameter, N=N,
                         useHardwareAcceleration=useHardwareAcceleration)

    def _getInitialDirections(self):
        thetaDiameter = np.tan(self._divergence/2) * 2
        directions = self._getUniformlySampledDisc(thetaDiameter)
        directions += self._direction.array
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        return directions

    @property
    def _hashComponents(self) -> tuple:
        return self._position, self._direction, self._diameter, self._divergence
