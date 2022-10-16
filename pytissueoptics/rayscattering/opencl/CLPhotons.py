import os
import time

try:
    import pyopencl as cl

    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
import numpy as np

from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.CLObjects import PhotonCL, MaterialCL, DataPointCL, SeedCL, RandomNumberCL, \
    BBoxIntersectionCL
from pytissueoptics.rayscattering.tissues import InfiniteTissue
from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Logger
from pytissueoptics.scene.logger import InteractionKey
from pytissueoptics.scene.geometry import Environment

PROPAGATION_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'propagation.c')


class CLPhotons:
    def __init__(self, positions: np.ndarray, directions: np.ndarray, N: int, weightThreshold: float = 0.0001):
        self._positions = positions
        self._directions = directions
        self._N = np.uint32(N)
        self._weightThreshold = np.float32(weightThreshold)

        self._materials = None
        self._requiredLoggerSize = None
        self._sceneLogger = None

    def setContext(self, scene: RayScatteringScene, environment: Environment, logger: Logger = None):
        worldMaterial = environment.material
        self._materials = [worldMaterial]
        self._sceneLogger = logger

        safetyFactor = 1.8
        avgInteractions = int(-np.log(self._weightThreshold) / worldMaterial.getAlbedo())
        self._requiredLoggerSize = self._N * int(safetyFactor * avgInteractions)

    def propagate(self):
        t0 = time.time()
        photons = PhotonCL(self._positions, self._directions)
        materials = MaterialCL(self._materials)
        logger = DataPointCL(size=self._requiredLoggerSize)
        randomNumbers = RandomNumberCL(size=self._N)
        seeds = SeedCL(size=self._N)
        bboxIntersections = BBoxIntersectionCL(1000, 8)

        program = CLProgram(sourcePath=PROPAGATION_SOURCE_PATH)

        t1 = time.time()
        print(f"OpenCL Propagation Timer: \n ... {t1 - t0:.3f} s. [CLObjects initialization]")
        maxInteractions = np.uint32(self._requiredLoggerSize // self._N)
        program.launchKernel(kernelName='propagate', N=self._N,
                             arguments=[self._N, maxInteractions, self._weightThreshold,
                                        photons, materials, logger, randomNumbers, seeds,
                                        bboxIntersections])
        t2 = time.time()
        log = program.getData(logger)
        t3 = time.time()
        print(f" ... {t3 - t2:.3f} s. [CL logger copy]")
        if self._sceneLogger:
            self._sceneLogger.logDataPointArray(log, InteractionKey("universe", None))
        t4 = time.time()
        print(f" ... {t4 - t3:.3f} s. [Transfer to scene logger]")
        print(f">>> ({t4 - t0:.3f} s.)")
