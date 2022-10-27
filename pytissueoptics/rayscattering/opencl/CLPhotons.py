import os
import time

from pytissueoptics.rayscattering.opencl.CLKeyLog import CLKeyLog
from pytissueoptics.rayscattering.opencl.CLScene import CLScene

try:
    import pyopencl as cl

    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
import numpy as np

from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.CLObjects import PhotonCL, DataPointCL, SeedCL, RandomNumberCL
from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene.logger.logger import Logger
from pytissueoptics.scene.geometry import Environment

PROPAGATION_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'propagation.c')


class CLPhotons:
    def __init__(self, positions: np.ndarray, directions: np.ndarray, N: int, weightThreshold: float = 0.0001):
        self._positions = positions
        self._directions = directions
        self._N = np.uint32(N)
        self._weightThreshold = np.float32(weightThreshold)
        self._initialMaterial = None
        self._initialSolid = None

        self._scene = None
        self._requiredLoggerSize = None
        self._sceneLogger = None

    def setContext(self, scene: RayScatteringScene, environment: Environment, logger: Logger = None):
        self._scene = scene
        self._sceneLogger = logger
        self._initialMaterial = environment.material
        self._initialSolid = environment.solid

        safetyFactor = 1.8
        materials = scene.getMaterials()
        avgAlbedo = np.mean([m.getAlbedo() for m in materials])
        avgInteractions = int(-np.log(self._weightThreshold) / avgAlbedo)
        print(f"Approximate avgInteractions = {avgInteractions}")
        self._requiredLoggerSize = self._N * int(safetyFactor * avgInteractions)

    def propagate(self):
        t0 = time.time()

        scene = CLScene(self._scene, self._N)

        photons = PhotonCL(self._positions, self._directions,
                           materialID=scene.getMaterialID(self._initialMaterial),
                           solidID=scene.getSolidID(self._initialSolid))
        logger = DataPointCL(size=self._requiredLoggerSize)
        randomNumbers = RandomNumberCL(size=self._N)
        seeds = SeedCL(size=self._N)

        program = CLProgram(sourcePath=PROPAGATION_SOURCE_PATH)

        t1 = time.time()
        print(f"OpenCL Propagation Timer: \n ... {t1 - t0:.3f} s. [CLObjects initialization]")
        maxInteractions = np.uint32(self._requiredLoggerSize // self._N)
        program.launchKernel(kernelName='propagate', N=self._N,
                             arguments=[self._N, maxInteractions, self._weightThreshold,
                                        photons, scene.materials, logger, randomNumbers, seeds,
                                        scene.nSolids, scene.solids, scene.surfaces, scene.triangles, scene.vertices,
                                        scene.solidCandidates])
        t2 = time.time()
        log = program.getData(logger)
        t3 = time.time()
        print(f" ... {t3 - t2:.3f} s. [CL logger copy]")

        if not self._sceneLogger:
            return

        keyLog = CLKeyLog(log, sceneCL=scene)
        keyLog.toSceneLogger(self._sceneLogger)

        t4 = time.time()
        print(f" ... {t4 - t3:.3f} s. [Transfer to scene logger]")
        print(f">>> ({t4 - t0:.3f} s.)")

        # pts = self._sceneLogger.getDataPoints()
        # interactionCount = sum([p[0] != 0 for p in pts])
        # print(f"True avgInteractions: {interactionCount / self._N}")
