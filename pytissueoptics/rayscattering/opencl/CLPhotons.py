import os
import time
from typing import List

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
import numpy as np
from numpy.lib import recfunctions as rfn

from pytissueoptics.rayscattering.opencl.types import CLType, PhotonCLType, MaterialCLType, LoggerCLType, \
    RandomSeedCLType, RandomFloatCLType
from pytissueoptics.rayscattering.tissues import InfiniteTissue
from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Logger
from pytissueoptics.scene.logger import InteractionKey

PROPAGATION_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'propagation.c')


class CLProgram:
    def __init__(self, sourcePath: str):
        self._sourcePath = sourcePath
        self._context = cl.create_some_context()
        self._mainQueue = cl.CommandQueue(self._context)
        self._device = self._context.devices[0]
        self._program = None

    def build(self, CLObjects: List[CLType]):
        for CLObject in CLObjects:
            CLObject.build(self._device, self._context)

        typeDeclarations = ''.join([CLObject.declaration for CLObject in CLObjects])
        sourceCode = typeDeclarations + open(self._sourcePath).read()

        includeDir = os.path.dirname(self._sourcePath).replace('\\', '/')
        self._program = cl.Program(self._context, sourceCode).build(options=f"-I {includeDir}")


class CLPhotons:
    def __init__(self, positions: np.ndarray, directions: np.ndarray, N: int, weightThreshold: float = 0.0001):
        self._positions = positions
        self._directions = directions
        self._N = np.uint32(N)
        self._weightThreshold = np.float32(weightThreshold)

        self._program = CLProgram(sourcePath=PROPAGATION_SOURCE_PATH)

    def prepareAndPropagate(self, scene: RayScatteringScene, logger: Logger = None):
        self._extractFromScene(scene)
        self._createCLObjects()

        self._program.build(CLObjects=[self._photons, self._material, self._logger,
                                       self._randomSeed, self._randomFloat])

        self._propagate(sceneLogger=logger)

    def _extractFromScene(self, scene: RayScatteringScene):
        if type(scene) is not InfiniteTissue:
            raise TypeError("OpenCL propagation is only supported for InfiniteTissue for the moment.")
        self._worldMaterial = scene.getWorldEnvironment().material

    def _createCLObjects(self):
        self._photons = PhotonCLType(self._positions, self._directions)
        self._material = MaterialCLType(self._worldMaterial)
        self._logger = LoggerCLType(self._requiredLoggerSize())
        self._randomSeed = RandomSeedCLType(size=self._N)
        self._randomFloat = RandomFloatCLType(size=self._N)

    def _requiredLoggerSize(self) -> int:
        return int(-np.log(self._weightThreshold) / self._worldMaterial.getAlbedo()) * self._N

    def _propagate(self, sceneLogger: Logger = None):
        t0 = time.time_ns()

        # todo: whats up with this signature ? first 3 arguments not in C decl
        self._program._program.propagate(self._program._mainQueue, self._photons._HOST_buffer.shape, None, self._N,
                                         self._weightThreshold,
                                         self._photons._DEVICE_buffer,
                                         self._material._DEVICE_buffer, self._logger._DEVICE_buffer,
                                         self._randomFloat._DEVICE_buffer,
                                         self._randomSeed._DEVICE_buffer)
        self._program._mainQueue.finish()
        cl.enqueue_copy(self._program._mainQueue, dest=self._logger._HOST_buffer, src=self._logger._DEVICE_buffer)

        t1 = time.time_ns()
        print("CLPhotons.propagate: {} s".format((t1 - t0) / 1e9))

        log = rfn.structured_to_unstructured(self._logger._HOST_buffer)
        if sceneLogger:
            sceneLogger.logDataPointArray(log, InteractionKey("universe", None))
