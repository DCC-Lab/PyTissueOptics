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

PROPAGATION_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')


class CLProgram:
    def __init__(self, sourceDir: str):
        self._sourceDir = sourceDir
        self._context = cl.create_some_context()
        self._mainQueue = cl.CommandQueue(self._context)
        self._device = self._context.devices[0]
        self._program = None

    def build(self, CLObjects: List[CLType]):
        for CLObject in CLObjects:
            CLObject.build(self._device, self._context)

        self._program = cl.Program(self._context, self._getSourceCode(CLObjects)).build()

    def _getSourceCode(self, CLObjects: List[CLType]) -> str:
        sourceFiles = ['random.c', 'vectorOperators.c', 'propagation.c']
        # fixme: source file ordering matters. Maybe set propagation.c as single sourceFile with #includes at start...
        # sourceFiles = [f for f in list(os.walk(self._sourceDir))[0][2] if f.endswith('.c')]
        sourceCode = ''
        for sourceFile in sourceFiles:
            sourceCode += open(os.path.join(self._sourceDir, sourceFile)).read()

        typeDeclarations = ''.join([CLObject.declaration for CLObject in CLObjects])
        return typeDeclarations + sourceCode


class CLPhotons:
    def __init__(self, positions: np.ndarray, directions: np.ndarray, N: int, weightThreshold: float = 0.0001):
        self._positions = positions
        self._directions = directions
        self._N = np.uint32(N)
        self._weightThreshold = np.float32(weightThreshold)

        self._program = CLProgram(sourceDir=PROPAGATION_SOURCE_DIR)

    def prepareAndPropagate(self, scene: RayScatteringScene, logger: Logger):
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

    def _propagate(self, sceneLogger: Logger):
        t0 = time.time_ns()

        # todo: whats up with this signature ? first 3 arguments not in C decl
        self._program._program.propagate(self._program._mainQueue, self._photons._HOST_buffer.shape, None, self._N, self._weightThreshold,
                                self._photons._DEVICE_buffer,
                                self._material._DEVICE_buffer, self._logger._DEVICE_buffer, self._randomFloat._DEVICE_buffer,
                                self._randomSeed._DEVICE_buffer)
        self._program._mainQueue.finish()
        cl.enqueue_copy(self._program._mainQueue, dest=self._logger._HOST_buffer, src=self._logger._DEVICE_buffer)
        t1 = time.time_ns()
        print("CLPhotons.propagate: {} s".format((t1 - t0) / 1e9))

        log = rfn.structured_to_unstructured(self._logger._HOST_buffer)
        sceneLogger.logDataPointArray(log, InteractionKey("universe", None))
