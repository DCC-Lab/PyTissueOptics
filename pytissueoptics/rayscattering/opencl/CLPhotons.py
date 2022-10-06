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

from pytissueoptics.rayscattering.opencl.types import CLObject, PhotonCL, MaterialCL, LoggerCL, \
    RandomSeedCL, RandomFloatCL
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

    def launchKernel(self, kernelName: str, N: int, arguments: list):
        self._build(objects=[x for x in arguments if isinstance(x, CLObject)])
        arguments = [x.deviceBuffer if isinstance(x, CLObject) else x for x in arguments]

        kernel = getattr(self._program, kernelName)
        kernel(self._mainQueue, (N,), None, *arguments)
        self._mainQueue.finish()

    def _build(self, objects: List[CLObject]):
        for _object in objects:
            _object.build(self._device, self._context)

        typeDeclarations = ''.join([_object.declaration for _object in objects])
        sourceCode = typeDeclarations + open(self._sourcePath).read()

        includeDir = os.path.dirname(self._sourcePath).replace('\\', '/')
        self._program = cl.Program(self._context, sourceCode).build(options=f"-I {includeDir}")

    def getData(self, _object: CLObject):
        cl.enqueue_copy(self._mainQueue, dest=_object.hostBuffer, src=_object.deviceBuffer)
        return rfn.structured_to_unstructured(_object.hostBuffer)


class CLPhotons:
    # todo: might have to rename this class at some point (conflicts with PhotonCL)
    def __init__(self, positions: np.ndarray, directions: np.ndarray, N: int, weightThreshold: float = 0.0001):
        self._positions = positions
        self._directions = directions
        self._N = np.uint32(N)
        self._weightThreshold = np.float32(weightThreshold)

        self._program = CLProgram(sourcePath=PROPAGATION_SOURCE_PATH)

    def prepareAndPropagate(self, scene: RayScatteringScene, logger: Logger = None):
        self._extractFromScene(scene)
        self._propagate(sceneLogger=logger)

    def _extractFromScene(self, scene: RayScatteringScene):
        if type(scene) is not InfiniteTissue:
            raise TypeError("OpenCL propagation is only supported for InfiniteTissue for the moment.")
        self._worldMaterial = scene.getWorldEnvironment().material

    def _propagate(self, sceneLogger: Logger = None):
        photons = PhotonCL(self._positions, self._directions)
        material = MaterialCL(self._worldMaterial)
        logger = LoggerCL(size=self._requiredLoggerSize())
        randomFloat = RandomFloatCL(size=self._N)
        randomSeed = RandomSeedCL(size=self._N)

        t0 = time.time_ns()
        self._program.launchKernel(kernelName='propagate', N=self._N,
                                   arguments=[self._N, self._weightThreshold,
                                              photons, material, logger, randomFloat, randomSeed])

        log = self._program.getData(logger)

        t1 = time.time_ns()
        print("CLPhotons.propagate: {} s".format((t1 - t0) / 1e9))

        if sceneLogger:
            sceneLogger.logDataPointArray(log, InteractionKey("universe", None))

    def _requiredLoggerSize(self) -> int:
        return int(-np.log(self._weightThreshold) / self._worldMaterial.getAlbedo()) * self._N
