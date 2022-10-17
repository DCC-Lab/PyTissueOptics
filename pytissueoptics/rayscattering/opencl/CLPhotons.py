import os
import time

try:
    import pyopencl as cl

    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
import numpy as np

from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.CLObjects import PhotonCL, MaterialCL, DataPointCL, SeedCL, RandomNumberCL
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
        if type(scene) is not InfiniteTissue:
            raise TypeError("OpenCL propagation is only supported for InfiniteTissue for the moment.")

        worldMaterial = environment.material
        self._materials = [worldMaterial]
        self._sceneLogger = logger

        safetyFactor = 1.8
        avgInteractions = int(-np.log(self._weightThreshold) / worldMaterial.getAlbedo())
        self._requiredLoggerSize = self._N * int(safetyFactor * avgInteractions)

    def propagate(self):
        """
        The propagation code on the CPU must manage dynamic batching of the photons, because the data generated
        by the photon weight deposition is too large to be handled at once on most of the parallel computing
        devices. Thus, we calculate the maximum size of the logger from the device total memory and subtract
        the photon and temporary variable size.

        It is estimated that the maximum speed increase for this purpose will be obtained by using 1 work-group
        with as many work-item as possible, since all the work-item launch the same kernel. The important metrics here:
        - photonsPerUnit =  N / workUnits
        - photonsPerBatch = 10 (A number that needs to be  played with (probably by sending a first batch
                                and trying to estimate the average interaction per photon)
        - maxLoggerSize = globalMemorySize - photonSize - geometryObjects - materialsObjects
        - maxLoggableInteractionsPerUnit = maxLoggerSize / workUnits

        Having the whole body of photons generated on the CPU is fast enough for know, thus
        (mutable) kernelPhotons = PhotonCL(positions[:kernelSize], directions[:kernelSize])
        to save space, lets exclude the initial kernel photons from the pool of (remaining) photons
        (immutable) poolOfPhotons = PhotonCL(positions[kernelSize:], directions[kernelSize:])

        Once a GPU batch as ended, when all the photons had X interactions, the weight of each photon is checked.
        The photons with a weight of 0 are replaced by photons from the photon pool. The other photons are sent
        on the gpu again to continue their propagation. A counter keeps count of which photon will be sent next.
        """


        photons = PhotonCL(self._positions, self._directions)
        materials = MaterialCL(self._materials)

        program = CLProgram(sourcePath=PROPAGATION_SOURCE_PATH)
        workUnits = program.max_compute_units
        totalMemory = program.global_memory_size

        photonsPerUnit = int(self._N / workUnits)
        photonsPerBatch = 10

        maxLoggerSize = 1500 * 10**6
        maxInteractions = maxLoggerSize / 16
        maxInteractionsPerWorkUnit = maxInteractions / workUnits
        logger = DataPointCL(size=maxInteractions)

        propagatedPhoton = 0
        tempPhotons = photons.hostBuffer[0:workUnits*photonsPerBatch]
        print(tempPhotons)

        # while propagatedPhoton < self._N:
        #     pass



    def propagateBatch(self):

        logger = DataPointCL(size=self._requiredLoggerSize)
        randomNumbers = RandomNumberCL(size=self._N)
        seeds = SeedCL(size=self._N)


        t0 = time.time_ns()
        maxInteractions = np.uint32(self._requiredLoggerSize // self._N)
        program.launchKernel(kernelName='propagate', N=self._N,
                             arguments=[self._N, maxInteractions, self._weightThreshold,
                                        photons, materials, logger, randomNumbers, seeds])
        log = program.getData(logger)
        t1 = time.time_ns()
        print("CLPhotons.propagate: {} s".format((t1 - t0) / 1e9))

        if self._sceneLogger:
            self._sceneLogger.logDataPointArray(log, InteractionKey("universe", None))

