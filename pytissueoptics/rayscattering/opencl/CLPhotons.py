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

        program = CLProgram(sourcePath=PROPAGATION_SOURCE_PATH)
        workUnits = program.max_compute_units
        # totalMemory = program.global_memory_size
        photonsPerUnit = 1000  # to be changed with a scout batch
        kernelSize = photonsPerUnit * workUnits
        maxLoggerSize = 150 * 10 ** 6  # first should be calculated then, to be optimized after the scout batch
        maxInteractions = np.int32(maxLoggerSize / 16)
        # maxInteractionsPerWorkUnit = maxInteractions / workUnits

        kernelPhotons = PhotonCL(self._positions[0:kernelSize], self._directions[0:kernelSize])
        poolOfPhotons = PhotonCL(self._positions[kernelSize:], self._directions[kernelSize:])
        poolOfPhotons.make(program.device)

        materials = MaterialCL(self._materials)
        seeds = SeedCL(size=kernelSize)

        photonCount = 0
        t0 = time.time_ns()
        while photonCount <= self._N:
            logger = DataPointCL(size=100000000)
            t2 = time.time_ns()
            program.launchKernel(kernelName="propagate",
                                 N=np.int32(kernelSize), arguments=[np.int32(kernelSize), np.int32(maxInteractions), self._weightThreshold,
                                                          kernelPhotons, materials, seeds, logger])
            t1 = time.time_ns()
            print(f"CLPhotons.propagate: {kernelSize} in {((t1 - t2) / 1e9)} s")


            log = program.getData(logger)
            print(len(log))
            # print(sizeof(log))
            if self._sceneLogger:
                self._sceneLogger.logDataPointArray(log, InteractionKey("universe", None))
            #print(program.getData(kernelPhotons))
            #print(np.array(kernelPhotons.hostBuffer)[0]["weight"])
            for i in range(kernelSize):
                if kernelPhotons.hostBuffer[i]["weight"] == 0:
                    newPhotonsRemaining = photonCount + kernelSize < self._N
                    if newPhotonsRemaining:
                        kernelPhotons.hostBuffer[i] = poolOfPhotons.hostBuffer[photonCount]
                        # print(kernelPhotons.hostBuffer[i])
                    photonCount += 1
            t3 = time.time_ns()
            print(f"Total Propagation: {photonCount * 100 / self._N}% in {((t3 - t0) / 1e9)}  s")

