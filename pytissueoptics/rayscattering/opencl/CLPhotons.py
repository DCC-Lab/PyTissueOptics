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
        workUnits = 100
        photonsPerUnit = 200
        kernelLength = photonsPerUnit * workUnits
        maxLoggerLength = 50000000

        if kernelLength >= self._N:
            currentKernelLength = self._N
        else:
            currentKernelLength = kernelLength

        maxUnitLoggerLength = maxLoggerLength / workUnits
        kernelPhotons = PhotonCL(self._positions[0:currentKernelLength], self._directions[0:currentKernelLength])
        photonPool = PhotonCL(self._positions[currentKernelLength:], self._directions[currentKernelLength:])
        photonPool.make(program.device)
        materials = MaterialCL(self._materials)

        photonCount = 0
        batchCount = 0
        t0 = time.time_ns()
        logArrays = []

        while photonCount < self._N:

            maxUnitPhotons = currentKernelLength / workUnits
            seeds = SeedCL(size=currentKernelLength)
            logger = DataPointCL(size=maxLoggerLength)
            t2 = time.time_ns()
            program.launchKernel(kernelName="propagate", N=np.int32(workUnits),
                                 arguments=[np.int32(maxUnitPhotons), np.int32(maxUnitLoggerLength),
                                            self._weightThreshold, np.int32(workUnits), kernelPhotons,
                                            materials, seeds, logger])
            t1 = time.time_ns()

            logArrays.append(program.getData(logger))

            program.getData(kernelPhotons)
            photonsToRemove = []
            localPhotonCount = 0
            for i in range(currentKernelLength):
                if kernelPhotons.hostBuffer[i]["weight"] == 0:
                    newPhotonsRemaining = photonCount + currentKernelLength < self._N
                    if newPhotonsRemaining:
                        kernelPhotons.hostBuffer[i] = photonPool.hostBuffer[photonCount]
                    else:
                        photonsToRemove.append(i)
                    photonCount += 1
                    localPhotonCount += 1
            kernelPhotons.hostBuffer = np.delete(kernelPhotons.hostBuffer, photonsToRemove)

            t3 = time.time_ns()
            print(f"{photonCount}/{self._N}\t{((t3 - t0) / 1e9)} s\t ::"
                  f" Batch #{batchCount}\t :: {currentKernelLength} \t{((t1 - t2) / 1e9):.2f} s\t ::"
                  f" ETA: {((t3 - t0) / 1e9) * (self._N / photonCount - 1):.2f} s\t ::"
                  f"({(photonCount * 100 / self._N):.2f}%) \t ::"
                  f"Eff: {((t1 - t2) / 1e3)/localPhotonCount:.2f} us/photon\t ::"
                  f"Finished propagation: {localPhotonCount}")

            currentKernelLength = kernelPhotons.size
            batchCount += 1

        t4 = time.time_ns()
        log = np.concatenate(logArrays)
        print(f"create time: {(time.time_ns() - t4) / 1e9}s")
        if self._sceneLogger:
            t5 = time.time_ns()
            self._sceneLogger.logDataPointArray(log, InteractionKey("universe", None))
            print(f"log time: {(time.time_ns()-t5)/1e9}s")

