import dataclasses
import os
import time

try:
    import pyopencl as cl

    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
import numpy as np

from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.CLObjects import PhotonCL, MaterialCL, DataPointCL, SeedCL
from pytissueoptics.scene.logger import InteractionKey

PROPAGATION_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'propagation.c')


@dataclasses.dataclass
class CLPropagationParameters:
    DataPointSize = 16
    PhotonSize = 64
    SeedSize = 16

    maxWorkGroupSize: int
    maxWorkItemDimensions: int
    maxMemoryAllocationSize: int
    maxLoggableInteractions: int
    maxLoggableInteractionsPerWorkItem: int
    maxPhotonsPerWorkItem: int

    currentLoggableInteractions: int
    currentLoggableInteractionsPerWorkItem: int
    currentPhotonsPerWorkItem: int


class CLPropagatorOptimizer:
    def __init__(self, photons: PhotonCL):
        self._photons = photons
        self._materials = None
        self._optimizedParameters = None
        self._currentParameters = CLPropagationParameters(0, 0, 0, 0, 0)

    def _evaluateOptimalParameters(self):
        pass

    def _searchMemoryAllocLimit(self):
        pass

    def _searchWorkUnitLimit(self):
        pass

    def _searchOptimalLoggerSize(self):
        pass

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
        workUnits = 3024
        photonsPerUnit = 20
        maxLoggerLength = 10000000
        maxUnitLoggerLength = np.int(maxLoggerLength / workUnits)
        maxKernelLength = photonsPerUnit * workUnits

        if maxKernelLength >= self._N:
            currentKernelLength = self._N
        else:
            currentKernelLength = maxKernelLength

        kernelPhotons = PhotonCL(self._positions[0:currentKernelLength], self._directions[0:currentKernelLength])
        photonPool = PhotonCL(self._positions[currentKernelLength:], self._directions[currentKernelLength:])
        photonPool.make(program.device)
        materials = MaterialCL(self._materials)
        seeds = SeedCL(maxKernelLength)

        photonCount = 0
        batchCount = 0
        logArrays = []
        t0 = time.time_ns()

        while photonCount < self._N:
            maxUnitPhotons = np.int32(np.ceil(currentKernelLength / workUnits))
            if maxUnitPhotons == 1:
                workUnits = currentKernelLength

            logger = DataPointCL(size=maxLoggerLength)
            t1 = time.time_ns()
            program.launchKernel(kernelName="propagate", N=np.int32(workUnits),
                                 arguments=[np.int32(maxUnitPhotons), np.int32(maxUnitLoggerLength),
                                            self._weightThreshold, np.int32(workUnits), kernelPhotons,
                                            materials, seeds, logger])
            t2 = time.time_ns()

            logArrays.append(program.getData(logger))
            program.getData(kernelPhotons)
            batchPhotonCount = self._replaceFullyPropagatedPhotons(kernelPhotons, photonPool,
                                                                   photonCount, currentKernelLength)

            self._showProgress(photonCount, batchPhotonCount, batchCount, t0, t1, t2, currentKernelLength)
            currentKernelLength = kernelPhotons.size
            batchCount += 1

        self._logDataFromLogArrays(logArrays)

    def _replaceFullyPropagatedPhotons(self, kernelPhotons: PhotonCL, photonPool: PhotonCL,
                                       photonCount: int, currentKernelLength: int) -> int:
        photonsToRemove = []
        batchPhotonCount = 0
        for i in range(currentKernelLength):
            if kernelPhotons.hostBuffer[i]["weight"] == 0:
                newPhotonsRemaining = photonCount + currentKernelLength < self._N
                if newPhotonsRemaining:
                    kernelPhotons.hostBuffer[i] = photonPool.hostBuffer[photonCount]
                else:
                    photonsToRemove.append(i)
                photonCount += 1
                batchPhotonCount += 1
        kernelPhotons.hostBuffer = np.delete(kernelPhotons.hostBuffer, photonsToRemove)
        return batchPhotonCount

    def _logDataFromLogArrays(self, logArrays):
        t4 = time.time_ns()
        log = np.concatenate(logArrays)
        print(f"Concatenate multiple logger arrays: {(time.time_ns() - t4) / 1e9}s")
        if self._sceneLogger:
            t5 = time.time_ns()
            self._sceneLogger.logDataPointArray(log, InteractionKey("universe", None))
            print(f"Transfer logging data to Logger object.: {(time.time_ns() - t5) / 1e9}s")

    def _showProgress(self, photonCount: int, localPhotonCount: int, batchCount: int, t0: float, t1: float,
                      t2: float, currentKernelLength: int):
        print(f"{photonCount}/{self._N}\t{((time.time_ns() - t0) / 1e9)} s\t ::"
              f" Batch #{batchCount}\t :: {currentKernelLength} \t{((t2 - t1) / 1e9):.2f} s\t ::"
              f" ETA: {((time.time_ns() - t0) / 1e9) * (np.float64(self._N) / np.float64(photonCount)):.2f} s\t ::"
              f"({(photonCount * 100 / self._N):.2f}%) \t ::"
              f"Eff: {np.float64((t2 - t1) / 1e3) / localPhotonCount:.2f} us/photon\t ::"
              f"Finished propagation: {localPhotonCount}")
