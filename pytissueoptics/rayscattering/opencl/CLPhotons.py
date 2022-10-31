import os
import time

from pytissueoptics.rayscattering.opencl.CLParameters import CLParameters

try:
    import pyopencl as cl

    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
import numpy as np

from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.CLObjects import PhotonCL, MaterialCL, DataPointCL, SeedCL
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
        self._sceneLogger = None

    def setContext(self, scene: RayScatteringScene, environment: Environment, logger: Logger = None):
        if type(scene) is not InfiniteTissue:
            raise TypeError("OpenCL propagation is only supported for InfiniteTissue for the moment.")

        worldMaterial = environment.material
        self._materials = [worldMaterial]
        self._sceneLogger = logger

    def propagate(self):
        program = CLProgram(sourcePath=PROPAGATION_SOURCE_PATH)
        params = CLParameters(1e8, 10, 10000)

        if params.photonAmount >= self._N:
            params.photonAmount = self._N

        kernelPhotons = PhotonCL(self._positions[0:params.photonAmount], self._directions[0:params.photonAmount])
        photonPool = PhotonCL(self._positions[params.photonAmount:], self._directions[params.photonAmount:])
        photonPool.make(program.device)
        materials = MaterialCL(self._materials)
        seeds = SeedCL(params.photonAmount)
        logger = DataPointCL(size=params.maxLoggableInteractions)

        photonCount = 0
        batchCount = 0
        logArrays = []
        t0 = time.time_ns()

        while photonCount < self._N:
            t1 = time.time_ns()
            program.launchKernel(kernelName="propagate", N=np.int32(params.workItemAmount),
                                 arguments=[np.int32(params.photonsPerWorkItem),
                                            np.int32(params.maxLoggableInteractionsPerWorkItem),
                                            self._weightThreshold, np.int32(params.workItemAmount), kernelPhotons,
                                            materials, seeds, logger])
            t2 = time.time_ns()
            logArrays.append(program.getData(logger))
            logger.reset()
            program.getData(kernelPhotons)
            batchPhotonCount, photonCount = self._replaceFullyPropagatedPhotons(kernelPhotons, photonPool,
                                                                                photonCount, params.photonAmount)

            self._showProgress(photonCount, batchPhotonCount, batchCount, t0, t1, t2, params.photonAmount)
            params.photonAmount = kernelPhotons.length
            batchCount += 1

        self._logDataFromLogArrays(logArrays)

    def _replaceFullyPropagatedPhotons(self, kernelPhotons: PhotonCL, photonPool: PhotonCL,
                                       photonCount: int, currentKernelLength: int) -> (int, int):
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
        return batchPhotonCount, photonCount

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
