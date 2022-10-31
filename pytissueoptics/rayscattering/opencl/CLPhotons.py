import os
import time

try:
    import pyopencl as cl

    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
import numpy as np

from pytissueoptics.rayscattering.opencl.CLKeyLog import CLKeyLog
from pytissueoptics.rayscattering.opencl.CLScene import CLScene
from pytissueoptics.rayscattering.opencl.CLParameters import CLParameters
from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.CLObjects import PhotonCL, DataPointCL, SeedCL
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
        self._sceneLogger = None

    def setContext(self, scene: RayScatteringScene, environment: Environment, logger: Logger = None):
        self._scene = scene
        self._sceneLogger = logger
        self._initialMaterial = environment.material
        self._initialSolid = environment.solid

    def propagate(self):
        t0 = time.time()

        scene = CLScene(self._scene, self._N)

        program = CLProgram(sourcePath=PROPAGATION_SOURCE_PATH)
        params = CLParameters()

        t1 = time.time()
        print(f"OpenCL Propagation Timer: \n ... {t1 - t0:.3f} s. [CLScene initialization]")

        if params.photonAmount >= self._N:
            params.photonAmount = self._N

        kernelPhotons = PhotonCL(self._positions[0:params.photonAmount], self._directions[0:params.photonAmount],
                                 materialID=scene.getMaterialID(self._initialMaterial), solidID=scene.getSolidID(self._initialSolid))
        photonPool = PhotonCL(self._positions[params.photonAmount:], self._directions[params.photonAmount:],
                              materialID=scene.getMaterialID(self._initialMaterial), solidID=scene.getSolidID(self._initialSolid))
        photonPool.make(program.device)
        seeds = SeedCL(params.photonAmount)

        photonCount = 0
        batchCount = 0
        logArrays = []
        t0 = time.time_ns()

        while photonCount < self._N:
            logger = DataPointCL(size=params.maxLoggableInteractions)
            t1 = time.time_ns()
            program.launchKernel(kernelName="propagate", N=np.int32(params.workItemAmount),
                                 arguments=[np.int32(params.photonsPerWorkItem),
                                            np.int32(params.maxLoggableInteractionsPerWorkItem),
                                            self._weightThreshold, np.int32(params.workItemAmount), kernelPhotons,
                                            scene.materials, scene.nSolids, scene.solids, scene.surfaces, scene.triangles,
                                            scene.vertices, scene.solidCandidates, seeds, logger])
            t2 = time.time_ns()
            logArrays.append(program.getData(logger))
            program.getData(kernelPhotons)
            batchPhotonCount, photonCount = self._replaceFullyPropagatedPhotons(kernelPhotons, photonPool,
                                                                                photonCount, params.photonAmount)

            self._showProgress(photonCount, batchPhotonCount, batchCount, t0, t1, t2, params.photonAmount)
            params.photonAmount = kernelPhotons.length
            batchCount += 1

        self._logDataFromLogArrays(logArrays, scene)

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

    def _logDataFromLogArrays(self, logArrays, sceneCL):
        t4 = time.time_ns()
        log = np.concatenate(logArrays)
        print(f"Concatenate multiple logger arrays: {(time.time_ns() - t4) / 1e9}s")
        if not self._sceneLogger:
            return

        t5 = time.time_ns()
        keyLog = CLKeyLog(log, sceneCL=sceneCL)
        keyLog.toSceneLogger(self._sceneLogger)
        print(f"Transfer logging data to Logger object.: {(time.time_ns() - t5) / 1e9}s")

    def _showProgress(self, photonCount: int, localPhotonCount: int, batchCount: int, t0: float, t1: float,
                      t2: float, currentKernelLength: int):
        print(f"{photonCount}/{self._N}\t{((time.time_ns() - t0) / 1e9)} s\t ::"
              f" Batch #{batchCount}\t :: {currentKernelLength} \t{((t2 - t1) / 1e9):.2f} s\t ::"
              f" ETA: {((time.time_ns() - t0) / 1e9) * (np.float64(self._N) / np.float64(photonCount)):.2f} s\t ::"
              f"({(photonCount * 100 / self._N):.2f}%) \t ::"
              f"Eff: {np.float64((t2 - t1) / 1e3) / localPhotonCount:.2f} us/photon\t ::"
              f"Finished propagation: {localPhotonCount}")
