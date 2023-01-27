import os
import time

import numpy as np

from pytissueoptics.rayscattering.opencl import WEIGHT_THRESHOLD
from pytissueoptics.rayscattering.opencl.utils import CLKeyLog, CLParameters
from pytissueoptics.rayscattering.opencl.CLScene import CLScene
from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.buffers.seedCL import SeedCL
from pytissueoptics.rayscattering.opencl.buffers.dataPointCL import DataPointCL
from pytissueoptics.rayscattering.opencl.buffers.photonCL import PhotonCL
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.scene.logger.logger import Logger
from pytissueoptics.scene.geometry import Environment

PROPAGATION_SOURCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'propagation.c')


class CLPhotons:
    def __init__(self, positions: np.ndarray, directions: np.ndarray):
        assert positions.shape == directions.shape, "Positions and directions must have the same shape."
        self._positions = positions
        self._directions = directions
        self._N = np.uint32(len(positions))
        self._weightThreshold = np.float32(WEIGHT_THRESHOLD)
        self._initialMaterial = None
        self._initialSolid = None

        self._scene = None
        self._sceneLogger = None

    def setContext(self, scene: ScatteringScene, environment: Environment, logger: Logger = None):
        self._scene = scene
        self._sceneLogger = logger
        self._initialMaterial = environment.material
        self._initialSolid = environment.solid

    def propagate(self, IPP: float, verbose: bool = False):
        assert self._scene is not None, "Context must be set before propagation."
        program = CLProgram(sourcePath=PROPAGATION_SOURCE_PATH)
        params = CLParameters(self._N, AVG_IT_PER_PHOTON=IPP)

        scene = CLScene(self._scene, params.workItemAmount)

        kernelPhotons = PhotonCL(self._positions[0:params.maxPhotonsPerBatch], self._directions[0:params.maxPhotonsPerBatch],
                                 materialID=scene.getMaterialID(self._initialMaterial), solidID=scene.getSolidID(self._initialSolid))
        photonPool = PhotonCL(self._positions[params.maxPhotonsPerBatch:], self._directions[params.maxPhotonsPerBatch:],
                              materialID=scene.getMaterialID(self._initialMaterial), solidID=scene.getSolidID(self._initialSolid))
        photonPool.make(program.device)
        seeds = SeedCL(params.maxPhotonsPerBatch)
        logger = DataPointCL(size=params.maxLoggableInteractions)

        photonCount = 0
        batchCount = 0
        t0 = time.time_ns()

        while photonCount < self._N:
            t1 = time.time_ns()
            program.launchKernel(kernelName="propagate", N=np.int32(params.workItemAmount),
                                 arguments=[np.int32(params.photonsPerWorkItem),
                                            np.int32(params.maxLoggableInteractionsPerWorkItem),
                                            self._weightThreshold, np.int32(params.workItemAmount), kernelPhotons,
                                            scene.materials, scene.nSolids, scene.solids, scene.surfaces, scene.triangles,
                                            scene.vertices, scene.solidCandidates, seeds, logger])
            t2 = time.time_ns()

            log = program.getData(logger)
            self._translateToSceneLogger(log, scene)

            logger.reset()
            program.getData(kernelPhotons)
            batchPhotonCount, photonCount = self._replaceFullyPropagatedPhotons(kernelPhotons, photonPool,
                                                                                photonCount, params.maxPhotonsPerBatch)

            self._showProgress(photonCount, batchPhotonCount, batchCount, t0, t1, t2, params.maxPhotonsPerBatch, verbose)
            params.maxPhotonsPerBatch = kernelPhotons.length
            batchCount += 1

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

    def _translateToSceneLogger(self, log, sceneCL, verbose: bool = False):
        if not self._sceneLogger:
            return

        t5 = time.time()
        keyLog = CLKeyLog(log, sceneCL=sceneCL)
        keyLog.toSceneLogger(self._sceneLogger)
        if verbose:
            print(f" ... {time.time() - t5:.3f} s. [Translate OpenCL Logger to Scene Logger]")

    def _showProgress(self, photonCount: int, localPhotonCount: int, batchCount: int, t0: float, t1: float,
                      t2: float, currentKernelLength: int, verbose: bool = True):
        if not verbose:
            return
        if localPhotonCount == 0:
            print(f"{photonCount}/{self._N}\t{((time.time_ns() - t0) / 1e9):.4f} s\t ::"
                  f" Batch #{batchCount}\t :: {currentKernelLength} \t{((t2 - t1) / 1e9):.2f} s\t ::"
                  f"({(photonCount * 100 / self._N):.2f}%) \t ::"
                  f"Finished propagation: {localPhotonCount}")
        else:
            print(f"{photonCount}/{self._N}\t{((time.time_ns() - t0) / 1e9):.4f} s\t ::"
                  f" Batch #{batchCount}\t :: {currentKernelLength} \t{((t2 - t1) / 1e9):.2f} s\t ::"
                  f" ETA: {((time.time_ns() - t0) / 1e9) * (np.float64(self._N) / np.float64(photonCount)):.2f} s\t ::"
                  f"({(photonCount * 100 / self._N):.2f}%) \t ::"
                  f"Eff: {np.float64((t2 - t1) / 1e3) / localPhotonCount:.2f} us/photon\t ::"
                  f"Finished propagation: {localPhotonCount}")
