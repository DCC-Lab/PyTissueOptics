import os
import time

import numpy as np

from pytissueoptics.rayscattering.opencl import WEIGHT_THRESHOLD
from pytissueoptics.rayscattering.opencl.buffers.dataPointCL import DataPointCL
from pytissueoptics.rayscattering.opencl.buffers.photonCL import PhotonCL
from pytissueoptics.rayscattering.opencl.buffers.seedCL import SeedCL
from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.CLScene import CLScene
from pytissueoptics.rayscattering.opencl.utils import BatchTiming, CLKeyLog, CLParameters
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene
from pytissueoptics.scene.geometry import Environment
from pytissueoptics.scene.logger.logger import Logger

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
        
        if verbose:
            timing = BatchTiming(self._N)

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
            t3 = time.time_ns()
            self._translateToSceneLogger(log, scene)
            t4 = time.time_ns()

            logger.reset()
            program.getData(kernelPhotons, returnData=False)
            batchPhotonCount, photonCount = self._replaceFullyPropagatedPhotons(kernelPhotons, photonPool,
                                                                                photonCount, params.maxPhotonsPerBatch)
            if verbose:
                timing.recordBatch(batchPhotonCount, propagationTime=(t2 - t1), dataTransferTime=(t3 - t2),
                                   dataConversionTime=(t4 - t3), totalTime=(time.time_ns() - t1))

            params.maxPhotonsPerBatch = kernelPhotons.length
            batchCount += 1

    def _replaceFullyPropagatedPhotons(self, kernelPhotons: PhotonCL, photonPool: PhotonCL, photonCount: int,
                                       currentKernelLength: int) -> (int, int):
        photonsToReplace = np.where(kernelPhotons.hostBuffer["weight"] == 0)[0]
        batchPhotonCount = len(photonsToReplace)

        if batchPhotonCount == 0:
            return batchPhotonCount, photonCount

        newPhotonsRemaining = photonCount + currentKernelLength < self._N
        if newPhotonsRemaining:
            replacement_photons = photonPool.hostBuffer[photonCount:photonCount + batchPhotonCount]
            photonsToRemove = photonsToReplace[len(replacement_photons):]
            photonsToReplace = photonsToReplace[:len(replacement_photons)]
            kernelPhotons.hostBuffer[photonsToReplace] = replacement_photons
        else:
            photonsToRemove = photonsToReplace

        kernelPhotons.hostBuffer = np.delete(kernelPhotons.hostBuffer, photonsToRemove)

        photonCount += batchPhotonCount
        return batchPhotonCount, photonCount

    def _translateToSceneLogger(self, log, sceneCL):
        if not self._sceneLogger:
            return

        keyLog = CLKeyLog(log, sceneCL=sceneCL)
        keyLog.toSceneLogger(self._sceneLogger)
