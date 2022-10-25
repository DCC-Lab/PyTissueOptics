import os
import time

from pytissueoptics.rayscattering.opencl.CLScene import CLScene

try:
    import pyopencl as cl

    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
import numpy as np

from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.CLObjects import PhotonCL, DataPointCL, SeedCL, RandomNumberCL
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
        self._initialMaterial = None
        self._initialSolid = None

        self._scene = None
        self._requiredLoggerSize = None
        self._sceneLogger = None

    def setContext(self, scene: RayScatteringScene, environment: Environment, logger: Logger = None):
        self._scene = scene
        self._sceneLogger = logger
        self._initialMaterial = environment.material
        self._initialSolid = environment.solid

        safetyFactor = 1.8
        materials = scene.getMaterials()
        avgAlbedo = np.mean([m.getAlbedo() for m in materials])
        avgInteractions = int(-np.log(self._weightThreshold) / avgAlbedo)
        print(f"Approximate avgInteractions = {avgInteractions}")
        self._requiredLoggerSize = self._N * int(safetyFactor * avgInteractions)

    def propagate(self):
        t0 = time.time()

        scene = CLScene(self._scene, self._N)

        photons = PhotonCL(self._positions, self._directions,
                           materialID=scene.getMaterialID(self._initialMaterial),
                           solidID=scene.getSolidID(self._initialSolid))
        logger = DataPointCL(size=self._requiredLoggerSize)
        randomNumbers = RandomNumberCL(size=self._N)
        seeds = SeedCL(size=self._N)

        program = CLProgram(sourcePath=PROPAGATION_SOURCE_PATH)

        t1 = time.time()
        print(f"OpenCL Propagation Timer: \n ... {t1 - t0:.3f} s. [CLObjects initialization]")
        maxInteractions = np.uint32(self._requiredLoggerSize // self._N)
        program.launchKernel(kernelName='propagate', N=self._N,
                             arguments=[self._N, maxInteractions, self._weightThreshold,
                                        photons, scene.materials, logger, randomNumbers, seeds,
                                        scene.nSolids, scene.solids, scene.surfaces, scene.triangles, scene.vertices,
                                        scene.solidCandidates])
        t2 = time.time()
        log = program.getData(logger)
        t3 = time.time()
        print(f" ... {t3 - t2:.3f} s. [CL logger copy]")

        if not self._sceneLogger:
            return

        localSort = True

        if localSort:
            self._logWithLocalSort(log, scene)
        else:
            self._log(log, scene)

        t4 = time.time()
        print(f">>> ({t4 - t0:.3f} s.)")

        # pts = self._sceneLogger.getDataPoints()
        # interactionCount = sum([p[0] != 0 for p in pts])
        # print(f"True avgInteractions: {interactionCount / self._N}")

    def _logWithLocalSort(self, log, scene):
        """
        Temporary CPU workUnit-wise sort. Todo: move to OpenCL
        """
        t3 = time.time()
        keyIndices = []
        batchSize = log.shape[0] // 1000
        for i in range(0, log.shape[0], batchSize):
            ba, bb = i, i + batchSize
            batchLog = log[ba:bb]
            batchKeyIndices = {}

            batchLog = batchLog[batchLog[:, 4].argsort()]
            solidChanges = np.unique(batchLog[:, 4], return_index=True)[1]
            solidChanges = np.append(solidChanges, len(batchLog))
            for i in range(len(solidChanges) - 1):
                solidID = batchLog[solidChanges[i], 4]
                if solidID == 0:
                    continue
                a, b = solidChanges[i], solidChanges[i + 1]
                batchLog[a:b] = batchLog[a:b][batchLog[a:b, 5].argsort()]
                surfaceChanges = np.unique(batchLog[a:b, 5], return_index=True)[1]
                surfaceChanges = np.append(surfaceChanges, b - a)
                for j in range(len(surfaceChanges) - 1):
                    surfaceID = batchLog[a + surfaceChanges[j], 5]
                    c, d = surfaceChanges[j], surfaceChanges[j + 1]
                    batchKeyIndices[(int(solidID), int(surfaceID))] = (a + c, a + d)
            keyIndices.append(batchKeyIndices)
            log[ba:bb] = batchLog

        tParse = time.time()
        print(f" ... {tParse - t3:.3f} s. [Logger parsing]")

        keyData = {}
        for i, batchKeyIndices in enumerate(keyIndices):
            bn = i * batchSize
            for keyIDs, indices in batchKeyIndices.items():
                if len(indices) == 0:
                    continue
                key = InteractionKey(scene.getSolidLabel(keyIDs[0]), scene.getSurfaceLabel(keyIDs[0], keyIDs[1]))
                points = log[indices[0] + bn: indices[1] + bn, :4]
                if key not in keyData:
                    keyData[key] = [points]
                else:
                    keyData[key].append(points)

        for key, points in keyData.items():
            points = np.concatenate(points)
            self._sceneLogger.logDataPointArray(points, key)

        t4 = time.time()
        print(f" ... {t4 - tParse:.3f} s. [Transfer to scene logger]")

    def _log(self, log, scene):
        t3 = time.time()
        log = log[log[:, 4] != 0]

        keyToIndices = {}
        solidIDs = scene.getSolidIDs()
        for solidID in solidIDs:
            surfaceIDs = scene.getSurfaceIDs(solidID)
            for surfaceID in surfaceIDs:
                key = InteractionKey(scene.getSolidLabel(solidID), scene.getSurfaceLabel(solidID, surfaceID))
                surfaceIndices = np.asarray(log[:, 5] == surfaceID).nonzero()[0]
                surfacePoints = log[surfaceIndices]
                solidSurfaceIndices = np.asarray(surfacePoints[:, 4] == solidID).nonzero()[0]
                keyToIndices[key] = surfaceIndices[solidSurfaceIndices]

        tParse = time.time()
        print(f" ... {tParse - t3:.3f} s. [Logger parsing]")

        for key, indices in keyToIndices.items():
            if len(indices) == 0:
                continue
            points = log[indices, :4]

            self._sceneLogger.logDataPointArray(points, key)

        t4 = time.time()
        print(f" ... {t4 - tParse:.3f} s. [Transfer to scene logger]")

"""
30k photons
transfer to logger time:
    No sort:
        3.1s
    Global sort:
        0.16s
    Local sort:
        0.49s with 30k units
        0.24s with 1k units
"""
