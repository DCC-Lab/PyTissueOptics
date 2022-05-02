import os
import time
import pyopencl as cl
import numpy as np
from numpy.lib import recfunctions as rfn

from pytissueoptics.rayscattering.opencl.types import makePhotonType, makeMaterialType, makeLoggerType
from pytissueoptics.rayscattering.tissues import InfiniteTissue
from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Logger
from pytissueoptics.scene.logger import InteractionKey


class CLPhotons:
    def __init__(self, positions: np.ndarray, directions: np.ndarray, N: int, weightThreshold: float = 0.0001):
        self._sourceFolderPath = os.path.dirname(os.path.abspath(__file__)) + "{0}src{0}".format(os.sep)
        self._positions = positions
        self._directions = directions
        self._N = N
        self._weightThreshold = np.float32(weightThreshold)
        self._logger = None

        self._context = cl.create_some_context()
        self._mainQueue = cl.CommandQueue(self._context)
        self._device = self._context.devices[0]
        self._program = None

        self._HOST_photons, self._DEVICE_photons, self._photon_dtype, self._c_decl_photon = None, None, None, None
        self._HOST_material, self._DEVICE_material, self._material_dtype, self._c_decl_mat = None, None, None, None
        self._HOST_logger, self._DEVICE_logger, self._logger_dtype, self._c_decl_logger = None, None, None, None
        self._HOST_randomSeed, self._DEVICE_randomSeed, self._HOST_randomFloat, self._DEVICE_randomFloat = None, None, None, None

        self._makeTypes()

    def _extractFromScene(self, scene: RayScatteringScene):
        if type(scene) is not InfiniteTissue:
            raise TypeError("OpenCL propagation is only supported for InfiniteTissue for the moment.")
        self._worldMaterial = scene.getWorldEnvironment().material

    def _buildProgram(self):
        randomSource = open(os.path.join(self._sourceFolderPath, "random.c")).read()
        vectorSource = open(os.path.join(self._sourceFolderPath, "vectorOperators.c")).read()
        propagationSource = open(os.path.join(self._sourceFolderPath, "propagation.c")).read()

        self._program = cl.Program(self._context, self._c_decl_photon + self._c_decl_mat + self._c_decl_logger +
                                   randomSource + vectorSource + propagationSource).build()

    def prepareAndPropagate(self, scene: RayScatteringScene, logger: Logger):
        self._logger = logger
        self._extractFromScene(scene)
        self._makeBuffers()
        self._buildProgram()
        self._propagate()

    def _propagate(self):
        t0 = time.time_ns()
        datasize = np.uint32(len(self._HOST_photons))
        self._program.propagate(self._mainQueue, self._HOST_photons.shape, None, datasize, self._weightThreshold,
                                self._DEVICE_photons,
                                self._DEVICE_material, self._DEVICE_logger, self._DEVICE_randomFloat,
                                self._DEVICE_randomSeed)
        self._mainQueue.finish()
        cl.enqueue_copy(self._mainQueue, dest=self._HOST_logger, src=self._DEVICE_logger)
        t1 = time.time_ns()
        print("CLPhotons.propagate: {} s".format((t1 - t0) / 1e9))

        log = rfn.structured_to_unstructured(self._HOST_logger)
        self._logger.logDataPointArray(log, InteractionKey("universe", None))

    def _makeTypes(self):
        self._photon_dtype, self._c_decl_photon = makePhotonType(self._device)
        self._material_dtype, self._c_decl_mat = makeMaterialType(self._device)
        self._logger_dtype, self._c_decl_logger = makeLoggerType(self._device)

    def _makeBuffers(self):
        self._makePhotonsBuffer()
        self._makeMaterialsBuffer()
        self._makeLoggerBuffer()
        self._makeRandomBuffer()

    def _makePhotonsBuffer(self):
        photonsPrototype = np.zeros(self._N, dtype=self._photon_dtype)
        photonsPrototype = rfn.structured_to_unstructured(photonsPrototype)
        photonsPrototype[:, 0:3] = self._positions[:, ::]
        photonsPrototype[:, 4:7] = self._directions[:, ::]
        photonsPrototype[:, 12] = 1.0
        photonsPrototype[:, 13] = 0
        self._HOST_photons = rfn.unstructured_to_structured(photonsPrototype, self._photon_dtype)
        self._DEVICE_photons = cl.Buffer(self._context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                         hostbuf=self._HOST_photons)

    def _makeRandomBuffer(self):
        self._HOST_randomSeed = np.random.randint(low=0, high=2 ** 32 - 1, size=self._N,
                                                  dtype=cl.cltypes.uint)
        self._DEVICE_randomSeed = cl.Buffer(self._context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                            hostbuf=self._HOST_randomSeed)
        self._HOST_randomFloat = np.empty(self._N, dtype=cl.cltypes.float)
        self._DEVICE_randomFloat = cl.Buffer(self._context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                             hostbuf=self._HOST_randomFloat)

    def _makeMaterialsBuffer(self):
        self._HOST_material = np.empty(1, dtype=self._material_dtype)
        self._HOST_material["mu_s"] = np.float32(self._worldMaterial.mu_s)
        self._HOST_material["mu_a"] = np.float32(self._worldMaterial.mu_a)
        self._HOST_material["mu_t"] = np.float32(self._worldMaterial.mu_t)
        self._HOST_material["g"] = np.float32(self._worldMaterial.g)
        self._HOST_material["n"] = np.float32(self._worldMaterial.index)
        self._HOST_material["albedo"] = np.float32(self._worldMaterial.getAlbedo())
        self._HOST_material["material_id"] = np.uint32(0)
        self._DEVICE_material = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                          hostbuf=self._HOST_material)

    def _makeLoggerBuffer(self):
        loggerSize = int(
            -np.log(self._weightThreshold) / self._worldMaterial.getAlbedo()) * self._N
        self._HOST_logger = np.empty(loggerSize, dtype=self._logger_dtype)
        self._DEVICE_logger = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                        hostbuf=self._HOST_logger)
