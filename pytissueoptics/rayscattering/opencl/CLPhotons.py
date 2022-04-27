import os
import time

import pyopencl as cl
import pyopencl.tools
import numpy as np
from numpy.lib import recfunctions as rfn

from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.scene.logger import InteractionKey

class CLPhotons:
    def __init__(self, source: 'CLSource', worldMaterial: ScatteringMaterial, logger=None, weightThreshold=0.0001):
        self._kernelPath = os.path.dirname(os.path.abspath(__file__)) + "{}CLPhotons.c".format(os.sep)
        self._source = source
        self._worldMaterial = worldMaterial
        self._weightThreshold = np.float32(weightThreshold)
        self._logger = logger

        self._context = cl.create_some_context()
        self._mainQueue = cl.CommandQueue(self._context)
        self._device = self._context.devices[0]
        self._program = None

        self._HOST_photons, self._DEVICE_photons, self._photon_dtype, self._c_decl_photon = None, None, None, None
        self._HOST_material, self._DEVICE_material, self._material_dtype, self._c_decl_mat = None, None, None, None
        self._HOST_logger, self._DEVICE_logger, self._logger_dtype, self._c_decl_logger = None, None, None, None
        self._HOST_randomSeed, self._DEVICE_randomSeed, self._HOST_randomFloat, self._DEVICE_randomFloat = None, None, None, None

        self._makeTypes()
        self._makeBuffers()

    def _fillPhotons(self):
        position = self._source.position
        direction = self._source.direction
        datasize = np.uint32(len(self._HOST_photons))
        if type(self._source).__name__ == "CLPencilSource":
            self._program.fillPencilPhotonsBuffer(self._mainQueue, (datasize,), None,
                                                  self._DEVICE_photons,
                                                  self._DEVICE_randomSeed,
                                                  cl.cltypes.make_float4(position.x, position.y, position.z, 0),
                                                  cl.cltypes.make_float4(direction.x, direction.y, direction.z, 0))

        elif type(self._source).__name__ == "CLIsotropicSource":
            self._program.fillIsotropicPhotonsBuffer(self._mainQueue, (datasize,), None,
                                                     self._DEVICE_photons, self._DEVICE_randomSeed,
                                                     cl.cltypes.make_float4(position.x, position.y, position.z, 0))
        cl.enqueue_copy(self._mainQueue, self._HOST_photons, self._DEVICE_photons)

    def _buildProgram(self):
        self._program = cl.Program(self._context, self._c_decl_photon + self._c_decl_mat + self._c_decl_logger +
                                   open(self._kernelPath).read()).build()

    def propagate(self):
        t0 = time.time_ns()
        self._buildProgram()
        self._fillPhotons()
        datasize = np.uint32(len(self._HOST_photons))
        self._program.propagate(self._mainQueue, self._HOST_photons.shape, None, datasize, self._weightThreshold,
                                self._DEVICE_photons,
                                self._DEVICE_material, self._DEVICE_logger, self._DEVICE_randomFloat,
                                self._DEVICE_randomSeed)
        self._mainQueue.finish()
        cl.enqueue_copy(self._mainQueue, dest=self._HOST_logger, src=self._DEVICE_logger)
        t1 = time.time_ns()
        print("CLPhotons.propagate: {} s".format((t1 - t0)/ 1e9))
        # print(np.asarray(self._HOST_logger, dtype=np.float32).shape)
        print(self._HOST_logger)
        log = rfn.structured_to_unstructured(self._HOST_logger)
        print(log)
        self._logger.logDataPointArray(log, InteractionKey("world", None))

    def _makeTypes(self):
        def makePhotonType():
            photonStruct = np.dtype(
                [("position", cl.cltypes.float4),
                 ("direction", cl.cltypes.float4),
                 ("er", cl.cltypes.float4),
                 ("weight", cl.cltypes.float),
                 ("material_id", cl.cltypes.uint)])
            name = "photonStruct"
            photonStruct, c_decl_photon = cl.tools.match_dtype_to_c_struct(self._device, name, photonStruct)
            photon_dtype = cl.tools.get_or_register_dtype(name, photonStruct)
            return photon_dtype, c_decl_photon

        def makeMaterialType():
            materialStruct = np.dtype(
                [("mu_s", cl.cltypes.float),
                 ("mu_a", cl.cltypes.float),
                 ("mu_t", cl.cltypes.float),
                 ("g", cl.cltypes.float),
                 ("n", cl.cltypes.float),
                 ("albedo", cl.cltypes.float),
                 ("material_id", cl.cltypes.uint)])
            name = "materialStruct"
            materialStruct, c_decl_mat = cl.tools.match_dtype_to_c_struct(self._device, name, materialStruct)
            material_dtype = cl.tools.get_or_register_dtype(name, materialStruct)
            return material_dtype, c_decl_mat

        def makeLoggerType():
            loggerStruct = np.dtype(
                [("delta_weight", cl.cltypes.float),
                 ("x", cl.cltypes.float),
                 ("y", cl.cltypes.float),
                 ("z", cl.cltypes.float)])
            name = "loggerStruct"
            loggerStruct, c_decl_logger = cl.tools.match_dtype_to_c_struct(self._device, name, loggerStruct)
            logger_dtype = cl.tools.get_or_register_dtype(name, loggerStruct)
            return logger_dtype, c_decl_logger

        self._photon_dtype, self._c_decl_photon = makePhotonType()
        self._material_dtype, self._c_decl_mat = makeMaterialType()
        self._logger_dtype, self._c_decl_logger = makeLoggerType()

    def _makeBuffers(self):
        self._makePhotonsBuffer()
        self._makeMaterialsBuffer()
        self._makeLoggerBuffer()
        self._makeRandomBuffer()

    def _makePhotonsBuffer(self):
        self._HOST_photons = np.empty(self._source.N, dtype=self._photon_dtype)

        self._DEVICE_photons = cl.Buffer(self._context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                         hostbuf=self._HOST_photons)

    def _makeRandomBuffer(self):
        self._HOST_randomSeed = np.random.randint(low=0, high=2 ** 32 - 1, size=self._source.N, dtype=cl.cltypes.uint)
        self._DEVICE_randomSeed = cl.Buffer(self._context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                            hostbuf=self._HOST_randomSeed)
        self._HOST_randomFloat = np.empty(self._source.N, dtype=cl.cltypes.float)
        self._DEVICE_randomFloat = cl.Buffer(self._context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                             hostbuf=self._HOST_randomFloat)

    def _makeMaterialsBuffer(self):
        materials = [self._worldMaterial]
        self._HOST_material = np.empty(len(materials), dtype=self._material_dtype)
        for i, mat in enumerate(materials):
            self._HOST_material[i]["mu_s"] = np.float32(mat.mu_s)
            self._HOST_material[i]["mu_a"] = np.float32(mat.mu_a)
            self._HOST_material[i]["mu_t"] = np.float32(mat.mu_t)
            self._HOST_material[i]["g"] = np.float32(mat.g)
            self._HOST_material[i]["n"] = np.float32(mat.index)
            self._HOST_material[i]["albedo"] = np.float32(mat.getAlbedo())
            self._HOST_material[i]["material_id"] = np.uint32(i)
        self._DEVICE_material = cl.Buffer(self._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                          hostbuf=self._HOST_material)

    def _makeLoggerBuffer(self):
        loggerSize = int(-np.log(self._weightThreshold) / self._worldMaterial.getAlbedo()) * self._source.N
        self._HOST_logger = np.empty(loggerSize, dtype=self._logger_dtype)
        self._DEVICE_logger = cl.Buffer(self._context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                        hostbuf=self._HOST_logger)
