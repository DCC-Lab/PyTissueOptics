import copy
import time
from typing import List, Tuple

import pyopencl as cl
import pyopencl.tools
import numpy as np

from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.scene import Vector

np.random.seed(2)


class CLPhotons:
    def __init__(self, source: 'CLSource', worldMaterial: ScatteringMaterial, logger=None, weightThreshold=0.0001):
        self.source = source
        self.worldMaterial = worldMaterial
        self.weightThreshold = np.float32(weightThreshold)
        self.logger = logger

        self.context = cl.create_some_context()
        self.mainQueue = cl.CommandQueue(self.context)
        self.device = self.context.devices[0]
        self.program = None

        self.HOST_photons, self.DEVICE_photons, self.photon_dtype, self.c_decl_photon = None, None, None, None
        self.HOST_photonAlive, self.DEVICE_photonAlive = None, None
        self.HOST_material, self.DEVICE_material, self.material_dtype, self.c_decl_mat = None, None, None, None
        self.HOST_logger, self.DEVICE_logger, self.logger_dtype, self.c_decl_logger = None, None, None, None
        self.HOST_randomSeed, self.DEVICE_randomSeed = None, None
        self.HOST_randomFloat, self.DEVICE_randomFloat = None, None

        self.makeTypes()
        self.makeBuffers()

    def fillPhotons(self):
        position = self.source.position
        direction = self.source.direction
        datasize = np.uint32(len(self.HOST_photons))
        if type(self.source).__name__ == "CLPencilSource":
            print(" === PENCIL SOURCE === ")
            self.program.fillPencilPhotonsBuffer(self.mainQueue, (datasize,), None,
                                                 self.DEVICE_photons,
                                                 self.DEVICE_randomSeed,
                                                 cl.cltypes.make_float4(position.x, position.y, position.z, 0),
                                                 cl.cltypes.make_float4(direction.x, direction.y, direction.z, 0))

        elif type(self.source).__name__ == "CLIsotropicSource":
            self.program.fillIsotropicPhotonsBuffer(self.mainQueue, (datasize,), None,
                                                    self.DEVICE_photons, self.DEVICE_randomSeed,
                                                    cl.cltypes.make_float4(position.x, position.y, position.z, 0))
        cl.enqueue_copy(self.mainQueue, self.HOST_photons, self.DEVICE_photons)
        print(self.HOST_photons)

    def buildProgram(self):
        t0 = time.time_ns()
        self.program = cl.Program(self.context,
                                  self.c_decl_photon + self.c_decl_mat + self.c_decl_logger + open(
                                      "./CLPhotons.c").read()).build()
        t1 = time.time_ns()
        print(" === COMPILE OPENCL PYTISSUEOPTICS === : " + str((t1 - t0) / 1e9) + " s")

    def propagate(self):
        self.buildProgram()
        self.fillPhotons()
        t0 = time.time_ns()

        print(" === PROPAGATE === : {}".format(len(self.HOST_photons)))
        datasize = np.uint32(len(self.HOST_photons))
        self.program.propagate(self.mainQueue, (datasize,), None, datasize, self.weightThreshold, self.DEVICE_photons,
                               self.DEVICE_material, self.DEVICE_logger, self.DEVICE_randomFloat, self.DEVICE_randomSeed)
        # cl.enqueue_copy(self.mainQueue, dest=self.HOST_photons, src=self.DEVICE_photons)
        self.mainQueue.finish()
        cl.enqueue_copy(self.mainQueue, dest=self.HOST_logger, src=self.DEVICE_logger)
        t1 = time.time_ns()
        print(" === GPU PHOTON SIMULATION === : " + str((t1 - t0) / 1e9) + " s")
        print(self.HOST_logger)

        for position, value in self.HOST_logger:
            self.logger.logDataPoint(value, Vector(position[0], position[1], position[2]))

    def makeTypes(self):
        def makePhotonType():
            photonStruct = np.dtype(
                [("position", cl.cltypes.float4),
                 ("direction", cl.cltypes.float4),
                 ("er", cl.cltypes.float4),
                 ("weight", cl.cltypes.float),
                 ("material_id", cl.cltypes.uint)])
            name = "photonStruct"
            photonStruct, c_decl_photon = cl.tools.match_dtype_to_c_struct(self.device, name, photonStruct)
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
            materialStruct, c_decl_mat = cl.tools.match_dtype_to_c_struct(self.device, name, materialStruct)
            material_dtype = cl.tools.get_or_register_dtype(name, materialStruct)
            return material_dtype, c_decl_mat

        def makeLoggerType():
            loggerStruct = np.dtype(
                [("position", cl.cltypes.float4),
                 ("delta_weight", cl.cltypes.float)])
            name = "loggerStruct"
            loggerStruct, c_decl_logger = cl.tools.match_dtype_to_c_struct(self.device, name, loggerStruct)
            logger_dtype = cl.tools.get_or_register_dtype(name, loggerStruct)
            return logger_dtype, c_decl_logger

        self.photon_dtype, self.c_decl_photon = makePhotonType()
        self.material_dtype, self.c_decl_mat = makeMaterialType()
        self.logger_dtype, self.c_decl_logger = makeLoggerType()

    def makeBuffers(self):
        self.makePhotonsBuffer()
        self.makeMaterialsBuffer()
        self.makeLoggerBuffer()
        self.makeRandomBuffer()

    def makePhotonsBuffer(self):
        self.HOST_photons = np.empty((self.source.N, ), dtype=self.photon_dtype)

        self.DEVICE_photons = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                        hostbuf=self.HOST_photons)

    def makeRandomBuffer(self):
        self.HOST_randomSeed = np.random.randint(low=0, high=2 ** 32 - 1, size=self.source.N, dtype=cl.cltypes.uint)
        self.DEVICE_randomSeed = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                           hostbuf=self.HOST_randomSeed)
        self.HOST_randomFloat = np.empty(self.source.N, dtype=cl.cltypes.float)
        self.DEVICE_randomFloat = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                            hostbuf=self.HOST_randomFloat)

    def makeMaterialsBuffer(self):
        materials = [self.worldMaterial]
        self.HOST_material = np.empty(len(materials), dtype=self.material_dtype)
        for i, mat in enumerate(materials):
            self.HOST_material[i]["mu_s"] = np.float32(mat.mu_s)
            self.HOST_material[i]["mu_a"] = np.float32(mat.mu_a)
            self.HOST_material[i]["mu_t"] = np.float32(mat.mu_t)
            self.HOST_material[i]["g"] = np.float32(mat.g)
            self.HOST_material[i]["n"] = np.float32(mat.index)
            self.HOST_material[i]["albedo"] = np.float32(mat.getAlbedo())
            self.HOST_material[i]["material_id"] = np.uint32(i)
        self.DEVICE_material = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                         hostbuf=self.HOST_material)

    def makeLoggerBuffer(self):
        loggerSize = int(-np.log(self.weightThreshold) / self.worldMaterial.getAlbedo())
        self.HOST_logger = np.empty(loggerSize, dtype=self.logger_dtype)
        self.DEVICE_logger = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                       hostbuf=self.HOST_logger)
