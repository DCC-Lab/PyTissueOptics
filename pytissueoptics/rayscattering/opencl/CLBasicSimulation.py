import copy
import time
from typing import List, Tuple

import pyopencl as cl
import pyopencl.tools
import numpy as np

from pytissueoptics.rayscattering.materials import ScatteringMaterial
from pytissueoptics.rayscattering.opencl.CLSource import CLSource, CLPencilSource, CLIsotropicSource
from pytissueoptics.rayscattering.opencl.CLStatistics import CLStatistics

np.random.seed(2)


class CLBasicSimulation:
    def __init__(self, source: CLSource, worldMaterial: ScatteringMaterial, stats: CLStatistics, stepSize: int = 100, batchSize=100):
        self.source = source
        self.worldMaterial = worldMaterial
        self.stats = stats
        self.stepSize = stepSize
        self.batchSize = batchSize

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
        self.HOST_bins, self.DEVICE_bins, self.binConfig_dtype, self.c_decl_binConfig = None, None, None, None
        self.log = None
        self.makeTypes()
        self.makeBuffers()

    def fillPhotons(self):
        p_x = self.source.position.x
        p_y = self.source.position.y
        p_z = self.source.position.z
        d_x = self.source.direction.x
        d_y = self.source.direction.y
        d_z = self.source.direction.z
        if isinstance(self.source, CLPencilSource):
            print("pencil")
            self.program.fillPencilPhotonsBuffer(self.mainQueue, (np.uint32(len(self.HOST_photons)),), None,
                                                 self.DEVICE_photons,
                                                 self.DEVICE_randomSeed, cl.cltypes.make_float4(p_x, p_y, p_z, 0),
                                                 cl.cltypes.make_float4(d_x, d_y, d_z, 0))

        elif isinstance(self.source, CLIsotropicSource):
            print("isotropic")
            self.program.fillIsotropicPhotonsBuffer(self.mainQueue, (np.uint32(len(self.HOST_photons)),), None,
                                                    self.DEVICE_photons, self.DEVICE_randomSeed,
                                                    cl.cltypes.make_float4(p_x, p_y, p_z, 0))
        cl.enqueue_copy(self.mainQueue, self.HOST_photons, self.DEVICE_photons)
        print(self.HOST_photons)

    def buildProgram(self):
        self.log = np.empty(self.HOST_logger.shape, dtype=self.logger_dtype)
        t0 = time.time_ns()
        self.program = cl.Program(self.context,
                                  self.c_decl_photon + self.c_decl_mat + self.c_decl_logger + open(
                                      "./CLPhotons.c").read()).build()
        t1 = time.time_ns()
        print(" === COMPILE === : " + str((t1 - t0) / 1e9) + " s")

    def launch(self):
        self.buildProgram()
        self.fillPhotons()
        t0 = time.time_ns()
        while np.any(self.HOST_photonAlive):
            print(" === STEP === : {}".format(len(self.HOST_photons)))
            for i in range(self.stepSize):
                self.program.oneStep(self.mainQueue, (np.uint32(len(self.HOST_photons)),), None,
                                np.uint32(len(self.HOST_photons)), np.uint32(i), self.DEVICE_photonAlive,
                                self.DEVICE_photons, self.DEVICE_material, self.DEVICE_logger, self.DEVICE_randomFloat,
                                self.DEVICE_randomSeed)
            cl.enqueue_copy(self.mainQueue, dest=self.HOST_photonAlive, src=self.DEVICE_photonAlive)
            cl.enqueue_copy(self.mainQueue, dest=self.HOST_photons, src=self.DEVICE_photons)
            self.HOST_photons = self.HOST_photons[self.HOST_photonAlive, ...]
            cl.enqueue_copy(self.mainQueue, dest=self.DEVICE_photons, src=self.HOST_photons)
            cl.enqueue_copy(self.mainQueue, dest=self.HOST_logger, src=self.DEVICE_logger)
            self.log = np.append(self.log, self.HOST_logger)
            # print("LOG", self.log)
            # self.makeLoggerBuffer()
        self.mainQueue.finish()
        t1 = time.time_ns()
        print(" === GPU PHOTON SIMULATION === : " + str((t1 - t0) / 1e9) + " s")

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

        def makeBinConfigType():
            binConfigStruct = np.dtype(
                [("minBounds", cl.cltypes.float3),
                 ("maxBounds", cl.cltypes.float3),
                 ("size", cl.cltypes.float3),
                 ("L", cl.cltypes.float3),
                 ("binSizes", cl.cltypes.float3)])
            name = "binConfigStruct"
            binConfigStruct, c_decl_binConfig = cl.tools.match_dtype_to_c_struct(self.device, name, binConfigStruct)
            binConfig_dtype = cl.tools.get_or_register_dtype(name, binConfigStruct)
            return binConfig_dtype, c_decl_binConfig

        self.photon_dtype, self.c_decl_photon = makePhotonType()
        self.material_dtype, self.c_decl_mat = makeMaterialType()
        self.logger_dtype, self.c_decl_logger = makeLoggerType()
        self.binConfig_dtype, self.c_decl_binConfig = makeBinConfigType()

    def makeBuffers(self):
        self.makePhotonsBuffer()
        self.makeMaterialsBuffer()
        self.makeLoggerBuffer()
        self.makeRandomBuffer()

    def makePhotonsBuffer(self):
        self.HOST_photons = np.empty(self.source.N, dtype=self.photon_dtype)
        self.HOST_photonAlive = np.ones(self.source.N, dtype=np.uint8)
        # for i, p in enumerate(photons):
        #     self.HOST_photons[i]["position"] = cl.cltypes.make_float4(p.position.x, p.position.y, p.position.z, 0)
        #     self.HOST_photons[i]["direction"] = cl.cltypes.make_float4(p.direction.x, p.direction.y, p.direction.z, 0)
        #     self.HOST_photons[i]["er"] = cl.cltypes.make_float4(p.er.x, p.er.y, p.er.z, 0)
        #     self.HOST_photons[i]["weight"] = p.weight
        #     self.HOST_photons[i]["material_id"] = 0

        self.DEVICE_photons = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                        hostbuf=self.HOST_photons)
        self.DEVICE_photonAlive = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                            hostbuf=self.HOST_photonAlive)

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
        self.HOST_logger = np.empty(self.source.N * self.stepSize, dtype=self.logger_dtype)
        self.DEVICE_logger = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                       hostbuf=self.HOST_logger)

    def makeBinsBuffer(self):
        u = self.stats.size[0]
        v = self.stats.size[1]
        w = self.stats.size[2]
        self.HOST_bins = np.zeros((u, v, w), dtype=cl.cltypes.float)
        self.DEVICE_bins = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.HOST_bins)
