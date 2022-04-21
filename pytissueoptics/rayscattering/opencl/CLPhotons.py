import time
from typing import List

import pyopencl as cl
import pyopencl.tools
import numpy as np

from pytissueoptics.rayscattering import Photon
from pytissueoptics.rayscattering.materials import ScatteringMaterial
np.random.seed(0)


class CLPhotons:
    def __init__(self, photons: List[Photon], worldMaterial=ScatteringMaterial()):
        self.worldMaterial = worldMaterial
        self.photons = photons

        self.context = cl.create_some_context()
        self.mainQueue = cl.CommandQueue(self.context)
        self.device = self.context.devices[0]

        self.HOST_photons, self.DEVICE_photons, self.photon_dtype, self.c_decl_photon = None, None, None, None
        self.HOST_material, self.DEVICE_material, self.material_dtype, self.c_decl_mat = None, None, None, None
        self.HOST_logger, self.DEVICE_logger, self.logger_dtype, self.c_decl_logger = None, None, None, None
        self.HOST_rand, self.DEVICE_rand = None, None

        self.makeTypes()
        self.makeBuffers()

    @property
    def logger(self):
        return self.HOST_logger

    def propagate(self):
        print(" === PROPAGATE === ")
        t00 = time.time_ns()
        program = cl.Program(self.context,
                             self.c_decl_photon + self.c_decl_mat + self.c_decl_logger + open("./CLPhotons.c").read()).build()
        t0 = time.time_ns()
        program.propagate(self.mainQueue, self.HOST_photons.shape, None, np.uint32(len(self.photons)), self.DEVICE_photons, self.DEVICE_material,
                          self.DEVICE_logger, self.DEVICE_randomFloat, self.DEVICE_randomInt)
        self.mainQueue.finish()
        t1 = time.time_ns()
        cl.enqueue_copy(self.mainQueue, dest=self.HOST_photons, src=self.DEVICE_photons)
        cl.enqueue_copy(self.mainQueue, dest=self.HOST_logger, src=self.DEVICE_logger)
        t2 = time.time_ns()
        print("Compile Time:", (t0 - t00) / 1e9)
        print("Execution Time:", (t1 - t0) / 1e9)
        print("Copy Time:", (t2 - t1) / 1e9)
        print("Total Time:", (t2 - t00) / 1e9)

    def makeTypes(self):
        def makePhotonType(self):
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

        def makeMaterialType(self):
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

        def makeLoggerType(self):
            loggerStruct = np.dtype(
                [("position", cl.cltypes.float4),
                 ("delta_weight", cl.cltypes.float)])
            name = "loggerStruct"
            loggerStruct, c_decl_logger = cl.tools.match_dtype_to_c_struct(self.device, name, loggerStruct)
            logger_dtype = cl.tools.get_or_register_dtype(name, loggerStruct)
            return logger_dtype, c_decl_logger

        self.photon_dtype, self.c_decl_photon = makePhotonType(self)
        self.material_dtype, self.c_decl_mat = makeMaterialType(self)
        self.logger_dtype, self.c_decl_logger = makeLoggerType(self)

    def makeBuffers(self):
        self.makePhotonsBuffer()
        self.makeMaterialsBuffer()
        self.makeLoggerBuffer()
        self.makeRandomBuffer()

    def makePhotonsBuffer(self):
        photons = self.photons
        self.HOST_photons = np.empty(len(photons), dtype=self.photon_dtype)
        for i, p in enumerate(photons):
            self.HOST_photons[i]["position"] = cl.cltypes.make_float4(p.position.x, p.position.y, p.position.z, 0)
            self.HOST_photons[i]["direction"] = cl.cltypes.make_float4(p.direction.x, p.direction.y, p.direction.z, 0)
            self.HOST_photons[i]["er"] = cl.cltypes.make_float4(p.er.x, p.er.y, p.er.z, 0)
            self.HOST_photons[i]["weight"] = p.weight
            self.HOST_photons[i]["material_id"] = 0

        self.DEVICE_photons = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                        hostbuf=self.HOST_photons)

    def makeRandomBuffer(self):
        self.HOST_randomInt = np.random.randint(low=0, high=2**32-1, size=len(self.photons), dtype=cl.cltypes.uint)
        self.DEVICE_randomInt = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=self.HOST_randomInt)
        self.HOST_randomFloat = np.empty(len(self.photons), dtype=cl.cltypes.float)
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
        photons = self.photons
        # the number of absorption events per photon is approx ln(0.0001)*mu_t/mu_a
        loggerLength = int((-np.log(0.0001) * photons[0].material.mu_t / photons[0].material.mu_a) * len(photons))
        print(loggerLength)
        self.HOST_logger = np.empty(loggerLength, dtype=self.logger_dtype)
        self.DEVICE_logger = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                       hostbuf=self.HOST_logger)
