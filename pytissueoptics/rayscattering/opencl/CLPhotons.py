from typing import List

import pyopencl as cl
import pyopencl.tools
import numpy as np

from pytissueoptics.rayscattering import Photon
from pytissueoptics.rayscattering.materials import ScatteringMaterial


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
        program = cl.Program(self.context,
                             self.c_decl_photon + self.c_decl_mat + self.c_decl_logger + open("./CLPhotons.c")).build()

        randomIndex = 0
        loggerIndex = 0
        program.propagate(self.mainQueue, self.HOST_photons.shape, None, self.DEVICE_photons, self.DEVICE_material,
                          self.DEVICE_rand, self.DEVICE_logger, randomIndex, loggerIndex)
        self.mainQueue.finish()
        cl.enqueue_copy(self.mainQueue, dest=self.HOST_logger, src=self.DEVICE_logger)

    def makeTypes(self):
        def makePhotonType(self):
            photonStruct = np.dtype(
                [("position", cl.cltypes.float4),
                 ("direction", cl.cltypes.float4),
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
            name = "logger"
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
        self.makeRandomBuffer(len(self.photons)*100)

    def makePhotonsBuffer(self):
        photons = self.photons
        self.HOST_photons = np.empty(len(photons), dtype=self.photon_dtype)
        for i, p in photons:
            self.HOST_photons[i]["position"] = cl.cltypes.make_float4(p.position.x, p.position.y, p.position.z, 0)
            self.HOST_photons[i]["direction"] = cl.cltypes.make_float4(p.direciton.x, p.direciton.y, p.direciton.z, 0)
            self.HOST_photons[i]["weight"] = p.weight
            self.HOST_photons[i]["material_id"] = 0

        self.DEVICE_photons = cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                        hostbuf=self.HOST_photons)

    def makeRandomBuffer(self, N):
        rng = np.random.default_rng()
        self.HOST_rand = rng.random(size=N, dtype=cl.cltypes.float)
        self.DEVICE_rand = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=self.HOST_rand)

    def makeMaterialsBuffer(self):
        materials = [self.worldMaterial]
        self.HOST_material = np.empty(len(materials), dtype=self.material_dtype)
        for i, mat in enumerate(materials):
            self.HOST_material[i]["mu_s"] = mat.mu_s
            self.HOST_material[i]["mu_a"] = mat.mu_a
            self.HOST_material[i]["mu_t"] = mat.mu_t
            self.HOST_material[i]["g"] = mat.g
            self.HOST_material[i]["n"] = mat.index
            self.HOST_material[i]["albedo"] = mat.getAlbedo()
            self.HOST_material[i]["material_id"] = i
        self.DEVICE_material = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                         hostbuf=self.HOST_material)

    def makeLoggerBuffer(self):
        photons = self.photons
        # the number of absorption events per photon is approx ln(0.0001)*mu_t/mu_a
        loggerLength = int((-np.log(0.0001) * photons[0].material.mu_t / photons[0].material.mu_a) * len(photons))
        self.HOST_logger = np.empty(loggerLength, dtype=self.logger_dtype)
        self.DEVICE_logger = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                       hostbuf=self.HOST_logger)
