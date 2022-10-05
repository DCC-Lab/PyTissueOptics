import os
import time
from typing import List

try:
    import pyopencl as cl
    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False
import numpy as np
from numpy.lib import recfunctions as rfn

from pytissueoptics.rayscattering.opencl.types import makePhotonType, makeMaterialType, makeLoggerType, CLType, \
    PhotonCLType
from pytissueoptics.rayscattering.tissues import InfiniteTissue
from pytissueoptics.rayscattering.tissues.rayScatteringScene import RayScatteringScene
from pytissueoptics.scene import Logger
from pytissueoptics.scene.logger import InteractionKey

PROPAGATION_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')


class CLProgram:
    def __init__(self, sourceDir: str):
        self._sourceDir = sourceDir
        self._context = cl.create_some_context()
        self._mainQueue = cl.CommandQueue(self._context)
        self._device = self._context.devices[0]
        self._program = None

    def build(self, CLObjects: List[CLType], _tmpDecl):
        sourceFiles = ['random.c', 'vectorOperators.c', 'propagation.c']
        # fixme: source file ordering matters
        # sourceFiles = [f for f in list(os.walk(self._sourceDir))[0][2] if f.endswith('.c')]
        sourceCode = ''
        for sourceFile in sourceFiles:
            sourceCode += open(os.path.join(self._sourceDir, sourceFile)).read()

        typeDeclarations = ''
        for CLObject in CLObjects:
            CLObject.build(self._device, self._context)
            typeDeclarations += CLObject.declaration
        # todo: replace with ''.join(List[str])

        # temporary for step refactor
        for decl in _tmpDecl:
            typeDeclarations += decl

        self._program = cl.Program(self._context, typeDeclarations + sourceCode).build()


class CLPhotons:
    def __init__(self, positions: np.ndarray, directions: np.ndarray, N: int, weightThreshold: float = 0.0001):
        self._positions = positions
        self._directions = directions
        self._N = N
        self._weightThreshold = np.float32(weightThreshold)
        self._logger = None

        self._program = CLProgram(sourceDir=PROPAGATION_SOURCE_DIR)

        self._photons = PhotonCLType(positions, directions)

        # self._material = None
        # self._logger = None
        # self._randomSeed = None
        # self._randomFloat = None

        # self._HOST_photons, self._DEVICE_photons, self._photon_dtype, self._c_decl_photon = None, None, None, None
        self._HOST_material, self._DEVICE_material, self._material_dtype, self._c_decl_mat = None, None, None, None
        self._HOST_logger, self._DEVICE_logger, self._logger_dtype, self._c_decl_logger = None, None, None, None
        self._HOST_randomSeed, self._DEVICE_randomSeed, self._HOST_randomFloat, self._DEVICE_randomFloat = None, None, None, None

        self._makeTypes()
        # self._program.build(CLObjects=[self._photons], _tmpDecl=[self._c_decl_mat, self._c_decl_logger])

    def _makeTypes(self):
        # self._photon_dtype, self._c_decl_photon = makePhotonType(self._device)
        self._material_dtype, self._c_decl_mat = makeMaterialType(self._program._device)
        self._logger_dtype, self._c_decl_logger = makeLoggerType(self._program._device)

    def _extractFromScene(self, scene: RayScatteringScene):
        if type(scene) is not InfiniteTissue:
            raise TypeError("OpenCL propagation is only supported for InfiniteTissue for the moment.")
        self._worldMaterial = scene.getWorldEnvironment().material

    def prepareAndPropagate(self, scene: RayScatteringScene, logger: Logger):
        self._logger = logger
        self._extractFromScene(scene)
        self._makeBuffers()

        # self._buildProgram()
        self._program.build(CLObjects=[self._photons], _tmpDecl=[self._c_decl_mat, self._c_decl_logger])

        self._propagate()

    def _propagate(self):
        t0 = time.time_ns()
        datasize = np.uint32(len(self._photons._HOST_buffer))  # todo: isnt that supposed to be self.N ?

        # todo: whats up with this signature ? first 3 arguments not in C decl
        self._program._program.propagate(self._program._mainQueue, self._photons._HOST_buffer.shape, None, datasize, self._weightThreshold,
                                self._photons._DEVICE_buffer,
                                self._DEVICE_material, self._DEVICE_logger, self._DEVICE_randomFloat,
                                self._DEVICE_randomSeed)
        self._program._mainQueue.finish()
        cl.enqueue_copy(self._program._mainQueue, dest=self._HOST_logger, src=self._DEVICE_logger)
        t1 = time.time_ns()
        print("CLPhotons.propagate: {} s".format((t1 - t0) / 1e9))

        log = rfn.structured_to_unstructured(self._HOST_logger)
        self._logger.logDataPointArray(log, InteractionKey("universe", None))

    def _makeBuffers(self):
        # self._makePhotonsBuffer()
        self._makeMaterialsBuffer()
        self._makeLoggerBuffer()
        self._makeRandomBuffer()

    # def _makePhotonsBuffer(self):
    #     # CLTypes = {'photons': PhotonCLType, 'materials': MaterialCLType, ... }
    #     # program.addType(key, class)
    #     # program.build
    #     # .... boff
    #     # cl.Photons(positions, directions), cl.RandomSeed(size), cl.RandomFloat(size), cl.Material(worldMaterial)
    #     # program.add(photons)
    #     # program.build -> photons.build(device), program.makeBuffers() -> photons.data
    #     # todo: could replace with HOST_photons = PhotonCLType.create(positions, directions)
    #     photonsPrototype = np.zeros(self._N, dtype=self._photon_dtype)
    #     photonsPrototype = rfn.structured_to_unstructured(photonsPrototype)
    #     photonsPrototype[:, 0:3] = self._positions[:, ::]
    #     photonsPrototype[:, 4:7] = self._directions[:, ::]
    #     photonsPrototype[:, 12] = 1.0
    #     photonsPrototype[:, 13] = 0
    #     self._HOST_photons = rfn.unstructured_to_structured(photonsPrototype, self._photon_dtype)
    #     self._DEVICE_photons = cl.Buffer(self._context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
    #                                      hostbuf=self._HOST_photons)

    def _makeRandomBuffer(self):
        self._HOST_randomSeed = np.random.randint(low=0, high=2 ** 32 - 1, size=self._N,
                                                  dtype=cl.cltypes.uint)
        self._DEVICE_randomSeed = cl.Buffer(self._program._context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                            hostbuf=self._HOST_randomSeed)
        self._HOST_randomFloat = np.empty(self._N, dtype=cl.cltypes.float)
        self._DEVICE_randomFloat = cl.Buffer(self._program._context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
                                             hostbuf=self._HOST_randomFloat)

    def _makeMaterialsBuffer(self):
        self._HOST_material = np.empty(1, dtype=self._material_dtype)
        self._HOST_material["mu_s"] = np.float32(self._worldMaterial.mu_s)
        self._HOST_material["mu_a"] = np.float32(self._worldMaterial.mu_a)
        self._HOST_material["mu_t"] = np.float32(self._worldMaterial.mu_t)
        self._HOST_material["g"] = np.float32(self._worldMaterial.g)
        self._HOST_material["n"] = np.float32(self._worldMaterial.n)
        self._HOST_material["albedo"] = np.float32(self._worldMaterial.getAlbedo())
        self._HOST_material["material_id"] = np.uint32(0)
        self._DEVICE_material = cl.Buffer(self._program._context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                          hostbuf=self._HOST_material)

    def _makeLoggerBuffer(self):
        loggerSize = int(
            -np.log(self._weightThreshold) / self._worldMaterial.getAlbedo()) * self._N
        self._HOST_logger = np.empty(loggerSize, dtype=self._logger_dtype)
        self._DEVICE_logger = cl.Buffer(self._program._context, cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                        hostbuf=self._HOST_logger)
