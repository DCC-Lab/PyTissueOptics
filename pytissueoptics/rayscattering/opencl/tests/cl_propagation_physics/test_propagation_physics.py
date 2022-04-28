import os
import pyopencl as cl
import pyopencl.tools
import numpy as np
import unittest
from numpy.lib import recfunctions as rfn
from pytissueoptics.scene import Vector


class TestPropagationPhysics(unittest.TestCase):
    def setUp(self):
        kernelPath = os.path.dirname(os.path.abspath(__file__)) + "{}propagation_physics.c".format(os.sep)
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.device = self.ctx.devices[0]
        self.mf = cl.mem_flags
        c_decl = self._makeTypes()
        self.program = cl.Program(self.ctx, c_decl + open(kernelPath, 'r').read()).build()

    def _makeTypes(self):
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
                [("delta_weight", cl.cltypes.float),
                 ("x", cl.cltypes.float),
                 ("y", cl.cltypes.float),
                 ("z", cl.cltypes.float)])
            name = "loggerStruct"
            loggerStruct, c_decl_logger = cl.tools.match_dtype_to_c_struct(self.device, name, loggerStruct)
            logger_dtype = cl.tools.get_or_register_dtype(name, loggerStruct)
            return logger_dtype, c_decl_logger

        photon_dtype, c_decl_photon = makePhotonType()
        material_dtype, c_decl_mat = makeMaterialType()
        logger_dtype, c_decl_logger = makeLoggerType()
        return c_decl_photon + c_decl_mat + c_decl_logger

    def makeRandomVectorsAndBuffers(self, N):
        rng = np.random.default_rng()
        randomErValues = rng.random((N, 3), dtype=np.float32)
        float4AddOn = np.zeros((N, 1), dtype=np.float32)
        CPU_VectorEr = [Vector(*randomErValues[i, :]) for i in range(N)]
        randomErValues = np.append(randomErValues, float4AddOn, axis=1)
        randomErValuesFloat4 = [cl.cltypes.make_float4(*randomErValues[i, :]) for i in range(N)]
        HOST_ErVectors = np.array(randomErValuesFloat4, dtype=cl.cltypes.float4)
        DEVICE_ErVectors = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=HOST_ErVectors)
        return CPU_VectorEr, HOST_ErVectors, DEVICE_ErVectors

    def makeRandomScalarsAndBuffers(self, N):
        rng = np.random.default_rng()
        randomScalarValues = rng.random((N, 1), dtype=np.float32)
        CPU_scalarValues = [randomScalarValues[i, 0] for i in range(N)]
        HOST_ScalarValues = np.array(randomScalarValues, dtype=np.float32)
        DEVICE_ScalarValues = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=HOST_ScalarValues)
        return CPU_scalarValues, HOST_ScalarValues, DEVICE_ScalarValues

    def getScatteringTheta(self, rndValue, g):
        if g == 0:
            cost = 2 * rndValue - 1
        else:
            temp = (1 - g * g) / (1 - g + 2 * g * rndValue)
            cost = (1 + g * g - temp * temp) / (2 * g)
        return np.arccos(cost)

    def test_whenGetScatteringAngleTheta_GPU_and_CPU_shouldReturnSameValues(self):
        N = 500
        g = 0.8
        CPU_rndValues, HOST_rndValues, DEVICE_rndValues = self.makeRandomScalarsAndBuffers(N)
        HOST_angleResults = np.zeros(N, dtype=np.float32)
        DEVICE_angleResults = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=HOST_angleResults)

        CPU_angleResults = np.array([self.getScatteringTheta(rndValue, g) for rndValue in CPU_rndValues])

        self.program.getScatteringAngleThetaKernel(self.queue, HOST_rndValues.shape, None, DEVICE_angleResults, DEVICE_rndValues,  np.float32(g))
        cl.enqueue_copy(self.queue, HOST_angleResults, DEVICE_angleResults)

        GPU_angleResults = HOST_angleResults

        self.assertTrue(np.all(np.isclose(CPU_angleResults, GPU_angleResults, atol=1e-3)))

    def test_whenGetScatteringAnglePhi_GPU_and_CPU_shouldReturnSameValues(self):
        pass
