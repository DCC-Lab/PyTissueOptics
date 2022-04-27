import os
import pyopencl as cl
import numpy as np
import unittest
from numpy.lib import recfunctions as rfn
from pytissueoptics.scene import Vector


class TestVectorOperations(unittest.TestCase):
    def setUp(self):
        kernelPath = os.path.dirname(os.path.abspath(__file__)) + "{}vector_operators.c".format(os.sep)
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.device = self.ctx.devices[0]
        self.mf = cl.mem_flags
        self.program = cl.Program(self.ctx, open(kernelPath, 'r').read()).build()

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

    def test_whenRotateAroundVector_GPU_and_CPU_shouldReturnSameValues(self):
        N = 500

        CPU_ErVectors, HOST_ErVectors, DEVICE_ErVectors = self.makeRandomVectorsAndBuffers(N)
        CPU_AxisVectors, HOST_AxisVectors, DEVICE_AxisVectors = self.makeRandomVectorsAndBuffers(N)
        CPU_PhiValues, HOST_PhiValues, DEVICE_PhiValues = self.makeRandomScalarsAndBuffers(N)

        for vector in CPU_AxisVectors:
            vector.normalize()
        for i, vector in enumerate(CPU_ErVectors):
            vector.rotateAround(CPU_AxisVectors[i], CPU_PhiValues[i])
        CPU_VectorErResults = np.array([[vector.x, vector.y, vector.z] for vector in CPU_ErVectors])

        self.program.rotateAroundAxisKernel(self.queue, HOST_AxisVectors.shape, None, DEVICE_ErVectors, DEVICE_AxisVectors, DEVICE_PhiValues)
        cl.enqueue_copy(self.queue, HOST_ErVectors, DEVICE_ErVectors)

        GPU_VectorErResults = rfn.structured_to_unstructured(HOST_ErVectors)
        GPU_VectorErResults = np.delete(GPU_VectorErResults, -1, axis=1)

        self.assertTrue(np.allclose(GPU_VectorErResults, CPU_VectorErResults, atol=1e-3))

    def test_whenNormalizeVector_GPU_and_CPU_shouldReturnSameValue(self):
        N = 300
        CPU_VectorEr, HOST_ErVectors, DEVICE_ErVectors = self.makeRandomVectorsAndBuffers(N)

        self.program.normalizeVectorKernel(self.queue, HOST_ErVectors.shape, None, DEVICE_ErVectors)
        cl.enqueue_copy(self.queue, HOST_ErVectors, DEVICE_ErVectors)

        GPU_VectorErResults = rfn.structured_to_unstructured(HOST_ErVectors)
        GPU_VectorErResults = np.delete(GPU_VectorErResults, -1, axis=1)
        for i, vector in enumerate(CPU_VectorEr):
            vector.normalize()
        CPU_VectorErResults = np.array([[vector.x, vector.y, vector.z] for vector in CPU_VectorEr])

        self.assertTrue(np.all(np.isclose(GPU_VectorErResults, CPU_VectorErResults)))
