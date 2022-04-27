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

    def test_whenRotateAroundVector_GPU_and_CPU_shouldReturnSameValues(self):
        N = 3

        # Make Random Values
        rng = np.random.default_rng()
        randomErValues = rng.random((N, 3), dtype=np.float32)
        randomAxisValues = rng.random((N, 3), dtype=np.float32)
        randomPhiValues = rng.random((N, 1), dtype=np.float32)

        # Create the Vectors to be used on the CPU Functions
        CPU_VectorEr = [Vector(*randomErValues[i, :]) for i in range(N)]
        CPU_VectorAxis = [Vector(*randomAxisValues[i, :]) for i in range(N)]
        CPU_ScalarPhi = [randomPhiValues[i, 0] for i in range(N)]

        # Create the Vectors to be used on the GPU Functions
        float4AddOn = np.zeros((N, 1))

        randomAxisValues = np.append(randomAxisValues, float4AddOn, axis=1)
        randomAxisValuesFloat4 = [cl.cltypes.make_float4(*randomAxisValues[i, :]) for i in range(N)]

        randomErValues = np.append(randomErValues, float4AddOn, axis=1)
        randomErValuesFloat4 = [cl.cltypes.make_float4(*randomErValues[i, :]) for i in range(N)]

        # Create the GPU Buffers
        HOST_AxisVectors = np.array(randomAxisValuesFloat4, dtype=cl.cltypes.float4)
        DEVICE_AxisVectors = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=HOST_AxisVectors)
        HOST_ErVectors = np.array(randomErValuesFloat4, dtype=cl.cltypes.float4)
        DEVICE_ErVectors = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=HOST_ErVectors)
        HOST_Phi = np.array(randomPhiValues, dtype=cl.cltypes.float)
        DEVICE_Phi = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=HOST_Phi)

        #print(HOST_ErVectors)

        # Execute the CPU Functions
        for i, vector in enumerate(CPU_VectorEr):
            vector.rotateAround(CPU_VectorAxis[i], CPU_ScalarPhi[i])
        CPU_VectorErResults = np.array([[vector.x, vector.y, vector.z] for vector in CPU_VectorEr])
        print(CPU_VectorErResults)

        # Execute the GPU Functions
        self.program.rotateAroundAxisKernel(self.queue, HOST_AxisVectors.shape, None, DEVICE_ErVectors, DEVICE_AxisVectors, DEVICE_Phi)
        cl.enqueue_copy(self.queue, HOST_ErVectors, DEVICE_ErVectors)

        #print(HOST_ErVectors)
        GPU_VectorErResults = rfn.structured_to_unstructured(HOST_ErVectors)
        GPU_VectorErResults = np.delete(GPU_VectorErResults, -1, axis=1)
        print(GPU_VectorErResults)



#
# HOST_random_er = np.random.random((N, 4), dtype=cl.cltypes.float4)
# DEVICE_random_er = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
#                                     hostbuf=HOST_random_er)
# # HOST_randomFloat = np.empty(N, dtype=cl.cltypes.float)
# # DEVICE_randomFloat = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR,
# #                                      hostbuf=HOST_randomFloat)
#
# with open(kernelPath, "r") as kernelFile:
#     kernelSource = kernelFile.read()
#     program = cl.Program(context, kernelSource).build()

# program.fillRandomFloatBuffer(mainQueue, (N,), None, DEVICE_randomSeed, DEVICE_randomFloat)
# cl.enqueue_copy(mainQueue, HOST_randomFloat, DEVICE_randomFloat)
# randomVectors
