import envtest
import os
import unittest
from typing import List

import numpy as np
from numpy.lib import recfunctions as rfn

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE
from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_SOURCE_DIR
from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.buffers import BufferOf

if OPENCL_AVAILABLE:
    import pyopencl as cl
else:
    cl = None


@unittest.skipIf(not OPENCL_AVAILABLE, 'Requires PyOpenCL.')
class TestCLVectorOperators(unittest.TestCase):
    def setUp(self):
        sourcePath = os.path.join(OPENCL_SOURCE_DIR, "vectorOperators.c")
        self.program = CLProgram(sourcePath)

    def testWhenNormalizeVector_shouldNormalize(self):
        inputVectors = [Vector(1, 0, 0), Vector(2, 0, 0), Vector(1, 1, 0), Vector(1, 1, 1)]
        N = len(inputVectors)
        vectorBuffer = self._vectorBuffer(inputVectors)

        self.program.launchKernel("normalizeVectorGlobalKernel", N=N, arguments=[vectorBuffer])

        normalizedVectors = self._vectorsFromBuffer(vectorBuffer)
        expectedVectors = [vector.copy() for vector in inputVectors]
        for vector in expectedVectors:
            vector.normalize()
        self.assertListEqual(expectedVectors, normalizedVectors)

    def testWhenRotateAroundAxisWithAngle_shouldRotateVectorAroundThisAxisByGivenAngle(self):
        vector = Vector(1, 1, 0)
        axis = Vector(0, 1, 0)
        angle = np.pi * 3 / 2
        N = 1
        vectorBuffer = self._vectorBuffer([vector])
        axisBuffer = self._vectorBuffer([axis])
        angleBuffer = BufferOf(np.array([angle], dtype=np.float32))

        self.program.launchKernel("rotateAroundAxisGlobalKernel", N=N, arguments=[vectorBuffer, axisBuffer, angleBuffer])

        rotatedVector = self._vectorsFromBuffer(vectorBuffer)[0]
        expectedVector = Vector(0, 1, 1)
        self.assertEqual(expectedVector, rotatedVector)

    def testWhenGetAnyOrthogonal_shouldReturnANewOrthogonalVector(self):
        n = 100
        direction_xs = np.random.uniform(-1, 1, n)
        direction_ys = np.random.uniform(-1, 1, n)
        direction_zs = np.random.uniform(-1, 1, n)
        vectors = []
        expectedOrthogonalVectors = []
        for i in range(n):
            vector = Vector(direction_xs[i], direction_ys[i], direction_zs[i])
            vectors.append(vector)
            expectedOrthogonalVectors.append(vector.getAnyOrthogonal())

        edgeCaseVectors = [Vector(-1, 0, 0), Vector(1, 0, 0),
                           Vector(0, 1, 0), Vector(0, -1, 0),
                           Vector(0, 0, 1), Vector(0, 0, -1)]
        for vector in edgeCaseVectors:
            vectors.append(vector)
            expectedOrthogonalVectors.append(vector.getAnyOrthogonal())

        N = len(vectors)
        vectorBuffer = self._vectorBuffer(vectors)
        orthogonalBuffer = BufferOf(np.empty(N, dtype=cl.cltypes.float3))

        self.program.launchKernel("getAnyOrthogonalGlobalKernel", N=N, arguments=[vectorBuffer, orthogonalBuffer])

        orthogonalVectors = self._vectorsFromBuffer(orthogonalBuffer)
        self.assertListEqual(expectedOrthogonalVectors, orthogonalVectors)

    @staticmethod
    def _vectorBuffer(vectors: List[Vector]):
        vectorsFloat3 = [cl.cltypes.make_float3(*vector.array) for vector in vectors]
        return BufferOf(np.array(vectorsFloat3))

    def _vectorsFromBuffer(self, vectorBuffer):
        vectors = self.program.getData(vectorBuffer)
        vectors = rfn.structured_to_unstructured(vectors, dtype=np.float32)
        return [Vector(*vector[:3]) for vector in vectors]

if __name__ == "__main__":
    unittest.main()
