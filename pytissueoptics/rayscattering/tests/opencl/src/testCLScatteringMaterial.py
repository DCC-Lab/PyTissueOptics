import math
import os
import unittest

import numpy as np

from pytissueoptics.rayscattering.opencl import OPENCL_OK
from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_SOURCE_DIR
from pytissueoptics.rayscattering.opencl.buffers import BufferOf, EmptyBuffer, RandomBuffer
from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram


@unittest.skipIf(not OPENCL_OK, 'OpenCL device not available.')
class TestCLScatteringMaterial(unittest.TestCase):
    def setUp(self):
        sourcePath = os.path.join(OPENCL_SOURCE_DIR, "scatteringMaterial.c")
        self.program = CLProgram(sourcePath)
        self.program.include(self._getMissingDeclarations())

    def testWhenGetScatteringDistance_shouldReturnCorrectScatteringDistance(self):
        nWorkUnits = 1
        randomNumber = 0.5
        randomNumberBuffer = BufferOf(np.array([randomNumber], dtype=np.float32))
        distanceBuffer = EmptyBuffer(nWorkUnits)
        mu_t = np.float32(4)

        self.program.launchKernel("getScatteringDistanceKernel", N=nWorkUnits,
                                  arguments=[distanceBuffer, randomNumberBuffer, mu_t])

        distance = self.program.getData(distanceBuffer)[0]
        self.assertAlmostEqual(-math.log(randomNumber) / mu_t, distance, places=5)

    def testWhenGetScatteringDistanceInVacuum_shouldReturnInfinity(self):
        nWorkUnits = 1
        randomNumber = 0.5
        randomNumberBuffer = BufferOf(np.array([randomNumber], dtype=np.float32))
        distanceBuffer = EmptyBuffer(nWorkUnits)
        mu_t = np.float32(0)

        self.program.launchKernel("getScatteringDistanceKernel", N=nWorkUnits,
                                  arguments=[distanceBuffer, randomNumberBuffer, mu_t])

        distance = self.program.getData(distanceBuffer)[0]
        self.assertEqual(math.inf, distance)

    def testWhenGetScatteringAnglePhi_shouldReturnAngleBetween0And2Pi(self):
        randomNumbers = [0, 0.5, 1]
        nWorkUnits = len(randomNumbers)
        randomNumberBuffer = BufferOf(np.array(randomNumbers, dtype=np.float32))
        anglePhiBuffer = EmptyBuffer(nWorkUnits)

        self.program.launchKernel("getScatteringAnglePhiKernel", N=nWorkUnits,
                                  arguments=[anglePhiBuffer, randomNumberBuffer])

        anglesPhi = self.program.getData(anglePhiBuffer)
        expectedAngles = [0, math.pi, 2 * math.pi]
        self.assertTrue(np.isclose(expectedAngles, anglesPhi).all())

    def testWhenGetScatteringAngleTheta_shouldReturnAngleBetween0AndPi(self):
        randomNumbers = [0, 0.5, 1]
        nWorkUnits = len(randomNumbers)
        randomNumberBuffer = BufferOf(np.array(randomNumbers, dtype=np.float32))
        angleThetaBuffer = EmptyBuffer(nWorkUnits)
        g = np.float32(0)
        self.program.launchKernel("getScatteringAngleThetaKernel", N=nWorkUnits,
                                  arguments=[angleThetaBuffer, randomNumberBuffer, g])

        anglesTheta = self.program.getData(angleThetaBuffer)
        expectedAngles = [math.pi, math.pi / 2, 0]
        self.assertTrue(np.isclose(expectedAngles, anglesTheta).all())

    def testGivenFullAnisotropyFactor_shouldReturnZeroThetaAngle(self):
        nWorkUnits = 10
        randomNumberBuffer = RandomBuffer(nWorkUnits)
        angleThetaBuffer = EmptyBuffer(nWorkUnits)
        g = np.float32(1)
        self.program.launchKernel("getScatteringAngleThetaKernel", N=nWorkUnits,
                                  arguments=[angleThetaBuffer, randomNumberBuffer, g])

        anglesTheta = self.program.getData(angleThetaBuffer)
        expectedAngles = np.zeros(nWorkUnits)
        self.assertTrue(np.isclose(expectedAngles, anglesTheta).all())

    @staticmethod
    def _getMissingDeclarations() -> str:
        return """
        struct Material {float g;};
        typedef struct Material Material;
        struct Photon {uint materialID;};
        typedef struct Photon Photon;
        """
