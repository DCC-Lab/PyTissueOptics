import os
import traceback
import unittest

import numpy as np

from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE
from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_SOURCE_DIR
from pytissueoptics.rayscattering.opencl.buffers import SeedCL, EmptyBuffer
from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram


@unittest.skipIf(not OPENCL_AVAILABLE, 'Requires PyOpenCL.')
class TestCLRandom(unittest.TestCase):
    def setUp(self):
        sourcePath = os.path.join(OPENCL_SOURCE_DIR, "random.c")
        self.program = CLProgram(sourcePath)

    def testWhenGetRandomValues_shouldBeRandom(self):
        nWorkUnits = 10
        np.random.seed(0)
        seeds = SeedCL(nWorkUnits)
        valueBuffer = EmptyBuffer(nWorkUnits)

        self.program.launchKernel("fillRandomFloatBuffer", N=nWorkUnits, arguments=[seeds, valueBuffer])

        randomValues = self.program.getData(valueBuffer)
        self.assertTrue(np.all(randomValues >= 0))
        self.assertTrue(np.all(randomValues <= 1))
        self.assertTrue(len(np.unique(randomValues)) == nWorkUnits)

    def testWhenGetRandomValuesASecondTime_shouldBeDifferent(self):
        nWorkUnits = 10
        np.random.seed(0)
        seeds = SeedCL(nWorkUnits)

        valueBuffer1 = EmptyBuffer(nWorkUnits)
        valueBuffer2 = EmptyBuffer(nWorkUnits)

        self.program.launchKernel("fillRandomFloatBuffer", N=nWorkUnits, arguments=[seeds, valueBuffer1])
        self.program.launchKernel("fillRandomFloatBuffer", N=nWorkUnits, arguments=[seeds, valueBuffer2])

        randomValues1 = self.program.getData(valueBuffer1)
        randomValues2 = self.program.getData(valueBuffer2)
        self.assertTrue(np.all(randomValues1 != randomValues2))

    def testGivenSameSeed_shouldGenerateSameRandomValues(self):
        nWorkUnits = 10
        np.random.seed(0)
        seeds = SeedCL(nWorkUnits)
        valueBuffer1 = EmptyBuffer(nWorkUnits)
        self.program.launchKernel("fillRandomFloatBuffer", N=nWorkUnits, arguments=[seeds, valueBuffer1])

        np.random.seed(0)
        seeds = SeedCL(nWorkUnits)
        valueBuffer2 = EmptyBuffer(nWorkUnits)
        self.program.launchKernel("fillRandomFloatBuffer", N=nWorkUnits, arguments=[seeds, valueBuffer2])

        randomValues1 = self.program.getData(valueBuffer1)
        randomValues2 = self.program.getData(valueBuffer2)
        self.assertTrue(np.all(randomValues1 == randomValues2))
