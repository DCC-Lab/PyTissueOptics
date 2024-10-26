import math
import os
import unittest
from dataclasses import dataclass

import numpy as np

from pytissueoptics import Vector, ScatteringMaterial, ScatteringScene
from pytissueoptics.rayscattering.opencl import OPENCL_AVAILABLE
from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_SOURCE_DIR
from pytissueoptics.rayscattering.opencl.CLProgram import CLProgram
from pytissueoptics.rayscattering.opencl.CLScene import NO_SURFACE_ID, NO_LOG_ID, NO_SOLID_ID, CLScene
from pytissueoptics.rayscattering.opencl.buffers import *
from pytissueoptics.scene.intersection.mollerTrumboreIntersect import EPS_CORRECTION


if OPENCL_AVAILABLE:
    import pyopencl as cl
else:
    cl = None


@unittest.skipIf(not OPENCL_AVAILABLE, 'Requires PyOpenCL.')
class TestCLProgram(unittest.TestCase):
    def test_init_valid_file(self):
        source_path = os.path.join(OPENCL_SOURCE_DIR, "random.c")
        self.assertIsNotNone(CLProgram(source_path))

    def test_init_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            self.assertIsNotNone(CLProgram("."))


    def test_call_random_value(self):
        nWorkUnits = 10
        seeds = SeedCL(nWorkUnits)
        valueBuffer = EmptyBuffer(nWorkUnits)

        source_path = os.path.join(OPENCL_SOURCE_DIR, "random.c")
        program = CLProgram(source_path)
        program.launchKernel(kernelName='fillRandomFloatBuffer', N=nWorkUnits, arguments = [seeds, valueBuffer])

        for value in seeds.hostBuffer:
            self.assertTrue(value != 0)

    def test_call_random_value_with_same_seeds_buffer_should_give_same_results(self):
        """
        If we use the same seeds, then the two calculations should be the same.
        This tests fails and should not, but the impact is minimal: the cude still runs fine
        it simply means we cannot run the same computation twice for testing.

        However, the same tests but with the CLObjects runs fine in testGeneralOpenCL.py
        """
        nWorkUnits = 10

        seed_array = np.array([12345]*nWorkUnits, dtype=cl.cltypes.uint)
        seeds1 = BufferOf(seed_array)
        seeds2 = BufferOf(seed_array)

        valueBuffer1 = Buffer(nWorkUnits, value=0)
        valueBuffer2 = Buffer(nWorkUnits, value=0)

        source_path = os.path.join(OPENCL_SOURCE_DIR, "random.c")
        program = CLProgram(source_path)
        
        program.launchKernel(kernelName='fillRandomFloatBuffer', N=nWorkUnits, arguments = [seeds1, valueBuffer1])
        program.launchKernel(kernelName='fillRandomFloatBuffer', N=nWorkUnits, arguments = [seeds2, valueBuffer2])

        # We need to sort because values are not necessarily in order
        print(sorted(valueBuffer1.hostBuffer))
        print(sorted(valueBuffer2.hostBuffer))

        for value1, value2 in zip(sorted(valueBuffer1.hostBuffer), sorted(valueBuffer2.hostBuffer)) :
            self.assertEqual(value1, value2)

    def test_call_random_value_with_single_seed_should_give_single_result(self):
        nWorkUnits = 10
        seed_array = np.array([12345]*nWorkUnits, dtype=cl.cltypes.uint)
        seeds = BufferOf(seed_array)

        valueBuffer = EmptyBuffer(nWorkUnits)

        source_path = os.path.join(OPENCL_SOURCE_DIR, "random.c")
        program = CLProgram(source_path)
        
        program.launchKernel(kernelName='fillRandomFloatBuffer', N=nWorkUnits, arguments = [seeds, valueBuffer])

        self.assertEqual(len(set(valueBuffer.hostBuffer)), 1)

if __name__ == "__main__":
    unittest.main()
