import os
import tempfile
import unittest

from pytissueoptics.rayscattering.opencl import OPENCL_OK
from pytissueoptics.rayscattering.opencl.config import CLConfig as clc


def tempConfigPath(func):
    def wrapper(*args, **kwargs):
        previousPath = clc.OPENCL_CONFIG_PATH
        with tempfile.TemporaryDirectory() as tempDir:
            clc.OPENCL_CONFIG_PATH = os.path.join(tempDir, "config.json")
            func(*args, **kwargs)
        clc.OPENCL_CONFIG_PATH = previousPath
    return wrapper


@unittest.skipIf(not OPENCL_OK, 'OpenCL device not available.')
class TestCLConfig(unittest.TestCase):
    @tempConfigPath
    def testGivenNoConfigFile_shouldWarnAndCreateANewOne(self):
        with self.assertWarns(UserWarning):
            clc.CLConfig()
        self.assertTrue(os.path.exists(clc.OPENCL_CONFIG_PATH))

    @tempConfigPath
    def testGivenNewConfigFile_shouldHaveDefaultValues(self):
        with self.assertWarns(UserWarning):
            config = clc.CLConfig()
        self.assertEqual(None, config.N_WORK_UNITS)
        self.assertEqual(None, config.MAX_MEMORY_MB)
        self.assertEqual(1000, config.IPP_TEST_N_PHOTONS)
        self.assertEqual(0.20, config.BATCH_LOAD_FACTOR)

    @tempConfigPath
    def testGivenCompleteConfigFile_shouldBeValid(self):
        with open(clc.OPENCL_CONFIG_PATH, "w") as f:
            f.write('{"DEVICE_INDEX": 0, "N_WORK_UNITS": 100, "MAX_MEMORY_MB": 1000, '
                    '"IPP_TEST_N_PHOTONS": 1000, "BATCH_LOAD_FACTOR": 0.2}')
        config = clc.CLConfig()
        config.validate()

    @tempConfigPath
    def testGivenMaxMemoryNotSet_whenValidate_shouldWarnAndSetMaxMemory(self):
        with open(clc.OPENCL_CONFIG_PATH, "w") as f:
            f.write('{"DEVICE_INDEX": 0, "N_WORK_UNITS": 100, "MAX_MEMORY_MB": null, '
                    '"IPP_TEST_N_PHOTONS": 1000, "BATCH_LOAD_FACTOR": 0.2}')
        config = clc.CLConfig()
        with self.assertWarns(UserWarning):
            config.validate()
        self.assertIsNotNone(config.MAX_MEMORY_MB)
        self.assertNotEqual(0, config.MAX_MEMORY_MB)

    @tempConfigPath
    def testGivenFileIsMissingParameter_whenValidate_shouldResetDefaultValueAndRaise(self):
        with open(clc.OPENCL_CONFIG_PATH, "w") as f:
            f.write('{"DEVICE_INDEX": 0, "N_WORK_UNITS": 100, "MAX_MEMORY_MB": 1000, '
                    '"IPP_TEST_N_PHOTONS": 1000}')
        config = clc.CLConfig()
        with self.assertRaises(ValueError):
            config.validate()
        config = clc.CLConfig()
        self.assertEqual(0.20, config.BATCH_LOAD_FACTOR)

    @tempConfigPath
    def testGivenFileWithAParameterBelowOrEqualToZero_whenValidate_shouldResetDefaultValueAndRaise(self):
        with open(clc.OPENCL_CONFIG_PATH, "w") as f:
            f.write('{"DEVICE_INDEX": 0, "N_WORK_UNITS": 100, "MAX_MEMORY_MB": 0, '
                    '"IPP_TEST_N_PHOTONS": 1000, "BATCH_LOAD_FACTOR": 0.2}')
        config = clc.CLConfig()
        with self.assertRaises(ValueError):
            config.validate()
        config = clc.CLConfig()
        self.assertIsNone(config.MAX_MEMORY_MB)
