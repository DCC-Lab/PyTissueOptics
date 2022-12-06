import json
import os
import warnings

try:
    import pyopencl as cl

    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

from .IPPTable import IPPTable
OPENCL_SOURCE_DIR = os.path.join(os.path.dirname(__file__), "src")
OPENCL_PATH = os.path.dirname(os.path.abspath(__file__))

CONFIG = json.load(open(os.path.join(OPENCL_PATH, 'config.json')))

# Hardware-specific constants
N_WORK_UNITS = CONFIG['N_WORK_UNITS']
MAX_MEMORY = CONFIG['MAX_MEMORY_MB'] * 1024 ** 2
BATCH_LOAD_FACTOR = CONFIG['BATCH_LOAD_FACTOR']

# Constants
IPP_TEST_N_PHOTONS = CONFIG['IPP_TEST_N_PHOTONS']
WEIGHT_THRESHOLD = 0.0001


def validateOpenCL() -> bool:
    if not OPENCL_AVAILABLE:
        warnings.warn("Hardware acceleration not available. Falling back to CPU. Please install pyopencl.")
        return False
    errorMessage = "The rayscattering/opencl/config.json file is not valid. "
    parameterKeys = ["N_WORK_UNITS", "MAX_MEMORY_MB", "BATCH_LOAD_FACTOR", "IPP_TEST_N_PHOTONS"]
    for key in parameterKeys:
        if key not in CONFIG:
            raise ValueError(errorMessage + "The parameter '{}' is missing.".format(key))
    if CONFIG["N_WORK_UNITS"] is None:
        raise ValueError(errorMessage + "The parameter 'N_WORK_UNITS' is not set. Please set it to the optimal amount "
                                        "for your OpenCL device. The script 'rayscattering/opencl/testWorkUnits.py' "
                                        "can help you find this value.")
    for key in parameterKeys:
        if not CONFIG[key] > 0:
            raise ValueError(errorMessage + "The parameter '{}' must be greater than 0.".format(key))
    return True
