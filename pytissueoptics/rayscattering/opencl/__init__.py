import os

from .CLPhotons import CLPhotons, OPENCL_AVAILABLE
from .IPPTable import IPPTable
from .CLParameters import IPP_TEST_N_PHOTONS, WEIGHT_THRESHOLD
OPENCL_SOURCE_DIR = os.path.join(os.path.dirname(__file__), "src")
