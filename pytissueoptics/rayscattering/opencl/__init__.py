import os

from .CLPhotons import CLPhotons, OPENCL_AVAILABLE
from .IPPTable import IPPTable
OPENCL_SOURCE_DIR = os.path.join(os.path.dirname(__file__), "src")
