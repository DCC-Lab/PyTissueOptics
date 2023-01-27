from pytissueoptics.rayscattering.opencl.config.CLConfig import warnings, CLConfig, OPENCL_AVAILABLE, WEIGHT_THRESHOLD
from pytissueoptics.rayscattering.opencl.config.IPPTable import IPPTable

if OPENCL_AVAILABLE:
    CONFIG = CLConfig()
else:
    CONFIG = None


def validateOpenCL() -> bool:
    if not OPENCL_AVAILABLE:
        warnings.warn("Hardware acceleration not available. Falling back to CPU. Please install pyopencl.")
        return False

    CONFIG.validate()
    return True
