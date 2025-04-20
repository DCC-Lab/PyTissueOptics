from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_AVAILABLE, WEIGHT_THRESHOLD, CLConfig, warnings
from pytissueoptics.rayscattering.opencl.config.IPPTable import IPPTable

OPENCL_OK = True

if OPENCL_AVAILABLE:
    try:
        CONFIG = CLConfig()
    except Exception as e:
        warnings.warn("Error creating OpenCL config: " + str(e))
        OPENCL_OK = False
        CONFIG = None
else:
    CONFIG = None


def validateOpenCL() -> bool:
    notAvailableMessage = "Error: Hardware acceleration not available. Falling back to CPU. "
    if not OPENCL_AVAILABLE:
        warnings.warn(notAvailableMessage + "Please install pyopencl.")
        return False
    if not OPENCL_OK:
        warnings.warn(notAvailableMessage + "Please fix OpenCL error above.")
        return False

    CONFIG.validate()
    return True


def hardwareAccelerationIsAvailable() -> bool:
    return OPENCL_AVAILABLE and OPENCL_OK


__all__ = [
    "IPPTable",
    "WEIGHT_THRESHOLD"
]
