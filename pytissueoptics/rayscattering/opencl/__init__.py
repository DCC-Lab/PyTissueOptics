from pytissueoptics.rayscattering.opencl.config.CLConfig import OPENCL_AVAILABLE, WEIGHT_THRESHOLD, CLConfig, warnings
from pytissueoptics.rayscattering.opencl.config.IPPTable import IPPTable
import os

OPENCL_OK = True
OPENCL_DISABLED = os.environ.get("PTO_DISABLE_OPENCL", "0") == "1"

if OPENCL_AVAILABLE:
    try:
        CONFIG = CLConfig()
    except Exception as e:
        warnings.warn("Error creating OpenCL config: " + str(e))
        OPENCL_OK = False
        CONFIG = None
else:
    CONFIG = None


def disableOpenCL():
    os.environ["PTO_DISABLE_OPENCL"] = "1"
    print("You can define PTO_DISABLE_OPENCL=1 in your profile to avoid this call.")


def validateOpenCL() -> bool:
    notAvailableMessage = "Error: Hardware acceleration not available. Falling back to CPU. "

    if os.environ.get("PTO_DISABLE_OPENCL", "0") == "1":
        warnings.warn("User requested not to use OpenCL with environment variable 'PTO_DISABLE_OPENCL'=1.")
        return False
    if not OPENCL_AVAILABLE:
        warnings.warn(notAvailableMessage + "Please install pyopencl.")
        return False
    if not OPENCL_OK:
        warnings.warn(notAvailableMessage + "Please fix OpenCL error above.")
        return False

    CONFIG.validate()
    return True


def hardwareAccelerationIsAvailable() -> bool:
    return OPENCL_AVAILABLE and OPENCL_OK and not OPENCL_DISABLED


__all__ = ["IPPTable", "WEIGHT_THRESHOLD"]
