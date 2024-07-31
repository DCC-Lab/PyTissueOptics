from pytissueoptics.rayscattering.opencl.config.CLConfig import warnings, CLConfig, OPENCL_AVAILABLE, WEIGHT_THRESHOLD
from pytissueoptics.rayscattering.opencl.config.IPPTable import IPPTable
import os

OPENCL_OK = True
# PYTISSUE_FORCE_CPU = os.environ.get('PYTISSUE_FORCE_CPU', '0')
# AVOID_SMOOTHING_BUG = os.environ.get('AVOID_SMOOTHING_BUG', '0')

if OPENCL_AVAILABLE:
    try:
        CONFIG = CLConfig()
    except Exception as e:
        OPENCL_OK = False
        CONFIG = None
else:
    CONFIG = None

def forceCalculationOnCPU():
    os.environ["PYTISSUE_FORCE_CPU"] = "1"
    print("You can define PYTISSUE_FORCE_CPU=1 in your profile to avoid this call.")

def avoidSmoothingBug():
    os.environ["AVOID_SMOOTHING_BUG"] = "1"
    print("You can define AVOID_SMOOTHING_BUG=1 in your profile to avoid this call.")

def validateOpenCL() -> bool:
    notAvailableMessage = "Error: Hardware acceleration not available. Falling back to CPU. "
    
    if os.environ.get("PYTISSUE_FORCE_CPU", '0') != '0':
        warnings.warn("User requested not using OpenCL with environment variable 'PYTISSUE_FORCE_CPU'=1.")
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
    PYTISSUE_FORCE_CPU = (os.environ.get('PYTISSUE_FORCE_CPU', '0') == '1')
    return OPENCL_AVAILABLE and OPENCL_OK and (not PYTISSUE_FORCE_CPU)
