import os
import json
import warnings
from typing import List

try:
    import pyopencl as cl

    OPENCL_AVAILABLE = True
except ImportError:
    OPENCL_AVAILABLE = False

warnings.formatwarning = lambda msg, *args, **kwargs: f'{msg}\n'

OPENCL_PATH = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH = os.path.dirname(os.path.dirname(OPENCL_PATH))
OPENCL_SOURCE_DIR = os.path.join(OPENCL_PATH, "src")

OPENCL_CONFIG_PATH = os.path.join(OPENCL_PATH, "config.json")
OPENCL_CONFIG_RELPATH = os.path.relpath(OPENCL_CONFIG_PATH, MODULE_PATH)

DEFAULT_CONFIG = {
    "DEVICE_INDEX": None,
    "N_WORK_UNITS": None,
    "MAX_MEMORY_MB": None,
    "IPP_TEST_N_PHOTONS": 1000,
    "BATCH_LOAD_FACTOR": 0.20
}

DEFAULT_MAX_MEMORY_MB = 1024

WEIGHT_THRESHOLD = 0.0001


class CLConfig:
    def __init__(self):
        self._config = None
        self._load()

        self._clContext = cl.create_some_context()

    def validate(self):
        errorMessage = f"The OpenCL config file at '{OPENCL_CONFIG_RELPATH}' is not valid. "
        parameterKeys = list(DEFAULT_CONFIG.keys())
        for key in parameterKeys:
            if key not in self._config:
                raise ValueError(errorMessage + "The parameter '{}' is missing.".format(key))

        self._validateDeviceIndex()
        self._validateMaxMemory()

        if self.N_WORK_UNITS is None:
            raise ValueError(
                errorMessage + "The parameter 'N_WORK_UNITS' is not set. Please set it to the optimal amount "
                               "for your OpenCL device. The script 'rayscattering/opencl/testWorkUnits.py' "
                               "can help you find this value. ")

        parameterKeys.pop(0)
        for key in parameterKeys:
            if not self._config[key] > 0:
                raise ValueError(errorMessage + "The parameter '{}' must be greater than 0.".format(key))

    def _validateDeviceIndex(self):
        numberOfDevices = len(self._devices)
        if self._DEVICE_INDEX is not None:
            if self._DEVICE_INDEX not in range(numberOfDevices):
                warnings.warn(
                    f"Invalid device index {self._DEVICE_INDEX}. Resetting to 'null' for automatic selection.")
                self._config["DEVICE_INDEX"] = None
                return self._validateDeviceIndex()
        elif numberOfDevices == 0:
            raise ValueError("No OpenCL devices found. Please install the OpenCL drivers for your hardware.")
        elif numberOfDevices == 1:
            self._showAvailableDevices()
            warnings.warn(
                f"Using the only available OpenCL device 0. If your desired device doesn't show, it may be "
                f"because its OpenCL drivers are not installed. To reset device selection, "
                f"reset DEVICE_INDEX parameter to 'null' in '{OPENCL_CONFIG_RELPATH}'.")
            self._config["DEVICE_INDEX"] = 0
        else:
            self._showAvailableDevices()
            deviceIndex = int(input(f"Please select your device by entering the corresponding index between "
                                    f"[0-{numberOfDevices - 1}]: "))
            assert deviceIndex in range(numberOfDevices), f"Invalid device index '{deviceIndex}'. Not in " \
                                                          f"range [0-{numberOfDevices - 1}]. "
            self._config["DEVICE_INDEX"] = deviceIndex
        self._save()

    def _validateMaxMemory(self):
        """
        If the user has not set a value for MAX_MEMORY_MB, it is set to the minimum value between 1GB and 75% of the
        device's max memory. The 1GB limit is used to prevent the user from running out of memory when using the CPU
        as OpenCL device (which has a max memory equal to the RAM).
        """
        if self._config["MAX_MEMORY_MB"] is None:
            maxDeviceMemoryMB = self.device.global_mem_size // 1024 ** 2
            if maxDeviceMemoryMB * 0.75 < DEFAULT_MAX_MEMORY_MB:
                self._config["MAX_MEMORY_MB"] = maxDeviceMemoryMB * 0.75
                warnings.warn(f"Setting MAX_MEMORY_MB to {self._config['MAX_MEMORY_MB']} MB, which corresponds to 75% "
                              f"of your device's memory.")
            else:
                self._config["MAX_MEMORY_MB"] = DEFAULT_MAX_MEMORY_MB
                warnings.warn(f"Setting MAX_MEMORY_MB to {self._config['MAX_MEMORY_MB']} MB to limit OutOfMemory "
                              f"errors even though your device has a capacity of {maxDeviceMemoryMB} MB. If you want "
                              f"to allow for more memory, you can manually change this value inside the config file "
                              f"at '{OPENCL_CONFIG_RELPATH}'.")
            self._save()

    def _load(self):
        self._assertExists()

        with open(OPENCL_CONFIG_PATH, "r") as f:
            self._config = json.load(f)

    def _assertExists(self):
        if not os.path.exists(OPENCL_CONFIG_PATH):
            warnings.warn("No OpenCL config file found. Creating a new one.")
            self._config = DEFAULT_CONFIG
            self._save()

    def _save(self):
        with open(OPENCL_CONFIG_PATH, "w") as f:
            json.dump(self._config, f, indent=4)

    @property
    def _devices(self) -> List[cl.Device]:
        return self._clContext.devices

    @property
    def _DEVICE_INDEX(self):
        return self._config["DEVICE_INDEX"]

    @property
    def device(self) -> cl.Device:
        return self._clContext.devices[self._DEVICE_INDEX]

    @property
    def N_WORK_UNITS(self):
        return self._config["N_WORK_UNITS"]

    @property
    def MAX_MEMORY(self):
        return self._config["MAX_MEMORY_MB"] * 1024 ** 2

    @property
    def IPP_TEST_N_PHOTONS(self):
        return self._config["IPP_TEST_N_PHOTONS"]

    @property
    def BATCH_LOAD_FACTOR(self):
        return self._config["BATCH_LOAD_FACTOR"]

    @property
    def WEIGHT_THRESHOLD(self):
        return WEIGHT_THRESHOLD

    @property
    def clContext(self):
        return self._clContext

    def _showAvailableDevices(self):
        print("Available devices:")
        for i, device in enumerate(self._devices):
            print(f"... Device [{i}]: {device.name} ({device.global_mem_size // 1024 ** 2} MB "
                  f"| {device.max_clock_frequency} MHz)")
