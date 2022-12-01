import json
import os

import numpy as np

from pytissueoptics.rayscattering.opencl import CLObjects as clObjects

OPENCL_PATH = os.path.dirname(os.path.abspath(__file__))


# Experiment-specific constants
AVG_IT_PER_PHOTON = None

# Hardware-specific constants
# todo: run an hardware test to determine the best work item amount if not specified in config
config = json.load(open(os.path.join(OPENCL_PATH, 'config.json')))
N_WORK_UNITS = config['N_WORK_UNITS']
MAX_MEMORY = config['MAX_MEMORY_MB'] * 1024 ** 2

# Constants
N_BATCHES = 5
WEIGHT_THRESHOLD = 0.0001
DATAPOINT_SIZE = clObjects.DataPointCL.getItemSize()


class CLParameters:
    def __init__(self, N):
        self._photonAmount = int(N / min(N_BATCHES, N_WORK_UNITS))
        self._maxLoggerMemory = self._photonAmount * AVG_IT_PER_PHOTON * DATAPOINT_SIZE
        self._maxLoggerMemory = min(self._maxLoggerMemory, MAX_MEMORY)
        self._photonAmount = min(2 * self._photonAmount, N)

        self._workItemAmount = N_WORK_UNITS

    @property
    def workItemAmount(self):
        return np.int32(self._workItemAmount)

    @workItemAmount.setter
    def workItemAmount(self, value: int):
        self._workItemAmount = value

    @property
    def maxLoggerMemory(self):
        return np.int32(self._maxLoggerMemory)

    @maxLoggerMemory.setter
    def maxLoggerMemory(self, value: int):
        self._maxLoggerMemory = value

    @property
    def photonAmount(self):
        return np.int32(self._photonAmount)

    @photonAmount.setter
    def photonAmount(self, value: int):
        if value < self._workItemAmount:
            self._workItemAmount = value
        self._photonAmount = value

    @property
    def maxLoggableInteractions(self):
        return np.int32(self._maxLoggerMemory / DATAPOINT_SIZE)

    @maxLoggableInteractions.setter
    def maxLoggableInteractions(self, value):
        self._maxLoggerMemory = np.int32(value * DATAPOINT_SIZE)

    @property
    def maxLoggableInteractionsPerWorkItem(self):
        return np.int32((self.maxLoggerMemory / DATAPOINT_SIZE) / self._workItemAmount)

    @maxLoggableInteractionsPerWorkItem.setter
    def maxLoggableInteractionsPerWorkItem(self, value):
        self._maxLoggerMemory = np.int32((value * DATAPOINT_SIZE) * self._workItemAmount)

    @property
    def photonsPerWorkItem(self):
        return np.int32(np.floor(self._photonAmount / self._workItemAmount))
    
    @photonsPerWorkItem.setter
    def photonsPerWorkItem(self, value: int):
        self._photonAmount = np.int32(value * self._workItemAmount)
