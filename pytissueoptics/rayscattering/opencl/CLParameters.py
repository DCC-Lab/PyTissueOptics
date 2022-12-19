import psutil
import numpy as np

from pytissueoptics.rayscattering.opencl.CLObjects import DataPointCL
from pytissueoptics.rayscattering.opencl import CONFIG, warnings


DATAPOINT_SIZE = DataPointCL.getItemSize()


class CLParameters:
    def __init__(self, N, AVG_IT_PER_PHOTON):
        nBatch = 1/CONFIG.BATCH_LOAD_FACTOR
        avgPhotonsPerBatch = int(N / min(nBatch, CONFIG.N_WORK_UNITS))
        self._maxLoggerMemory = avgPhotonsPerBatch * AVG_IT_PER_PHOTON * DATAPOINT_SIZE
        self._maxLoggerMemory = min(self._maxLoggerMemory, CONFIG.MAX_MEMORY)
        self._maxPhotonsPerBatch = min(2 * avgPhotonsPerBatch, N)
        self._workItemAmount = CONFIG.N_WORK_UNITS

        self._assertEnoughRAM()

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
    def maxPhotonsPerBatch(self):
        return np.int32(self._maxPhotonsPerBatch)

    @maxPhotonsPerBatch.setter
    def maxPhotonsPerBatch(self, value: int):
        if value < self._workItemAmount:
            self._workItemAmount = value
        self._maxPhotonsPerBatch = value

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
        return np.int32(np.floor(self._maxPhotonsPerBatch / self._workItemAmount))
    
    @photonsPerWorkItem.setter
    def photonsPerWorkItem(self, value: int):
        self._maxPhotonsPerBatch = np.int32(value * self._workItemAmount)

    @property
    def requiredRAMBytes(self) -> float:
        averageNBatches = int(1.3 * (1 / CONFIG.BATCH_LOAD_FACTOR))
        concatenationFactor = 2
        overHead = 1.15
        return overHead * concatenationFactor * averageNBatches * self._maxLoggerMemory

    def _assertEnoughRAM(self):
        freeSystemRAM = psutil.virtual_memory().available
        if self.requiredRAMBytes > 0.9 * freeSystemRAM:
            warnings.warn(f"WARNING: Available system RAM might not be enough for the simulation. "
                          f"Estimated requirement: {self.requiredRAMBytes // 1024**2} MB, "
                          f"Available: {freeSystemRAM // 1024**2} MB.")
