import numpy as np
import psutil

from pytissueoptics.rayscattering.opencl import CONFIG, warnings
from pytissueoptics.rayscattering.opencl.buffers import DataPointCL

DATAPOINT_SIZE = DataPointCL.getItemSize()


class CLParameters:
    def __init__(self, N, AVG_IT_PER_PHOTON):
        nBatch = 1 / CONFIG.BATCH_LOAD_FACTOR
        avgPhotonsPerBatch = int(np.ceil(N / min(nBatch, CONFIG.N_WORK_UNITS)))
        self._maxLoggerMemory = self._calculateAverageBatchMemorySize(avgPhotonsPerBatch, AVG_IT_PER_PHOTON)
        self._workItemAmount = CONFIG.N_WORK_UNITS
        self.maxPhotonsPerBatch = min(2 * avgPhotonsPerBatch, N)

        self._assertEnoughRAM()

    def _calculateAverageBatchMemorySize(self, avgPhotonsPerBatch: int, avgInteractionsPerPhoton: float) -> int:
        """
        Calculates the required number of bytes to allocate for each batch when expecting the given average number of
        interactions per photon. Note that each work unit requires a minimum of 2 available log entries to operate.
        """
        avgInteractions = avgPhotonsPerBatch * avgInteractionsPerPhoton
        minInteractions = 2 * CONFIG.N_WORK_UNITS
        batchSize = max(avgInteractions, minInteractions) * DATAPOINT_SIZE
        maxSize = CONFIG.MAX_MEMORY_MB * 1024**2
        return min(batchSize, maxSize)

    @property
    def workItemAmount(self):
        return np.int32(self._workItemAmount)

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

    @property
    def maxLoggableInteractionsPerWorkItem(self):
        return np.int32(self.maxLoggableInteractions / self._workItemAmount)

    @property
    def photonsPerWorkItem(self):
        return np.int32(np.floor(self._maxPhotonsPerBatch / self._workItemAmount))

    @property
    def requiredRAMBytes(self) -> float:
        averageNBatches = 1.4 * (1 / CONFIG.BATCH_LOAD_FACTOR)
        overHead = 1.15
        return overHead * averageNBatches * self._maxLoggerMemory

    def _assertEnoughRAM(self):
        freeSystemRAM = psutil.virtual_memory().available
        if self.requiredRAMBytes > 0.8 * freeSystemRAM:
            warnings.warn(
                f"WARNING: Available system RAM might not be enough for the simulation. "
                f"Estimated requirement: {self.requiredRAMBytes // 1024**2} MB, "
                f"Available: {freeSystemRAM // 1024**2} MB."
            )
