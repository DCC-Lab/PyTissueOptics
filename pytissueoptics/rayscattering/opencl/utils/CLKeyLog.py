from multiprocessing.pool import ThreadPool
from typing import List

import numpy as np

from pytissueoptics.rayscattering.opencl.CLScene import NO_LOG_ID, NO_SOLID_LABEL, CLScene
from pytissueoptics.scene.logger import InteractionKey, Logger

SOLID_ID_COL = 4
SURFACE_ID_COL = 5


class CLKeyLog:
    """Parses a DataPointCL array of shape (N, 6) where each point is of the form
    (weight, x, y, z, solidID, surfaceID) to extract a dictionary of InteractionKey
    and their corresponding datapoint array of the form (weight, x, y, z). The
    translation from IDs to their corresponding labels is done using the given CLScene."""

    def __init__(self, log: np.ndarray, sceneCL: CLScene):
        self._log = log
        self._sceneCL = sceneCL

        self._keyIndices = []
        self._keyLog = {}

        self._batchSize = min(50000, len(self._log))

        self._extractKeyLog()

    def toSceneLogger(self, sceneLogger: Logger):
        """Writes the extracted key-based log to the given scene logger."""
        for key, points in self._keyLog.items():
            sceneLogger.logDataPointArray(points, key)

    def _extractKeyLog(self):
        if self._sceneCL.nSolids == 0:
            return self._extractNoKeyLog()
        self._sortLocal()
        self._merge()

    def _extractNoKeyLog(self):
        noInteractionIndices = np.where(self._log[:, SOLID_ID_COL] == NO_LOG_ID)[0]
        self._log = self._log[:, :4]
        self._log = np.delete(self._log, noInteractionIndices, axis=0)
        self._keyLog[InteractionKey(NO_SOLID_LABEL, None)] = self._log

    def _sortLocal(self):
        """Sorts the log locally by solidID and surfaceID,
        and for each local batch, it also extracts start and end indices for each key."""

        pool = ThreadPool(8)
        try:
            results = [
                pool.apply_async(self._sortBatch, args=(batchStartIdx,))
                for batchStartIdx in range(0, self._log.shape[0], self._batchSize)
            ]
            self._keyIndices = [res.get() for res in results]
        finally:
            pool.close()
            pool.join()

    def _sortBatch(self, startIdx: int):
        ba, bb = startIdx, startIdx + self._batchSize
        batchLog = self._sort(self._log[ba:bb], columns=[SOLID_ID_COL, SURFACE_ID_COL])

        solidChanges = self._getValueChangeIndices(batchLog, column=SOLID_ID_COL)

        batchKeyIndices = {}
        for i in range(len(solidChanges) - 1):
            solidID = batchLog[solidChanges[i], SOLID_ID_COL]
            if solidID == NO_LOG_ID:
                continue
            a, b = solidChanges[i], solidChanges[i + 1]
            batchSolidLog = batchLog[a:b]

            surfaceChanges = self._getValueChangeIndices(batchSolidLog, column=SURFACE_ID_COL)
            for j in range(len(surfaceChanges) - 1):
                surfaceID = batchSolidLog[surfaceChanges[j], SURFACE_ID_COL]
                c, d = surfaceChanges[j], surfaceChanges[j + 1]
                batchKeyIndices[(int(solidID), int(surfaceID))] = (a + c, a + d)

            batchLog[a:b] = batchSolidLog
        self._log[ba:bb] = batchLog
        return batchKeyIndices

    @staticmethod
    def _getValueChangeIndices(log: np.ndarray, column: int):
        """Returns the indices where there is a change in the value of the given column."""
        indices = np.where(log[:-1, column] != log[1:, column])[0] + 1
        return np.concatenate(([0], indices, [log.shape[0]]))

    @staticmethod
    def _sort(log: np.ndarray, columns: List[int]):
        return log[np.lexsort([log[:, i] for i in columns])]

    def _merge(self):
        """Merges the local batches into a single key log with unique interaction keys."""
        self._log = self._log[:, :4]

        for i, batchKeyIndices in enumerate(self._keyIndices):
            batchStartIndex = i * self._batchSize
            for keyIDs, indices in batchKeyIndices.items():
                if len(indices) == 0:
                    continue
                a, b = batchStartIndex + indices[0], batchStartIndex + indices[1]
                points = self._log[a:b]
                key = self._getInteractionKey(*keyIDs)
                if key not in self._keyLog:
                    self._keyLog[key] = [points]
                else:
                    self._keyLog[key].append(points)

        for key in self._keyLog:
            self._keyLog[key] = np.concatenate(self._keyLog[key])

    def _getInteractionKey(self, solidID: int, surfaceID: int):
        return InteractionKey(self._sceneCL.getSolidLabel(solidID), self._sceneCL.getSurfaceLabel(solidID, surfaceID))
