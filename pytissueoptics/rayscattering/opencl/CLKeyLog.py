import numpy as np

from pytissueoptics.scene.logger.logger import Logger
from pytissueoptics.rayscattering.opencl.CLScene import CLScene
from pytissueoptics.scene.logger import InteractionKey


SOLID_ID_COL = 4
SURFACE_ID_COL = 5


class CLKeyLog:
    def __init__(self, log: np.ndarray, sceneCL: CLScene):
        self._log = log
        self._sceneCL = sceneCL

        self._keyIndices = []
        self._keyLog = {}
        self._nBatch = 1000

        self._extractKeyLog()

    def toSceneLogger(self, sceneLogger: Logger):
        for key, points in self._keyLog.items():
            sceneLogger.logDataPointArray(points, key)

    def _extractKeyLog(self):
        self._sortLocal()
        self._merge()

    def _sortLocal(self):
        """ Sorts the log locally by solidID and surfaceID,
        and for each local batch, it also extracts start and end indices for each key. """

        # todo: try multi-threaded sort
        for bStart in range(0, self._log.shape[0], self._batchSize):
            ba, bb = bStart, bStart + self._batchSize
            batchLog = self._log[ba:bb]

            batchLog = batchLog[batchLog[:, 4].argsort()]

            solidChanges = self._getValueChangeIndices(batchLog, column=SOLID_ID_COL)

            batchKeyIndices = {}
            for i in range(len(solidChanges) - 1):
                solidID = batchLog[solidChanges[i], 4]
                if solidID == 0:
                    continue
                a, b = solidChanges[i], solidChanges[i + 1]
                batchLog[a:b] = batchLog[a:b][batchLog[a:b, 5].argsort()]

                surfaceChanges = self._getValueChangeIndices(batchLog[a:b], column=SURFACE_ID_COL)

                for j in range(len(surfaceChanges) - 1):
                    surfaceID = batchLog[a + surfaceChanges[j], 5]
                    c, d = surfaceChanges[j], surfaceChanges[j + 1]
                    batchKeyIndices[(int(solidID), int(surfaceID))] = (a + c, a + d)
            self._keyIndices.append(batchKeyIndices)
            self._log[ba:bb] = batchLog

    @staticmethod
    def _getValueChangeIndices(log: np.ndarray, column: int):
        """ Returns the indices where there is a change in the value of the given column. """
        indices = np.where(log[:-1, column] != log[1:, column])[0] + 1
        return np.concatenate(([0], indices, [log.shape[0]]))

    def _merge(self):
        """ Merges the local batches into a single key log with unique interaction keys. """
        self._log = self._log[:, :4]

        for i, batchKeyIndices in enumerate(self._keyIndices):
            bn = i * self._batchSize
            for keyIDs, indices in batchKeyIndices.items():
                if len(indices) == 0:
                    continue
                key = self._getInteractionKey(*keyIDs)

                points = self._log[indices[0] + bn: indices[1] + bn]
                if key not in self._keyLog:
                    self._keyLog[key] = [points]
                else:
                    self._keyLog[key].append(points)

        for key in self._keyLog:
            self._keyLog[key] = np.concatenate(self._keyLog[key])

    def _getInteractionKey(self, solidID: int, surfaceID: int):
        return InteractionKey(self._sceneCL.getSolidLabel(solidID), self._sceneCL.getSurfaceLabel(solidID, surfaceID))

    @property
    def _batchSize(self):
        return self._log.shape[0] // self._nBatch


"""
30k photons
transfer to logger time:
    No sort:
        3.1s
    Global sort:
        0.16s
    Local sort:
        0.49s with 30k units
        0.24s with 1k units

        3.7s + 1.8s = 5.5s with local sort
        [currently closer to 4.7s]

        10.8s + 1.7s = 12.5s without local sort
"""
