from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

import numpy as np

from pytissueoptics.scene.geometry import Vector


@dataclass(frozen=True)
class InteractionKey:
    solidLabel: str
    surfaceLabel: str = None


@dataclass
class InteractionData:
    points: np.ndarray = None
    dataPoints: np.ndarray = None
    segments: np.ndarray = None


class DataType(Enum):
    POINT = "points"
    DATA_POINT = "dataPoints"
    SEGMENT = "segments"


class Logger:
    def __init__(self):
        self._data: Dict[InteractionKey, InteractionData] = {}

    def getSolidLabels(self) -> List[str]:
        return list(set(key.solidLabel for key in self._data.keys()))

    def getSurfaceLabels(self, solidLabel: str) -> List[str]:
        return [key.surfaceLabel for key in self._data.keys() if key.solidLabel == solidLabel
                and key.surfaceLabel is not None]

    def logPoint(self, point: Vector, key: InteractionKey):
        self._appendData(np.array([[point.x, point.y, point.z]]).T, DataType.POINT, key)

    def logDataPoint(self, value: float, position: Vector, key: InteractionKey):
        self._appendData(np.array([[value, position.x, position.y, position.z]]).T, DataType.DATA_POINT, key)

    def logSegment(self, start: Vector, end: Vector, key: InteractionKey):
        self._appendData(np.array([[start.x, start.y, start.z, end.x, end.y, end.z]]).T, DataType.SEGMENT, key)

    def logPointArray(self, array: np.ndarray, key: InteractionKey):
        """ 'array' must be of shape (3, n) where first axis is (x, y, z) """
        assert array.shape[0] == 3 and array.ndim == 2, "Point array must be of shape (3, n)"
        self._appendData(array, DataType.POINT, key)

    def logDataPointArray(self, array: np.ndarray, key: InteractionKey):
        """ 'array' must be of shape (4, n) where first axis is (value, x, y, z) """
        assert array.shape[0] == 4 and array.ndim == 2, "Data point array must be of shape (4, n)"
        self._appendData(array, DataType.DATA_POINT, key)

    def logSegmentArray(self, array: np.ndarray, key: InteractionKey):
        """ 'array' must be of shape (6, n) where first axis is (x1, y1, z1, x2, y2, z2) """
        assert array.shape[0] == 6 and array.ndim == 2, "Segment array must be of shape (6, n)"
        self._appendData(array, DataType.SEGMENT, key)

    def _appendData(self, dataArray: np.ndarray, dataType: DataType, key: InteractionKey):
        self._validateKey(key)
        previousData = getattr(self._data[key], dataType.value)
        if previousData is None:
            setattr(self._data[key], dataType.value, dataArray)
        else:
            setattr(self._data[key], dataType.value, np.concatenate((previousData, dataArray), axis=1))

    def _validateKey(self, key: InteractionKey):
        if key not in self._data:
            self._data[key] = InteractionData()

    def getPoints(self, key: InteractionKey = None) -> np.ndarray:
        return self._getData(DataType.POINT, key)

    def getDataPoints(self, key: InteractionKey = None) -> np.ndarray:
        return self._getData(DataType.DATA_POINT, key)

    def getSegments(self, key: InteractionKey = None) -> np.ndarray:
        return self._getData(DataType.SEGMENT, key)

    def _getData(self, dataType: DataType, key: InteractionKey = None) -> Optional[np.ndarray]:
        if key and key.solidLabel:
            self._assertKeyExists(key)
            return getattr(self._data[key], dataType.value)
        else:
            data = []
            for interactionData in self._data.values():
                data.append(getattr(interactionData, dataType.value))
            if len(data) == 0:
                return None
            return np.concatenate(data, axis=1)

    def _assertKeyExists(self, key: InteractionKey):
        if key.solidLabel not in self.getSolidLabels():
            raise KeyError(f"Invalid solid label '{key.solidLabel}'. Available: {self.getSolidLabels()}. ")
        if key.surfaceLabel and key.surfaceLabel not in self.getSurfaceLabels(key.solidLabel):
            raise KeyError(f"Invalid surface label '{key.surfaceLabel}' for solid '{key.solidLabel}'. "
                           f"Available: {self.getSurfaceLabels(key.solidLabel)}. ")
