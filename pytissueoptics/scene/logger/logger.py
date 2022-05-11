import os
import pickle
import warnings
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
from enum import Enum

import numpy as np

from pytissueoptics.scene.logger.listArrayContainer import ListArrayContainer
from pytissueoptics.scene.geometry import Vector


@dataclass(frozen=True)
class InteractionKey:
    solidLabel: str
    surfaceLabel: str = None


@dataclass
class InteractionData:
    points: ListArrayContainer = None
    dataPoints: ListArrayContainer = None
    segments: ListArrayContainer = None


class DataType(Enum):
    POINT = "points"
    DATA_POINT = "dataPoints"
    SEGMENT = "segments"


class Logger:
    def __init__(self, fromFilepath: str = None):
        self._data: Dict[InteractionKey, InteractionData] = {}
        self.info: dict = {}
        self._filepath = None

        if fromFilepath:
            self.load(fromFilepath)

    def getSolidLabels(self) -> List[str]:
        return list(set(key.solidLabel for key in self._data.keys()))

    def getSurfaceLabels(self, solidLabel: str) -> List[str]:
        return [key.surfaceLabel for key in self._data.keys() if key.solidLabel == solidLabel
                and key.surfaceLabel is not None]

    def logPoint(self, point: Vector, key: InteractionKey):
        self._appendData([point.x, point.y, point.z], DataType.POINT, key)

    def logDataPoint(self, value: float, position: Vector, key: InteractionKey):
        self._appendData([value, position.x, position.y, position.z], DataType.DATA_POINT, key)

    def logSegment(self, start: Vector, end: Vector, key: InteractionKey):
        self._appendData([start.x, start.y, start.z, end.x, end.y, end.z], DataType.SEGMENT, key)

    def logPointArray(self, array: np.ndarray, key: InteractionKey):
        """ 'array' must be of shape (n, 3) where second axis is (x, y, z) """
        assert array.shape[1] == 3 and array.ndim == 2, "Point array must be of shape (n, 3)"
        self._appendData(array, DataType.POINT, key)

    def logDataPointArray(self, array: np.ndarray, key: InteractionKey):
        """ 'array' must be of shape (n, 4) where second axis is (value, x, y, z) """
        assert array.shape[1] == 4 and array.ndim == 2, "Data point array must be of shape (n, 4)"
        self._appendData(array, DataType.DATA_POINT, key)

    def logSegmentArray(self, array: np.ndarray, key: InteractionKey):
        """ 'array' must be of shape (n, 6) where second axis is (x1, y1, z1, x2, y2, z2) """
        assert array.shape[1] == 6 and array.ndim == 2, "Segment array must be of shape (n, 6)"
        self._appendData(array, DataType.SEGMENT, key)

    def _appendData(self, data: Union[List, np.ndarray], dataType: DataType, key: InteractionKey):
        self._validateKey(key)
        previousData = getattr(self._data[key], dataType.value)
        if previousData is None:
            previousData = ListArrayContainer()
            previousData.append(data)
            setattr(self._data[key], dataType.value, previousData)
        else:
            previousData.append(data)

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
            if not self._keyExists(key):
                return None
            data = getattr(self._data[key], dataType.value)
            data.merge()
            return data.mergedData
        else:
            tempData = ListArrayContainer()
            for interactionData in self._data.values():
                points = getattr(interactionData, dataType.value)
                if points is None:
                    continue
                tempData.extend(points)
            if len(tempData) == 0:
                return None
            tempData.merge()
            return tempData.mergedData

    def _keyExists(self, key: InteractionKey) -> bool:
        if key.solidLabel not in self.getSolidLabels():
            warnings.warn(f"No data stored for solid labeled '{key.solidLabel}'. Available: {self.getSolidLabels()}. ")
        if key.surfaceLabel and key.surfaceLabel not in self.getSurfaceLabels(key.solidLabel):
            warnings.warn(f"No data stored for surface labeled '{key.surfaceLabel}' for solid '{key.solidLabel}'. "
                          f"Available: {self.getSurfaceLabels(key.solidLabel)}. ")
        if key in self._data:
            return True
        if not key.surfaceLabel:
            warnings.warn(f"No data stored inside the solid labeled '{key.solidLabel}'.")
        return False

    def save(self, filepath: str = None):
        if filepath is None and self._filepath is None:
            filepath = "simulation.log"
            warnings.warn(f"No filepath specified. Saving to {filepath}.")
        elif filepath is None:
            filepath = self._filepath

        with open(filepath, "wb") as file:
            pickle.dump((self._data, self.info), file)

    def load(self, filepath: str):
        self._filepath = filepath

        if not os.path.exists(filepath):
            warnings.warn("No logger file found at '{}'. No data loaded.".format(filepath))
            return

        with open(filepath, "rb") as file:
            self._data, self.info = pickle.load(file)
