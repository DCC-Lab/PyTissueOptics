import os
import pickle
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.logger.listArrayContainer import ListArrayContainer


@dataclass(frozen=True)
class InteractionKey:
    solidLabel: str
    surfaceLabel: Optional[str] = None

    @property
    def volumetric(self) -> bool:
        return self.surfaceLabel is None


TransformFn = Callable[[InteractionKey, Optional[np.ndarray]], Optional[np.ndarray]]


def _noTransform(_: InteractionKey, data: Optional[np.ndarray]) -> Optional[np.ndarray]:
    return data


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
    DEFAULT_LOGGER_PATH = "simulation.log"

    def __init__(self, fromFilepath: str = None):
        self._data: Dict[InteractionKey, InteractionData] = {}
        self.info: dict = {}
        self._filepath = None
        self._labels = {}

        if fromFilepath:
            self.load(fromFilepath)

    def getSeenSolidLabels(self) -> List[str]:
        """Returns a list of all solid labels that have been logged in the past
        even if the data was discarded."""
        return list(self._labels.keys())

    def getSeenSurfaceLabels(self, solidLabel: str) -> List[str]:
        """Returns a list of all surface labels that have been logged in the past
        for the given solid even if the data was discarded."""
        return self._labels[solidLabel]

    def getStoredSolidLabels(self) -> List[str]:
        """Returns a list of all solid labels that are currently stored in the logger."""
        return list(set(key.solidLabel for key in self._data.keys()))

    def getStoredSurfaceLabels(self, solidLabel: str) -> List[str]:
        """Returns a list of all surface labels that are currently stored in the logger."""
        return [
            key.surfaceLabel
            for key in self._data.keys()
            if key.solidLabel == solidLabel and key.surfaceLabel is not None
        ]

    def logPoint(self, point: Vector, key: InteractionKey = None):
        self._appendData([point.x, point.y, point.z], DataType.POINT, key)

    def logDataPoint(self, value: float, position: Vector, key: InteractionKey):
        self._appendData([value, position.x, position.y, position.z], DataType.DATA_POINT, key)

    def logSegment(self, start: Vector, end: Vector, key: InteractionKey = None):
        self._appendData([start.x, start.y, start.z, end.x, end.y, end.z], DataType.SEGMENT, key)

    def logPointArray(self, array: np.ndarray, key: InteractionKey = None):
        """'array' must be of shape (n, 3) where the second axis is (x, y, z)"""
        assert array.shape[1] == 3 and array.ndim == 2, "Point array must be of shape (n, 3)"
        self._appendData(array, DataType.POINT, key)

    def logDataPointArray(self, array: np.ndarray, key: InteractionKey):
        """'array' must be of shape (n, 4) where the second axis is (value, x, y, z)"""
        assert array.shape[1] == 4 and array.ndim == 2, "Data point array must be of shape (n, 4)"
        self._appendData(array, DataType.DATA_POINT, key)

    def logSegmentArray(self, array: np.ndarray, key: InteractionKey = None):
        """'array' must be of shape (n, 6) where the second axis is (x1, y1, z1, x2, y2, z2)"""
        assert array.shape[1] == 6 and array.ndim == 2, "Segment array must be of shape (n, 6)"
        self._appendData(array, DataType.SEGMENT, key)

    def _appendData(self, data: Union[List, np.ndarray], dataType: DataType, key: InteractionKey = None):
        if key is None:
            key = InteractionKey(None, None)
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
        if key.solidLabel is None:
            return
        if key.solidLabel not in self._labels:
            self._labels[key.solidLabel] = []
        if key.surfaceLabel is None:
            return
        if key.surfaceLabel not in self._labels[key.solidLabel]:
            self._labels[key.solidLabel].append(key.surfaceLabel)

    def getPoints(self, key: InteractionKey = None) -> np.ndarray:
        return self._getData(DataType.POINT, key)

    def getRawDataPoints(self, key: InteractionKey = None) -> np.ndarray:
        """All raw 3D data points recorded for this InteractionKey (not binned). Array of shape (n, 4) where
        the second axis is (value, x, y, z)."""
        return self._getData(DataType.DATA_POINT, key)

    def getSegments(self, key: InteractionKey = None) -> np.ndarray:
        return self._getData(DataType.SEGMENT, key)

    def _getData(
        self, dataType: DataType, key: InteractionKey = None, transform: TransformFn = _noTransform
    ) -> Optional[np.ndarray]:
        if key and key.solidLabel:
            if not self._keyExists(key):
                return None
            container: ListArrayContainer = getattr(self._data[key], dataType.value)
            return transform(key, container.getData())
        else:
            container = ListArrayContainer()
            for interactionData in self._data.values():
                points: Optional[ListArrayContainer] = getattr(interactionData, dataType.value)
                if points is None:
                    continue
                container.extend(points)
            if len(container) == 0:
                return None
            return container.getData()

    def _keyExists(self, key: InteractionKey) -> bool:
        if key.solidLabel not in self.getStoredSolidLabels():
            warnings.warn(
                f"No data stored for solid labeled '{key.solidLabel}'. Available: {self.getStoredSolidLabels()}. "
            )
        elif key.surfaceLabel and key.surfaceLabel not in self.getStoredSurfaceLabels(key.solidLabel):
            warnings.warn(
                f"No data stored for surface labeled '{key.surfaceLabel}' for solid '{key.solidLabel}'. "
                f"Available: {self.getStoredSurfaceLabels(key.solidLabel)}. "
            )
        if key in self._data:
            return True
        return False

    def save(self, filepath: str = None):
        if filepath is None and self._filepath is None:
            filepath = self.DEFAULT_LOGGER_PATH
            warnings.warn(f"No filepath specified. Saving to {filepath}.")
        elif filepath is None:
            filepath = self._filepath

        with open(filepath, "wb") as file:
            pickle.dump((self._data, self.info, self._labels), file)

    def load(self, filepath: str):
        self._filepath = filepath

        if not os.path.exists(filepath):
            warnings.warn(
                "No logger file found at '{}'. No data loaded, but it will create a new file "
                "at this location if the logger is saved later on.".format(filepath)
            )
            return

        with open(filepath, "rb") as file:
            self._data, self.info, self._labels = pickle.load(file)

    @property
    def hasFilePath(self):
        return self._filepath is not None

    @property
    def nDataPoints(self) -> int:
        return sum(len(data.dataPoints) for data in self._data.values())
