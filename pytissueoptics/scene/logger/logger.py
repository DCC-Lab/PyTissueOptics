from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

from pytissueoptics.scene.geometry import Vector


@dataclass
class DataPoint:
    value: float
    position: Vector


@dataclass
class Segment:
    start: Vector
    end: Vector


@dataclass(frozen=True)
class InteractionKey:
    solidLabel: str
    surfaceLabel: str = None


class InteractionData:
    def __init__(self):
        self.points: List[Vector] = []
        self.dataPoints: List[DataPoint] = []
        self.segments: List[Segment] = []


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
        self._validateKey(key)
        self._data[key].points.append(point)

    def logDataPoint(self, value: float, position: Vector, key: InteractionKey):
        self._validateKey(key)
        self._data[key].dataPoints.append(DataPoint(value, position))

    def logSegment(self, start: Vector, end: Vector, key: InteractionKey):
        self._validateKey(key)
        self._data[key].segments.append(Segment(start, end))

    def _validateKey(self, key: InteractionKey):
        if key not in self._data:
            self._data[key] = InteractionData()

    def _assertKeyExists(self, key: InteractionKey):
        if key.solidLabel not in self.getSolidLabels():
            raise KeyError(f"Invalid solid label '{key.solidLabel}'. Available: {self.getSolidLabels()}. ")
        if key.surfaceLabel and key.surfaceLabel not in self.getSurfaceLabels(key.solidLabel):
            raise KeyError(f"Invalid surface label '{key.surfaceLabel}' for solid '{key.solidLabel}'. "
                           f"Available: {self.getSurfaceLabels(key.solidLabel)}. ")

    def getPoints(self, key: InteractionKey = None):
        return self._getData(DataType.POINT, key)

    def getDataPoints(self, key: InteractionKey = None):
        return self._getData(DataType.DATA_POINT, key)

    def getSegments(self, key: InteractionKey = None):
        return self._getData(DataType.SEGMENT, key)

    def _getData(self, dataType: DataType, key: InteractionKey = None):
        if key and key.solidLabel:
            self._assertKeyExists(key)
            return getattr(self._data[key], dataType.value)
        else:
            data = []
            for interactionData in self._data.values():
                data.extend(getattr(interactionData, dataType.value))
            return data
