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
    solidName: str
    surfaceName: str = None


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

    def getSolidNames(self) -> List[str]:
        return list(set(key.solidName for key in self._data.keys()))

    def getSurfaceNames(self, solidName: str) -> List[str]:
        return [key.surfaceName for key in self._data.keys() if key.solidName == solidName
                and key.surfaceName is not None]

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

    def getPoints(self, key: InteractionKey = None):
        return self._getData(DataType.POINT, key)

    def getDataPoints(self, key: InteractionKey = None):
        return self._getData(DataType.DATA_POINT, key)

    def getSegments(self, key: InteractionKey = None):
        return self._getData(DataType.SEGMENT, key)

    def _getData(self, dataType: DataType, key: InteractionKey = None):
        if key:
            return getattr(self._data[key], dataType.value)
        else:
            data = []
            for interactionData in self._data.values():
                data.extend(getattr(interactionData, dataType.value))
            return data
