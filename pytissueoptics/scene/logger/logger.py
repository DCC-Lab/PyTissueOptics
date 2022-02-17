import sys
from dataclasses import dataclass
from typing import List

from pytissueoptics.scene.geometry import Vector


@dataclass
class DataPoint:
    value: float
    position: Vector


@dataclass
class Segment:
    start: Vector
    end: Vector


class Logger:
    def __init__(self):
        self._points: List[Vector] = []
        self._dataPoints: List[DataPoint] = []
        self._segments: List[Segment] = []

    @property
    def points(self):
        return self._points

    @property
    def dataPoints(self):
        return self._dataPoints

    @property
    def segments(self):
        return self._segments

    def getMinDataPoint(self) -> DataPoint:
        minDataPoint = None
        minValue = sys.maxsize
        for dataPoint in self._dataPoints:
            if dataPoint.value < minValue:
                minValue = dataPoint.value
                minDataPoint = dataPoint
        return minDataPoint

    def getMaxDataPoint(self) -> DataPoint:
        maxDataPoint = None
        maxValue = -sys.maxsize
        for dataPoint in self._dataPoints:
            if dataPoint.value > maxValue:
                maxValue = dataPoint.value
                maxDataPoint = dataPoint
        return maxDataPoint

    def logPoint(self, point: Vector):
        self._points.append(point)

    def logDataPoint(self, value: float, position: Vector):
        self._dataPoints.append(DataPoint(value, position))

    def logSegment(self, start: Vector, end: Vector):
        self._segments.append(Segment(start, end))

