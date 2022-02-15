from typing import List, Tuple
from pytissueoptics.scene.geometry import Polygon

class Splitter:
    def calculateSplitLine(self, polygons: List[Polygon], nodeAxis: str) -> float:
        raise NotImplementedError


class MeanCentroidSplitter(Splitter):
    def calculateSplitLine(self, polygons: List[Polygon], nodeAxis: str) -> float:
        average = 0
        for polygon in polygons:
            if nodeAxis == "x":
                average += polygon.centroid.x
            elif nodeAxis == "y":
                average += polygon.centroid.y
            elif nodeAxis == "z":
                average += polygon.centroid.z
        average = average / len(polygons)
        return average


class BinarySplitter(Splitter):
    def calculateSplitLine(self, polygons: List[Polygon], nodeAxis: str) -> float:
        minLimit = 0
        maxLimit = 0
        for polygon in polygons:
            minLimit, maxLimit = self._compareMinMax(nodeAxis, polygon, minLimit, maxLimit)
        return (minLimit + maxLimit) / 2

    @staticmethod
    def _compareMinMax(nodeAxis: str, polygon: Polygon, minLimit: float, maxLimit: float) -> Tuple:
        polyMin = polygon.bbox.getAxisLimit(nodeAxis, "min")
        polyMax = polygon.bbox.getAxisLimit(nodeAxis, "max")
        if polyMin < minLimit:
            minLimit = polyMin
        if polyMax > maxLimit:
            maxLimit = polyMax
        return minLimit, maxLimit
