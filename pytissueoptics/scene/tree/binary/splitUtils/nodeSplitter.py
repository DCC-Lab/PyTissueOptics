from typing import Tuple, List
from pytissueoptics.scene.geometry import Polygon, BoundingBox


class NodeSplitter:
    def __init__(self, polyCounter):
        self._splitAxis = None
        self._polygons = None
        self._nodeBbox = None
        self._polyCounter = polyCounter

    def run(self, splitAxis: str, polygons: List[Polygon], nodeBbox: BoundingBox) -> Tuple:
        self._splitAxis = splitAxis
        self._polygons = polygons
        self._nodeBbox = nodeBbox
        self._splitLine = self._run()
        self._polyCounter.run(self._splitLine)

    def _run(self) -> Tuple:
        raise NotImplementedError


class CentroidNodeSplitter(NodeSplitter):
    def _run(self):
        average = 0
        for polygon in self._polygons:
            if self._splitAxis == "x":
                average += polygon.centroid.x
            elif self._splitAxis == "y":
                average += polygon.centroid.y
            elif self._splitAxis == "z":
                average += polygon.centroid.z
        average = average / len(self._polygons)
        return average


class BinaryNodeSplitter(NodeSplitter):
    def _run(self) -> float:
        minLimit = 0
        maxLimit = 0
        for polygon in self._polygons:
            minLimit, maxLimit = self._compareMinMax(self._splitAxis, polygon, minLimit, maxLimit)
        return (minLimit + maxLimit) / 2

    @staticmethod
    def _compareMinMax(splitAxis: str, polygon: Polygon, minLimit: float, maxLimit: float) -> Tuple:
        polyMin = polygon.bbox.getAxisLimit(splitAxis, "min")
        polyMax = polygon.bbox.getAxisLimit(splitAxis, "max")
        if polyMin < minLimit:
            minLimit = polyMin
        if polyMax > maxLimit:
            maxLimit = polyMax
        return minLimit, maxLimit


# class SAHSplitter(NodeSplitter):
#     def _getSplitLine(self) -> float:
#         self._nodeSAH = self._nodeBbox.getArea() * len(self._polygons)
#         minSAH = self._searchMinSAH()
#
#     def _searchMinSAH(self) -> float:
#         split = self._centroidSplit()
#         minSAH = nodeSAH
#         fisrtSplit = True
#         change = 0
#         delta = 1
#
#         for _ in range(10):
#             if change > 0.01:
#                 left, right = self._separateLeftRight(split, polygons, nodeAxis)
#                 tempLeftBbox = nodeBbox.changeToNew(nodeAxis, "max", split)
#                 tempRightBbox = nodeBbox.changeToNew(nodeAxis, "min", split)
#                 newSAH = len(left) * tempLeftBbox.getArea() + len(right) * tempRightBbox.getArea()
#
#                 if newSAH < minSAH:
#                     minSAH = newSAH
#                     delta = delta
#                     split = split + delta
#                 else:
#                     delta *= -0.5
#                     split = split + delta
#
#     def _getNewSAH(self):
#         left, right = self._separateLeftRight(split, polygons, nodeAxis)
#         tempLeftBbox = nodeBbox.changeToNew(nodeAxis, "max", split)
#         tempRightBbox = nodeBbox.changeToNew(nodeAxis, "min", split)
#         newSAH = len(left) * tempLeftBbox.getArea() + len(right) * tempRightBbox.getArea()
#
#     def _directionOfTheEmptySide(self) -> float:
#         lSize = len(left)
#         rSize = len(right)
#         direction = 0
#
#         if lSize > rSize:
#             direction = -1
#         elif rSize > lSize:
#             direction = 1
#         elif rSize == lSize:
#
#         return direction