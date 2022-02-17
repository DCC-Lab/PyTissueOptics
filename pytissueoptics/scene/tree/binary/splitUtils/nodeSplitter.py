from typing import Tuple, List
from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree.utils import meanCentroid


class NodeSplitter:
    def __init__(self, polyCounter):
        self._splitAxis = None
        self._polygons = None
        self._nodeBbox = None
        self._splitLine = None
        self._goingLeft = []
        self._goingRight = []
        self._stopCondition = 0
        self._polyCounter = polyCounter

    def run(self, splitAxis: str, polygons: List[Polygon], nodeBbox: BoundingBox) -> Tuple[int, float, List[Polygon], List[Polygon]]:
        self._splitAxis = splitAxis
        self._polygons = polygons
        self._nodeBbox = nodeBbox
        self._run()
        return self._stopCondition, self._splitLine, self._goingLeft, self._goingRight

    def _run(self):
        raise NotImplementedError


class CentroidNodeSplitter(NodeSplitter):
    def _run(self):
        self._splitLine = meanCentroid(self._splitAxis, self._polygons)
        self._goingLeft, self._goingRight = self._polyCounter.run(self._splitLine, self._splitAxis, self._polygons)


class BinaryNodeSplitter(NodeSplitter):
    def _run(self):
        minLimit = 0
        maxLimit = 0
        for polygon in self._polygons:
            minLimit, maxLimit = self._compareMinMax(self._splitAxis, polygon, minLimit, maxLimit)
        self._splitLine = (minLimit + maxLimit) / 2
        self._goingLeft, self._goingRight = self._polyCounter.run(self._splitLine, self._splitAxis, self._polygons)

    @staticmethod
    def _compareMinMax(splitAxis: str, polygon: Polygon, minLimit: float, maxLimit: float) -> Tuple:
        polyMin = polygon.bbox.getAxisLimit(splitAxis, "min")
        polyMax = polygon.bbox.getAxisLimit(splitAxis, "max")
        if polyMin < minLimit:
            minLimit = polyMin
        if polyMax > maxLimit:
            maxLimit = polyMax
        return minLimit, maxLimit


class DumbSAHSplitter(NodeSplitter):
    def _run(self):
        self._nbOfSplitPLanes = 100
        axisWidth = self._nodeBbox.getAxisWidth(self._splitAxis)
        self._aMin, self._aMax = self._nodeBbox.getAxisLimits(self._splitAxis)
        self._step = axisWidth / (self._nbOfSplitPLanes + 1)

        nodeSAH = self._nodeBbox.getArea() * len(self._polygons)
        minSAH = self._searchMinSAH()
        splitCost = 0

        if minSAH + splitCost < nodeSAH:
            self._stopCondition = 0

        else:
            self._stopCondition = 1

        self._goingLeft, self._goingRight = self._polyCounter.run(self._splitLine, self._splitAxis, self._polygons)

    def _searchMinSAH(self) -> float:
        lowestIndexSAH = 1
        minSAH = 0
        for i in range(1, self._nbOfSplitPLanes+1):
            split = self._aMin + i * self._step
            left, right = self._polyCounter.run(split, self._splitAxis, self._polygons)
            tempLeftBbox = self._nodeBbox.changeToNew(self._splitAxis, "max", split)
            tempRightBbox = self._nodeBbox.changeToNew(self._splitAxis, "min", split)
            newSAH = len(left) * tempLeftBbox.getArea() + len(right) * tempRightBbox.getArea()

            if i == 1:
                minSAH = newSAH

            if newSAH < minSAH:
                minSAH = newSAH
                lowestIndexSAH = i

        self._splitLine = self._aMin + lowestIndexSAH * self._step
        return minSAH
