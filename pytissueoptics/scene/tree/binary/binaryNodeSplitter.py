from typing import Tuple, List
from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree.splitUtils.utils import meanCentroid
from pytissueoptics.scene.tree.splitUtils import NodeSplitter
from pytissueoptics.scene.tree import SplitNodeResult


class BinaryNodeSplitter(NodeSplitter):
    def _run(self):
        raise NotImplementedError

    def _getNewChildrenBbox(self, splitLine) -> List[BoundingBox]:
        leftBbox = self._nodeBbox.changeToNew(self._splitAxis[0], "max", splitLine)
        rightBbox = self._nodeBbox.changeToNew(self._splitAxis[0], "min", splitLine)
        return [leftBbox, rightBbox]


class CentroidNodeSplitter(BinaryNodeSplitter):
    def _run(self):
        splitLine = meanCentroid(self._splitAxis[0], self._polygons)
        polygonGroups = self._polyCounter.run(splitLine, self._splitAxis[0], self._polygons)
        groupBbox = self._getNewChildrenBbox(splitLine)
        self._splitNodeResult = SplitNodeResult(False, groupBbox, polygonGroups)


class MiddleNodeSplitter(BinaryNodeSplitter):
    def _run(self):
        minLimit = 0
        maxLimit = 0
        for polygon in self._polygons:
            minLimit, maxLimit = self._compareMinMax(self._splitAxis[0], polygon, minLimit, maxLimit)
        splitLine = (minLimit + maxLimit) / 2
        groupBbox = self._getNewChildrenBbox(splitLine)
        polygonGroups = self._polyCounter.run(splitLine, self._splitAxis[0], self._polygons)
        self._splitNodeResult = SplitNodeResult(False, groupBbox, polygonGroups)

    @staticmethod
    def _compareMinMax(splitAxis: str, polygon: Polygon, minLimit: float, maxLimit: float) -> Tuple:
        polyMin = polygon.bbox.getAxisLimit(splitAxis, "min")
        polyMax = polygon.bbox.getAxisLimit(splitAxis, "max")
        if polyMin < minLimit:
            minLimit = polyMin
        if polyMax > maxLimit:
            maxLimit = polyMax
        return minLimit, maxLimit


class HardSAHSplitter(BinaryNodeSplitter):
    def _run(self):
        self._splitAxis = self._splitAxis[0]
        self._nbOfSplitPLanes = 50
        self._aMin, self._aMax = self._nodeBbox.getAxisLimits(self._splitAxis)
        self._step = self._nodeBbox.getAxisWidth(self._splitAxis) / (self._nbOfSplitPLanes + 1)

        nodeSAH = self._nodeBbox.getArea() * len(self._polygons)
        minSAH = self._searchMinSAH()
        splitCost = 0.1 * nodeSAH

        if minSAH + splitCost < nodeSAH:
            self._stopCondition = 0

        else:
            self._stopCondition = 1
            print("CockBlocked.")

        polygonGroups = self._polyCounter.run(self._splitLine, self._splitAxis, self._polygons)
        groupsBbox = self._getNewChildrenBbox(self._splitLine)
        self._splitNodeResult = SplitNodeResult(self._stopCondition, groupsBbox, polygonGroups)

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
