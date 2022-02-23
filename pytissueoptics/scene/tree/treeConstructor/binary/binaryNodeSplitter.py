from typing import Tuple, List

from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree.treeConstructor.utils import meanCentroid
from pytissueoptics.scene.tree.treeConstructor import NodeSplitter, SplitNodeResult


class MeanCentroidNodeSplitter(NodeSplitter):
    def run(self, splitAxis: str, nodeBbox: BoundingBox, polygons: List[Polygon]) -> SplitNodeResult:
        splitLine = meanCentroid(splitAxis, polygons)
        polygonGroups = self._polyCounter.run(splitLine, splitAxis, polygons)
        groupBbox = self._getNewChildrenBbox(nodeBbox, splitAxis, splitLine)
        return SplitNodeResult(False, splitAxis, splitLine, groupBbox, polygonGroups)


class MiddlePolygonSpanNodeSplitter(NodeSplitter):
    def run(self, splitAxis: str, nodeBbox: BoundingBox, polygons: List[Polygon]) -> SplitNodeResult:
        minLimit = 0
        maxLimit = 0
        for polygon in polygons:
            minLimit, maxLimit = self._compareMinMax(splitAxis, polygon, minLimit, maxLimit)
        splitLine = (minLimit + maxLimit) / 2

        polygonGroups = self._polyCounter.run(splitLine, splitAxis, polygons)
        groupBbox = self._getNewChildrenBbox(nodeBbox, splitAxis, splitLine)
        return SplitNodeResult(False, splitAxis, splitLine, groupBbox, polygonGroups)

    @staticmethod
    def _compareMinMax(splitAxis: str, polygon: Polygon, minLimit: float, maxLimit: float) -> Tuple:
        polyMin = polygon.bbox.getAxisLimit(splitAxis, "min")
        polyMax = polygon.bbox.getAxisLimit(splitAxis, "max")
        if polyMin < minLimit:
            minLimit = polyMin
        if polyMax > maxLimit:
            maxLimit = polyMax
        return minLimit, maxLimit


class HardSAHNodeSplitter(NodeSplitter):
    def run(self, splitAxis: str, nodeBbox: BoundingBox, polygons: List[Polygon]) -> SplitNodeResult:
        self._nodeBbox = nodeBbox
        self._splitAxis = splitAxis
        self._polygons = polygons
        self._splitLine = 0
        self._minSAH = 0
        self._nbOfSplitPlanes = self.kwargs["nbOfSplitPlanes"]
        self._aMin, self._aMax = self._nodeBbox.getAxisLimits(self._splitAxis)
        self._step = self._nodeBbox.getAxisWidth(self._splitAxis) / (self._nbOfSplitPlanes + 1)

        nodeSAH = self._nodeBbox.getArea() * len(polygons)
        self._searchMinSAH()
        splitCost = self.kwargs["splitCostPercentage"] * nodeSAH

        if self._minSAH + splitCost < nodeSAH:
            self._stopCondition = False

        else:
            self._stopCondition = True

        polygonGroups = self._polyCounter.run(self._splitLine, splitAxis, polygons)
        groupBbox = self._getNewChildrenBbox(nodeBbox, splitAxis, self._splitLine)
        return SplitNodeResult(self._stopCondition, self._splitAxis, self._splitLine, groupBbox, polygonGroups)

    def _searchMinSAH(self):
        lowestIndexSAH = 1
        minSAH = 0
        for i in range(1, self._nbOfSplitPlanes + 1):
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
        self._minSAH = minSAH
