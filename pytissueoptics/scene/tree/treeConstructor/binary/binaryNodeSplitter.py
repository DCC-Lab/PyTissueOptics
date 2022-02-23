from typing import Tuple, List

from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree.treeConstructor.utils import meanCentroid
from pytissueoptics.scene.tree.treeConstructor import NodeSplitter, SplitNodeResult


class MeanCentroidNodeSplitter(NodeSplitter):
    def split(self, splitAxis: str, nodeBbox: BoundingBox, polygons: List[Polygon]) -> SplitNodeResult:
        splitLine = meanCentroid(splitAxis, polygons)
        polygonGroups = self._polygonCounter.count(splitLine, splitAxis, polygons)
        groupBbox = self._getNewChildrenBbox(nodeBbox, splitAxis, splitLine)
        return SplitNodeResult(False, splitAxis, splitLine, groupBbox, polygonGroups)


class MiddlePolygonSpanNodeSplitter(NodeSplitter):
    def split(self, splitAxis: str, nodeBbox: BoundingBox, polygons: List[Polygon]) -> SplitNodeResult:
        minLimit = 0
        maxLimit = 0
        for polygon in polygons:
            minLimit, maxLimit = self._compareMinMax(splitAxis, polygon, minLimit, maxLimit)
        splitLine = (minLimit + maxLimit) / 2

        polygonGroups = self._polygonCounter.count(splitLine, splitAxis, polygons)
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
    def split(self, splitAxis: str, nodeBbox: BoundingBox, polygons: List[Polygon]) -> SplitNodeResult:
        nbOfSplitPlanes = self.kwargs["nbOfSplitPlanes"]
        aMin, aMax = nodeBbox.getAxisLimits(splitAxis)
        step = nodeBbox.getAxisWidth(splitAxis) / (nbOfSplitPlanes + 1)

        nodeSAH = nodeBbox.getArea() * len(polygons)
        splitLine, minSAH = self._searchMinSAH(nodeBbox, polygons, splitAxis, aMin, step, nbOfSplitPlanes)
        splitCost = self.kwargs["splitCostPercentage"] * nodeSAH

        if minSAH + splitCost < nodeSAH:
            stopCondition = False

        else:
            stopCondition = True

        polygonGroups = self._polygonCounter.count(splitLine, splitAxis, polygons)
        groupBbox = self._getNewChildrenBbox(nodeBbox, splitAxis, splitLine)
        return SplitNodeResult(stopCondition, splitAxis, splitLine, groupBbox, polygonGroups)

    def _searchMinSAH(self, nodeBbox, polygons, splitAxis, aMin, step, nbOfSplitPlanes):
        lowestIndexSAH = 1
        minSAH = 0
        for i in range(1, nbOfSplitPlanes + 1):
            split = aMin + i * step
            left, right = self._polygonCounter.count(split, splitAxis, polygons)
            tempLeftBbox = nodeBbox.copy()
            tempLeftBbox.update(splitAxis, "max", split)
            tempRightBbox = nodeBbox.copy()
            tempRightBbox.update(splitAxis, "min", split)
            newSAH = len(left) * tempLeftBbox.getArea() + len(right) * tempRightBbox.getArea()

            if i == 1:
                minSAH = newSAH

            if newSAH < minSAH:
                minSAH = newSAH
                lowestIndexSAH = i

        splitLine = aMin + lowestIndexSAH * step
        return splitLine, minSAH
