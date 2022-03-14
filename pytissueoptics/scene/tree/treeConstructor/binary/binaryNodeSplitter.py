from typing import Tuple, List

from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree.treeConstructor.utils import meanCentroid, getPolygonsBbox
from pytissueoptics.scene.tree.treeConstructor import NodeSplitter, SplitNodeResult, PolygonCounter


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


class SAHNodeSplitter(NodeSplitter):
    def __init__(self, polygonCounter: PolygonCounter = None,
                 nbOfSplitPlanes: int = 50, splitCostPercentage: float = 0.1):
        super().__init__(polygonCounter)
        self._nbOfSplitPlanes = nbOfSplitPlanes
        self._splitCostPercentage = splitCostPercentage
        self._polygonCounter = polygonCounter

    def split(self, splitAxis: str, nodeBbox: BoundingBox, polygons: List[Polygon]) -> SplitNodeResult:
        raise NotImplementedError

    def _compareSAHAndReturnSplitNode(self, splitAxis: str, splitLine: float, bboxToSplit: BoundingBox, polygons: List[Polygon], nodeSAH: float, minSAH: float):
        splitCost = self._splitCostPercentage * nodeSAH

        if minSAH + splitCost < nodeSAH:
            stopCondition = False

        else:
            stopCondition = True

        polygonGroups = self._polygonCounter.count(splitLine, splitAxis, polygons)
        groupBbox = self._getNewChildrenBbox(bboxToSplit, splitAxis, splitLine)
        return SplitNodeResult(stopCondition, splitAxis, splitLine, groupBbox, polygonGroups)

    def _searchMinSAH(self, nodeBbox, polygons, splitAxis, aMin, step):
        lowestIndexSAH = 1
        minSAH = 0
        for i in range(0, self._nbOfSplitPlanes):
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


class HardSAHNodeSplitter(SAHNodeSplitter):
    def __init__(self, polygonCounter: PolygonCounter = None,
                 nbOfSplitPlanes: int = 20, splitCostPercentage: float = 0.1):
        super().__init__(polygonCounter)
        self._nbOfSplitPlanes = nbOfSplitPlanes
        self._splitCostPercentage = splitCostPercentage

    def split(self, splitAxis: str, nodeBbox: BoundingBox, polygons: List[Polygon]) -> SplitNodeResult:
        aMin, aMax = nodeBbox.getAxisLimits(splitAxis)
        step = nodeBbox.getAxisWidth(splitAxis) / (self._nbOfSplitPlanes + 1)
        nodeSAH = nodeBbox.getArea() * len(polygons)
        splitLine, minSAH = self._searchMinSAH(nodeBbox, polygons, splitAxis, aMin, step)
        return self._compareSAHAndReturnSplitNode(splitAxis, splitLine, nodeBbox, polygons, nodeSAH, minSAH)


class ShrankBoxSAHNodeSplitter(SAHNodeSplitter):
    def __init__(self, polygonCounter: PolygonCounter = None,
                 nbOfSplitPlanes: int = 20, splitCostPercentage: float = 0.1):
        super().__init__(polygonCounter)
        self._nbOfSplitPlanes = nbOfSplitPlanes
        self._splitCostPercentage = splitCostPercentage

    def split(self, splitAxis: str, nodeBbox: BoundingBox, polygons: List[Polygon]) -> SplitNodeResult:
        polygonsBbox = getPolygonsBbox(polygons)
        newBounds = nodeBbox.copy()
        newBounds.shrinkTo(polygonsBbox)
        newBounds.update(splitAxis, "min", newBounds.getAxisLimit(splitAxis, "min")-0.1)
        newBounds.update(splitAxis, "max", newBounds.getAxisLimit(splitAxis, "max")+0.1)
        aMin, aMax = newBounds.getAxisLimits(splitAxis)
        step = newBounds.getAxisWidth(splitAxis) / (self._nbOfSplitPlanes + 1)
        nodeSAH = nodeBbox.getArea() * len(polygons)
        splitLine, minSAH = self._searchMinSAH(newBounds, polygons, splitAxis, aMin, step)
        return self._compareSAHAndReturnSplitNode(splitAxis, splitLine, newBounds, polygons, nodeSAH, minSAH)
