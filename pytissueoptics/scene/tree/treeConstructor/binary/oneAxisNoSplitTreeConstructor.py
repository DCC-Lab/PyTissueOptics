from typing import List
import sys

from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor import TreeConstructor, SplitNodeResult
from pytissueoptics.scene.tree.treeConstructor.binary import SAHSearchResult


class OneAxisNoSplitTreeConstructor(TreeConstructor):
    def __init__(self, nbOfSplitPlanes: int = 20, splitCostPercentage: float = 0.2, fallback: float = 1e-6):
        super().__init__()
        self._nbOfSplitPlanes = nbOfSplitPlanes
        self._splitCostPercentage = splitCostPercentage
        self._fallback = fallback
        self._fallBackPercentage = 0.01

    def _splitNode(self, node: Node) -> SplitNodeResult:
        nodeBbox = node.bbox
        polygons = node.polygons
        polygonsBbox = BoundingBox.fromPolygons(polygons)
        testBbox = nodeBbox.copy()
        testBbox.shrinkTo(polygonsBbox)
        testBbox.extendBy(self._fallback)
        result = self._searchMinSAH(testBbox, polygons)
        stopCondition = self._checkStopCondition(nodeBbox, polygons, result)
        return SplitNodeResult(stopCondition, [result.lBbox, result.rBbox], [result.lPolygons, result.rPolygons])

    def _checkStopCondition(self, nodeBbox: BoundingBox,  polygons: List[Polygon], result: SAHSearchResult):
        nodeSAH = nodeBbox.getArea() * len(polygons)
        splitCost = self._splitCostPercentage * nodeSAH
        if result.totalSAH + splitCost < nodeSAH:
            return False
        return True

    def _searchMinSAH(self, splitBbox, polygons):
        result = None
        minSAH = sys.maxsize
        splitAxis = self._selectAxis(polygons)
        aMin, aMax = splitBbox.getAxisLimits(splitAxis)
        step = splitBbox.getAxisWidth(splitAxis) / (self._nbOfSplitPlanes + 1)
        for i in range(1, self._nbOfSplitPlanes + 1):
            splitValue = aMin + i * step
            left, right = self._classifyFromBbox(polygons, splitValue, splitAxis)
            tempLeftBbox = splitBbox.copy()
            tempLeftBbox.update(splitAxis, "max", splitValue)
            tempRightBbox = splitBbox.copy()
            tempRightBbox.update(splitAxis, "min", splitValue)
            leftSAH = len(left) * tempLeftBbox.getArea()
            rightSAH = len(right) * tempRightBbox.getArea()
            newSAH = leftSAH + rightSAH
            if newSAH < minSAH:
                minSAH = newSAH
                result = SAHSearchResult(left, right, [], tempLeftBbox, tempRightBbox, len(left), len(right),
                                         leftSAH, rightSAH, leftSAH+rightSAH, splitAxis, splitValue)
        return result

    @staticmethod
    def _classifyFromBbox(polygons, splitValue, splitAxis):
        goingLeft = []
        goingRight = []
        for polygon in polygons:
            if polygon.bbox.getAxisLimit(splitAxis, "min") < splitValue and \
                    polygon.bbox.getAxisLimit(splitAxis, "max") < splitValue:
                goingLeft.append(polygon)
            elif polygon.bbox.getAxisLimit(splitAxis, "min") > splitValue and \
                    polygon.bbox.getAxisLimit(splitAxis, "max") > splitValue:
                goingRight.append(polygon)
            else:
                goingLeft.append(polygon)
                goingRight.append(polygon)

        return goingLeft, goingRight

    @staticmethod
    def _selectAxis(polygons: List[Polygon]) -> str:
        bbox = BoundingBox.fromPolygons(polygons)
        widths = [bbox.xWidth, bbox.yWidth, bbox.zWidth]
        axisIndex = widths.index(max(widths))
        axes = ["x", "y", "z"]
        return axes[axisIndex]
