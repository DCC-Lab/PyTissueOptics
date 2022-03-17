from typing import List
import sys
from dataclasses import dataclass
from pytissueoptics.scene.geometry import Polygon, BoundingBox
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor import TreeConstructor, SplitNodeResult


@dataclass
class SAHSearchResult:
    leftTriangles: List[Polygon]
    rightTriangles: List[Polygon]
    toSplitTriangles: List[Polygon]
    leftBbox: BoundingBox
    rightBbox: BoundingBox
    nLeft: int
    nRight: int
    leftSAH: float
    rightSAH: float


class FastBinaryTreeConstructor(TreeConstructor):
    def __init__(self):
        self._traversalCost = 100
        self._intersectionCost = 50
        self._reductionFactor = 0.9
        self._fallBackPercentage = 0.01
        self._nbOfPlanes = 20

    def _splitNode(self, node: Node) -> SplitNodeResult:
        nodeBbox = node.bbox
        nodePolygons = node.polygons
        polygonsBbox = nodeBbox.copy()
        polygonsBbox.shrinkTo(polygonsBbox)
        SAHResult = self._searchMinSAH(polygonsBbox, nodePolygons, self._nbOfPlanes)
        if self._checkIfWorthNodeSplit(nodeBbox.getArea(), SAHResult):
            #self._splitTriangles(SAHResult.toSplitTriangles)
            return SplitNodeResult(False, "", 0, [SAHResult.leftBbox, SAHResult.rightBbox], [SAHResult.leftTriangles, SAHResult.rightTriangles])
        else:
            return SplitNodeResult(True, None, None, None, None)

    def _checkIfWorthNodeSplit(self, nodeSA: float, SAHResult: SAHSearchResult):
        approximatedSplitCost = self._traversalCost + self._intersectionCost * (
                    (SAHResult.leftSAH + SAHResult.rightSAH) / nodeSA)
        currentTraversalCost = self._intersectionCost * (SAHResult.nLeft + SAHResult.nRight)
        if SAHResult.nLeft == 0 or SAHResult.nRight == 0:
            approximatedSplitCost = approximatedSplitCost * self._reductionFactor
        if approximatedSplitCost < currentTraversalCost:
            nodeSplit = True
        else:
            nodeSplit = False
        return nodeSplit

    def _searchMinSAH(self, bbox, polygons, nbOfPlanes):
        SAHresult = None
        minSAH = sys.maxsize
        for splitAxis in ["x", "y", "z"]:
            aMin, aMax = bbox.getAxisLimits(splitAxis)
            step = bbox.getAxisWidth(splitAxis) / (nbOfPlanes + 1)
            for i in range(0, nbOfPlanes):
                splitLine = aMin + i * step
                left, right, both = self._classifyPolygons(splitLine, splitAxis, polygons)
                leftBbox = bbox.copy()
                leftBbox.update(splitAxis, "max", splitLine)
                rightBbox = bbox.copy()
                rightBbox.update(splitAxis, "min", splitLine)
                nLeft = len(left) + len(both)
                nRight = len(right) + len(both)
                leftSAH = nLeft * leftBbox.getArea()
                rightSAH = nRight * rightBbox.getArea()
                newSAH = leftSAH + rightSAH
                if (nLeft == 0 or nRight == 0) and len(both) == 0:
                    newSAH *= self._reductionFactor
                if newSAH < minSAH:
                    minSAH = newSAH
                    SAHresult = SAHSearchResult(left, right, both, leftBbox, rightBbox, nLeft, nRight, leftSAH,
                                                rightSAH)
        return SAHresult

    def _splitTriangles(self, toSplit):
        left = []
        right = []
        return left, right

    @staticmethod
    def _classifyPolygons(splitLine: float, splitAxis: str, polygons: List[Polygon]):
        goingLeft = []
        goingRight = []
        toBeSplit = []

        for polygon in polygons:
            limits = polygon.bbox.getAxisLimits(splitAxis)
            if limits[0] < splitLine and limits[1] < splitLine:
                goingLeft.append(polygon)
            elif limits[0] > splitLine and limits[1] > splitLine:
                goingRight.append(polygon)
            else:
                toBeSplit.append(polygon)

        return [goingLeft, goingRight, toBeSplit]

    def growTree(self, node: Node, maxDepth: int, minLeafSize: int):
        if node.depth >= maxDepth or len(node.polygons) <= minLeafSize:
            return

        splitNodeResult = self._splitNode(node)
        if splitNodeResult.stopCondition:
            return

        for i, polygonGroup in enumerate(splitNodeResult.polygonGroups):
            if len(polygonGroup) <= 0:
                continue
            childNode = Node(parent=node, polygons=polygonGroup, bbox=splitNodeResult.bboxes[i], depth=node.depth + 1)
            node.children.append(childNode)
            self.growTree(childNode, maxDepth, minLeafSize)
