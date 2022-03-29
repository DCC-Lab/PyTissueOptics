from typing import List, Tuple
import sys

from pytissueoptics.scene.geometry import BoundingBox
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor import SplitNodeResult
from pytissueoptics.scene.tree.treeConstructor import TreeConstructor
from pytissueoptics.scene.tree.treeConstructor.binary import SAHSearchResult


class NoSplitOneAxisConstructor(TreeConstructor):

    def __init__(self, nbOfSplitPlanes: int = 20, intersectionCost: float = 0.5, traversalCost: float = 1,
                 noSharedBonus: float = 2, emptySpaceBonus: float = 2):
        super().__init__()
        self._nbOfSplitPlanes = nbOfSplitPlanes
        self._noSharedBonus = noSharedBonus
        self._emptySpaceBonus = emptySpaceBonus
        self._traversalCost = traversalCost
        self._intersectionCost = intersectionCost
        self.result = None
        self.currentNode = None

    def _splitNode(self, node: Node) -> SplitNodeResult:
        self.currentNode = node
        axis = self._selectBboxWidestAxis(self.currentNode.bbox)
        splitBbox = self.currentNode.bbox.copy()
        self._searchMinSAHOnAxis(splitBbox, axis)
        self.result.leftPolygons.extend(self.result.splitPolygons)
        self.result.rightPolygons.extend(self.result.splitPolygons)
        self._trimChildrenBbox()
        stopCondition = self._checkStopCondition()
        newNodeResult = SplitNodeResult(stopCondition, [self.result.leftBbox, self.result.rightBbox],
                                        [self.result.leftPolygons, self.result.rightPolygons])
        return newNodeResult

    def _checkStopCondition(self):
        nodeSA = self.currentNode.bbox.getArea()
        newNodeIntersectionCost = self._intersectionCost * (self.result.SAH / nodeSA)
        approximatedSplitCost = self._traversalCost + newNodeIntersectionCost
        currentTraversalCost = self._intersectionCost * len(self.currentNode.polygons)
        if approximatedSplitCost < currentTraversalCost:
            return False
        return True

    def _searchMinSAHOnAxis(self, splitBbox: BoundingBox, splitAxis: str = "x", minSAH: float = sys.float_info.max):
        minSAH = minSAH
        aMin, aMax = splitBbox.getAxisLimits(splitAxis)
        step = splitBbox.getAxisWidth(splitAxis) / (self._nbOfSplitPlanes + 1)
        for i in range(1, self._nbOfSplitPlanes):
            splitValue = aMin + i * step
            left, right, both = self._classifyNodePolygons(splitAxis, splitValue)
            nLeft = len(left)
            nRight = len(right)
            nBoth = len(both)
            tempLeftBbox = splitBbox.copy()
            tempLeftBbox.update(splitAxis, "max", splitValue)
            tempRightBbox = splitBbox.copy()
            tempRightBbox.update(splitAxis, "min", splitValue)
            leftSAH = (nLeft + nBoth) * tempLeftBbox.getArea()
            rightSAH = (nRight + nBoth) * tempRightBbox.getArea()
            newSAH = leftSAH + rightSAH
            if nBoth == 0 and nLeft != 0 and nRight != 0:
                newSAH /= self._noSharedBonus
            if (nLeft == 0 or nRight == 0) and nBoth == 0:
                newSAH /= self._emptySpaceBonus
            if newSAH < minSAH:
                minSAH = newSAH
                self.result = SAHSearchResult(left, right, both, tempLeftBbox, tempRightBbox, splitAxis, splitValue)
        return minSAH

    def _trimChildrenBbox(self):
        lBBox = BoundingBox.fromPolygons(self.result.leftPolygons)
        rBbox = BoundingBox.fromPolygons(self.result.rightPolygons)
        if lBBox:
            self.result.leftBbox.shrinkTo(lBBox)
        if rBbox:
            self.result.rightBbox.shrinkTo(rBbox)

    def _classifyNodePolygons(self, splitAxis: str, splitValue: float) -> Tuple[List, List, List]:
        goingLeft = []
        goingRight = []
        goingBoth = []
        for polygon in self.currentNode.polygons:
            minLimit = polygon.bbox.getAxisLimit(splitAxis, "min")
            maxLimit = polygon.bbox.getAxisLimit(splitAxis, "max")
            if minLimit < splitValue and maxLimit <= splitValue:
                goingLeft.append(polygon)
            elif minLimit >= splitValue and maxLimit > splitValue:
                goingRight.append(polygon)
            else:
                goingBoth.append(polygon)
        return goingLeft, goingRight, goingBoth

    @staticmethod
    def _selectBboxWidestAxis(nodeBbox: BoundingBox) -> str:
        widths = [nodeBbox.xWidth, nodeBbox.yWidth, nodeBbox.zWidth]
        axisIndex = widths.index(max(widths))
        axes = ["x", "y", "z"]
        return axes[axisIndex]
