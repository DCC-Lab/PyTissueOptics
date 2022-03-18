from typing import List
import sys
from itertools import permutations
from dataclasses import dataclass
from pytissueoptics.scene.geometry import Polygon, BoundingBox, Quad, Vector
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.intersection.quadIntersect import MollerTrumboreQuadIntersect

from pytissueoptics.scene.tree.treeConstructor import TreeConstructor, SplitNodeResult


@dataclass
class SAHSearchResult:
    leftPolygons: List[Polygon]
    rightPolygons: List[Polygon]
    toSplitPolygons: List[Polygon]
    leftBbox: BoundingBox
    rightBbox: BoundingBox
    nLeft: int
    nRight: int
    leftSAH: float
    rightSAH: float
    splitAxis : str
    splitValue: float


class FastBinaryTreeConstructor(TreeConstructor):
    def __init__(self):
        self._traversalCost = 100
        self._intersectionCost = 50
        self._reductionFactor = 0.9
        self._fallBackPercentage = 0.01
        self._nbOfPlanes = 20
        self._quadIntersector = MollerTrumboreQuadIntersect()

    def _splitNode(self, node: Node) -> SplitNodeResult:
        nodeBbox = node.bbox
        nodePolygons = node.polygons
        polygonsBbox = nodeBbox.copy()
        polygonsBbox.shrinkTo(polygonsBbox)
        SAHResult = self._searchMinSAH(polygonsBbox, nodePolygons, self._nbOfPlanes)
        if self._checkIfWorthNodeSplit(nodeBbox.getArea(), SAHResult):
            splitPlane = self._makeSplitPlane(nodeBbox, SAHResult.splitAxis, SAHResult.splitValue)
            goingLeft, goingRight = self._splitPolygons(SAHResult.toSplitPolygons, splitPlane, SAHResult.splitAxis, SAHResult.splitValue)
            SAHResult.leftPolygons.extend(goingLeft)
            SAHResult.rightPolygons.extend(goingRight)
            return SplitNodeResult(False, "", 0, [SAHResult.leftBbox, SAHResult.rightBbox], [SAHResult.leftPolygons, SAHResult.rightPolygons])
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
                splitValue = aMin + i * step
                left, right, both = self._classifyPolygons(splitValue, splitAxis, polygons)
                leftBbox = bbox.copy()
                leftBbox.update(splitAxis, "max", splitValue)
                rightBbox = bbox.copy()
                rightBbox.update(splitAxis, "min", splitValue)
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
                                                rightSAH, splitAxis, splitValue)
        return SAHresult

    def _splitPolygons(self, polygonsToSplit: List[Polygon], plane: Quad, splitAxis, splitValue):
        left = []
        right = []
        for polygon in polygonsToSplit:
            polygonRays = self._getPolygonAsRays(polygon)
            newVertices = []

            for ray in polygonRays:
                newVertices.append(ray.origin)
                intersectionPoint = self._quadIntersector.getIntersection(ray, plane)
                if intersectionPoint:
                    newVertices.append(intersectionPoint)

            newPolygons = []
            for i in range(len(newVertices) - 2):
                newPolygons.append(Polygon(vertices=[newVertices[0], newVertices[i + 1], newVertices[i + 2]]))

            for newPolygon in newPolygons:
                if splitAxis == "x":
                    if newPolygon.getCentroid().x < splitValue:
                        left.append(newPolygon)
                    else:
                        right.append(newPolygon)
                elif splitAxis == "y":
                    if newPolygon.getCentroid().y < splitValue:
                        left.append(newPolygon)
                    else:
                        right.append(newPolygon)
                else:
                    if newPolygon.getCentroid().z < splitValue:
                        left.append(newPolygon)
                    else:
                        right.append(newPolygon)
        return left, right

    @staticmethod
    def _makeSplitPlane(bbox: BoundingBox, splitAxis: str, splitValue: float) -> Quad:
        axes = ["x", "y", "z"]
        getLimitAxes = axes
        futureVertices = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        splitAxisIndex = axes.index(splitAxis)
        for futureVertex in futureVertices:
            futureVertex[splitAxisIndex] = splitValue

        getLimitAxes.remove(splitAxis)
        firstLimitIndex = axes.index(getLimitAxes[0])
        secondLimitIndex = axes.index(getLimitAxes[1])
        limitsFirstAxis = bbox.getAxisLimits(getLimitAxes[0])
        limitsSecondAxis = bbox.getAxisLimits(getLimitAxes[1])
        limitCombinaisons = [[limitsFirstAxis[0], limitsSecondAxis[0]],
                             [limitsFirstAxis[0], limitsSecondAxis[1]],
                             [limitsFirstAxis[1], limitsSecondAxis[0]],
                             [limitsFirstAxis[1], limitsSecondAxis[1]]]
        for i, combinaison in enumerate(limitCombinaisons):
            futureVertices[i][firstLimitIndex] = combinaison[0]
            futureVertices[i][secondLimitIndex] = combinaison[1]

        vectors = [Vector(*vertex) for vertex in futureVertices]
        return Quad(*vectors)


    @staticmethod
    def _getPolygonAsRays(polygon):
        polygonRays = []
        for i, vertex in enumerate(polygon.vertices):
            if i == len(polygon.vertices) - 1:
                nextVertex = polygon.vertices[0]
            else:
                nextVertex = polygon.vertices[i + 1]
            direction = nextVertex - vertex
            polygonRays.append(Ray(vertex, direction, direction.getNorm()))
        return polygonRays

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
