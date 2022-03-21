from typing import List, Tuple
from dataclasses import dataclass
from math import isclose
import sys

from pytissueoptics.scene.geometry import Polygon, BoundingBox, Vector
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.intersection import Ray
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
    splitAxis: str
    splitValue: float


class FastBinaryTreeConstructor(TreeConstructor):
    """
    This is an implementation of the proposed algorithms found in

    Wald, Ingo, and Vlastimil Havran. 2006. “On Building Fast Kd-Trees for Ray Tracing, and on Doing That in O(N Log N).
    ” In 2006 IEEE Symposium on Interactive Ray Tracing, 61–69.

    Soupikov, Alexei, Maxim Shevtsov, and Alexander Kapustin. 2008. “Improving Kd-Tree Quality at a Reasonable
    Construction Cost.” In Interactive Ray Tracing, 2008. RT 2008. IEEE Symposium on, 67–72. unknown.

    -

    This algorithm uses a
    1. Surface Area Heuristic (SAH)
    2. Perfect Splits
    3. Minimum SAH search on 3 dimensions

    When the minimum SAH axis and value are found, a split cost is calculated.
    If the children's cost is lower than the parent, splitting is allowed, else, no splitting occurs.

    """
    def __init__(self):
        super(FastBinaryTreeConstructor, self).__init__()
        self._traversalCost = 8
        self._intersectionCost = 1
        self._reductionFactor = 0.8
        self._nbOfPlanes = 20

    def _splitNode(self, node: Node) -> SplitNodeResult:
        nodeBbox = node.bbox
        nodePolygons = node.polygons
        polygonsBbox = BoundingBox.fromPolygons(nodePolygons)
        SAHResult = self._searchMinSAH(polygonsBbox, nodePolygons, self._nbOfPlanes)
        if self._checkIfWorthNodeSplit(nodeBbox.getArea(), SAHResult):
            normal, point = self._makeSplitPlane(SAHResult.splitAxis, SAHResult.splitValue)
            goingLeft, goingRight = self._splitPolygons(SAHResult.toSplitPolygons, normal, point,
                                                        SAHResult.splitAxis, SAHResult.splitValue)
            SAHResult.leftPolygons.extend(goingLeft)
            SAHResult.rightPolygons.extend(goingRight)
            return SplitNodeResult(False, "", 0, [SAHResult.leftBbox, SAHResult.rightBbox],
                                   [SAHResult.leftPolygons, SAHResult.rightPolygons])
        else:
            return SplitNodeResult(True, None, None, None, None)

    def _checkIfWorthNodeSplit(self, nodeSA: float, SAHResult: SAHSearchResult):
        newNodeIntersectionCost = self._intersectionCost * ((SAHResult.leftSAH + SAHResult.rightSAH) / nodeSA)
        approximatedSplitCost = self._traversalCost + newNodeIntersectionCost
        currentTraversalCost = self._intersectionCost * (SAHResult.nLeft + SAHResult.nRight)
        if approximatedSplitCost < currentTraversalCost:
            return True
        return False

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

    def _splitPolygons(self, polygonsToSplit: List[Polygon], planeNormal: Vector, planePoint: Vector, splitAxis, splitValue):
        left = []
        right = []
        for polygon in polygonsToSplit:
            polygonRays = self._getPolygonAsRays(polygon)
            intersectionPoints = []
            leftVertices = []
            rightVertices = []
            for ray in polygonRays:
                if splitAxis == "x":
                    rayValue = ray.origin.x
                elif splitAxis == "y":
                    rayValue = ray.origin.y
                else:
                    rayValue = ray.origin.z

                if isclose(rayValue, splitValue, abs_tol=1e-6):
                    intersectionPoints.append(ray.origin)
                elif rayValue < splitValue:
                    leftVertices.append(ray.origin)
                else:
                    rightVertices.append(ray.origin)
                intersectionPoint = self._intersectPlaneWithRay(planeNormal, planePoint, ray)
                if intersectionPoint:
                    intersectionPoints.append(intersectionPoint)

            if leftVertices:
                leftVertices.extend(intersectionPoints)
                left.append(Polygon(vertices=leftVertices))
            if rightVertices:
                rightVertices.extend(intersectionPoints)
                right.append(Polygon(vertices=rightVertices))
        return left, right

    @staticmethod
    def _intersectPlaneWithRay(normal: Vector, planePoint: Vector, ray: Ray, tol=1e-6):
        """
        algorithm from scratchpixel.com
        1. normal.dot(direction), to check if plane and are are coplanar
        2. the dot product of two perpendicular vectors is equal to 0  | (normal.dot(planePoint - origin)) = 0
        3. we parametrize the ray equation as | origin + direction * t = planePoint
        4. we solve for t.
        5. in our case, the hit point has to be within the predefined polygon, so we verify 't' with the ray.length
        """
        coplanar = normal.dot(ray.direction)
        if abs(coplanar) > tol:
            inPlane = planePoint - ray.origin
            t = inPlane.dot(normal) / coplanar
            hit = ray.origin + ray.direction * t
            if (hit - ray.origin).getNorm() <= ray.length:
                return hit
        return None

    @staticmethod
    def _makeSplitPlane(splitAxis: str, splitValue: float) -> Tuple[Vector, Vector]:
        if splitAxis == "x":
            normal = Vector(1, 0, 0)
            planePoint = Vector(splitValue, 0, 0)
            return normal, planePoint
        elif splitAxis == "y":
            normal = Vector(0, 1, 0)
            planePoint = Vector(0, splitValue, 0)
            return normal, planePoint
        elif splitAxis == "z":
            normal = Vector(0, 0, 1)
            planePoint = Vector(0, 0, splitValue)
            return normal, planePoint

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
