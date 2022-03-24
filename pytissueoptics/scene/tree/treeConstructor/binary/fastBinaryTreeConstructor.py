from typing import List, Tuple
from dataclasses import dataclass
import sys

from pytissueoptics.scene.geometry import Polygon, BoundingBox, Vector, Triangle, Quad
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
    EPSILON = 1e-6

    def __init__(self, traversalCost=1, intersectionCost=1, emptySpaceBonus=0.7, nbOfPlanes=20):
        super(FastBinaryTreeConstructor, self).__init__()
        self._traversalCost = traversalCost
        self._intersectionCost = intersectionCost
        self._emptySpaceBonus = emptySpaceBonus
        self._nbOfPlanes = nbOfPlanes

    def _splitNode(self, node: Node) -> SplitNodeResult:
        nodeBbox = node.bbox
        nodePolygons = node.polygons
        polygonsBbox = BoundingBox.fromPolygons(nodePolygons)
        SAHResult = self._searchMinSAH(polygonsBbox, nodePolygons, self._nbOfPlanes)
        if self._checkIfWorthNodeSplit(nodeBbox.getArea(), len(nodePolygons), polygonsBbox, SAHResult):
            normal, point = self._makeSplitPlane(SAHResult.splitAxis, SAHResult.splitValue)
            goingLeft, goingRight = self._splitTriangles(SAHResult.toSplitPolygons, normal, point)
            SAHResult.leftPolygons.extend(goingLeft)
            SAHResult.rightPolygons.extend(goingRight)
            return SplitNodeResult(False, "", 0, [SAHResult.leftBbox, SAHResult.rightBbox],
                                   [SAHResult.leftPolygons, SAHResult.rightPolygons])
        else:
            return SplitNodeResult(True, None, None, None, None)

    def _checkIfWorthNodeSplit(self, nodeSA: float, nbTrianglesBefore, polygonsBbox, SAHResult: SAHSearchResult):
        newNodeIntersectionCost = self._intersectionCost * ((SAHResult.leftSAH + SAHResult.rightSAH) / nodeSA)
        approximatedSplitCost = self._traversalCost + newNodeIntersectionCost
        currentTraversalCost = self._intersectionCost * nbTrianglesBefore
        if approximatedSplitCost < currentTraversalCost:
            return True
        return False

    def _searchMinSAH(self, bbox, polygons, nbOfPlanes):
        SAHresult = None
        minSAH = sys.maxsize
        for splitAxis in ["x", "y", "z"]:
            aMin, aMax = bbox.getAxisLimits(splitAxis)
            step = bbox.getAxisWidth(splitAxis) / (nbOfPlanes + 1)
            for i in range(1, nbOfPlanes):
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
                if (nLeft == 0 or nRight == 0) and len(both) == 0:
                    leftSAH *= self._emptySpaceBonus
                    rightSAH *= self._emptySpaceBonus
                newSAH = leftSAH + rightSAH
                if newSAH < minSAH:
                    minSAH = newSAH
                    SAHresult = SAHSearchResult(left, right, both, leftBbox, rightBbox, nLeft, nRight, leftSAH,
                                                rightSAH, splitAxis, splitValue)
        return SAHresult

    # def _splitPolygons(self, polygons: List[Polygon], planeNormal: Vector, planePoint: Vector, splitAxis, splitValue):
    #     left = []
    #     right = []
    #     for polygon in polygons:
    #         polygonRays = self._getPolygonAsRays(polygon)
    #         intersectionPoints = []
    #         leftVertices = []
    #         rightVertices = []
    #         for ray in polygonRays:
    #             if splitAxis == "x":
    #                 rayValue = ray.origin.x
    #             elif splitAxis == "y":
    #                 rayValue = ray.origin.y
    #             else:
    #                 rayValue = ray.origin.z
    #
    #             if isclose(rayValue, splitValue, abs_tol=1e-6):
    #                 intersectionPoints.append(ray.origin)
    #             elif rayValue < splitValue:
    #                 leftVertices.append(ray.origin)
    #             else:
    #                 rightVertices.append(ray.origin)
    #             intersectionPoint = self._intersectPlaneWithRay(planeNormal, planePoint, ray)
    #             if intersectionPoint:
    #                 intersectionPoints.append(intersectionPoint)
    #
    #         if leftVertices:
    #             leftVertices.extend(intersectionPoints)
    #             left.append(self._makePolygonFromVertices(leftVertices))
    #         if rightVertices:
    #             rightVertices.extend(intersectionPoints)
    #             right.append(self._makePolygonFromVertices(rightVertices))
    #     return left, right

    def _splitTriangles(self, polygons: List[Polygon], planeNormal: Vector, planePoint: Vector):
        goingLeft, goingRight = [], []
        for polygon in polygons:
            left, right,  = self._splitTriangle(polygon, planeNormal, planePoint)
            goingLeft.extend(left)
            goingRight.extend(right)
        return goingLeft, goingRight

    def _splitTriangle(self, polygon: Polygon, planeNormal: Vector, planePoint: Vector):
        goingLeft, goingRight = [], []
        leftVertices, rightVertices, contained = self._checkVerticesPlaneSide(polygon.vertices, planeNormal, planePoint)
        if contained:
            left, right = self._splitFromContained(leftVertices, rightVertices, contained, polygon, planeNormal, planePoint)
            goingLeft.extend(left)
            goingRight.extend(right)

        else:
            left, right = self._splitFromNotContained(leftVertices, rightVertices, polygon, planeNormal, planePoint)
            goingLeft.extend(left)
            goingRight.extend(right)

        return goingLeft, goingRight

    def _splitFromNotContained(self, leftVertices, rightVertices, polygon, planeNormal, planePoint):
        nLeft = len(leftVertices)
        left, right,  = [], []
        if nLeft == 1:
            direction = leftVertices[0] - rightVertices[0]
            ray = Ray(leftVertices[0], direction, direction.getNorm())
            intersectionPoint1 = self._intersectPlaneWithRay(planeNormal, planePoint, ray)
            direction = leftVertices[0] - rightVertices[1]
            ray = Ray(leftVertices[0], direction, direction.getNorm())
            intersectionPoint2 = self._intersectPlaneWithRay(planeNormal, planePoint, ray)
            left.append(self._makeTriangleFromVertices(polygon, [intersectionPoint1, intersectionPoint2, leftVertices[0]]))
            right.append(self._makeTriangleFromVertices(polygon, [intersectionPoint1, rightVertices[0], intersectionPoint2]))
            right.append(self._makeTriangleFromVertices(polygon, [intersectionPoint2, rightVertices[0], rightVertices[1]]))
        else:
            direction = rightVertices[0] - leftVertices[0]
            ray = Ray(rightVertices[0], direction, direction.getNorm())
            intersectionPoint1 = self._intersectPlaneWithRay(planeNormal, planePoint, ray)
            direction = rightVertices[0] - leftVertices[1]
            ray = Ray(rightVertices[0], direction, direction.getNorm())
            intersectionPoint2 = self._intersectPlaneWithRay(planeNormal, planePoint, ray)
            right.append(self._makeTriangleFromVertices(polygon, [intersectionPoint1, intersectionPoint2, rightVertices[0]]))
            left.append(self._makeTriangleFromVertices(polygon, [intersectionPoint1, leftVertices[0], intersectionPoint2]))
            left.append(self._makeTriangleFromVertices(polygon, [intersectionPoint2, leftVertices[0], leftVertices[1]]))

        return left, right

    def _splitFromContained(self, leftVertices, rightVertices, contained, polygon, planeNormal, planePoint):
        nContained = len(contained)
        right, left = [], []
        if nContained == 3:
            right.append(polygon)
            left.append(polygon)
        elif nContained == 2:
            if leftVertices:
                left.append(polygon)
            elif rightVertices:
                right.append(polygon)
        elif nContained == 1:
            if not rightVertices:
                leftVertices.append(contained[0])
                left.append(self._makeTriangleFromVertices(polygon, leftVertices))
            elif not leftVertices:
                rightVertices.append(contained[0])
                right.append(self._makeTriangleFromVertices(polygon, rightVertices))
            else:
                direction = leftVertices[0] - rightVertices[0]
                ray = Ray(leftVertices[0], direction, direction.getNorm())
                intersectionPoint = self._intersectPlaneWithRay(planeNormal, planePoint, ray)
                if intersectionPoint:
                    leftVertices.append(intersectionPoint)
                    leftVertices.append(contained[0])
                    left.append(self._makeTriangleFromVertices(polygon, leftVertices))
                    rightVertices.append(intersectionPoint)
                    rightVertices.append(contained[0])
                    right.append(self._makeTriangleFromVertices(polygon, rightVertices))
                else:
                    raise
        return left, right

    @staticmethod
    def _checkVerticesPlaneSide(vertices: List[Vector], planeNormal: Vector, planePoint: Vector, tol=1e-6):
        """Based on the fact that the plane normal will always point towards the positive axis, and that we search our
         min(SAH) in that order as well, we can conclude that if diff is negative, the point is on the right side
        of the plane, and if diff is positive, the point is on the left side of the plane."""
        contained = []
        left = []
        right = []
        for vertex in vertices:
            diff = planeNormal.dot(vertex - planePoint)
            if abs(diff) < tol:
                contained.append(vertex)
            elif diff > 0:
                right.append(vertex)
            else:
                left.append(vertex)
        return left, right, contained

    @staticmethod
    def _intersectPlaneWithRay(normal: Vector, planePoint: Vector, ray: Ray, tol=1e-6):
        """
        Naive Algorithm. It is a little bit faster than Möller-Trombore for this purpose.
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
    def _makeTriangleFromVertices(parent, vertices):
        if len(vertices) == 3:
            return Triangle(*vertices, normal=parent.normal, insideMaterial=parent.insideMaterial,
                            outsideMaterial=parent.outsideMaterial)
        if len(vertices) == 4:
            return Quad(*vertices, normal=parent.normal, insideMaterial=parent.insideMaterial,
                        outsideMaterial=parent.outsideMaterial)
        if len(vertices) > 4:
            return Polygon(vertices=vertices, normal=parent.normal, insideMaterial=parent.insideMaterial,
                           outsideMaterial=parent.outsideMaterial)

    @staticmethod
    def _makeSplitPlane(splitAxis: str, splitValue: float) -> Tuple[Vector, Vector]:
        """The plane normal will always point towards the positive side of the axis,
        this is important for the  triangle splitting algorithm, which checks vertex plane side based on the normal."""
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

    def _classifyPolygons(self, splitLine: float, splitAxis: str, polygons: List[Polygon]):
        goingLeft = []
        goingRight = []
        toBeSplit = []
        normal, point = self._makeSplitPlane(splitAxis, splitLine)
        for i, polygon in enumerate(polygons):
            left, right, contained = self._checkVerticesPlaneSide(polygon.vertices, normal, point)
            nLeft = len(left)
            nRight = len(right)
            nContained = len(contained)
            if nLeft == 3 or (nLeft == 2 and nContained == 1) or (nLeft == 1 and nContained == 2):
                goingLeft.append(polygon)
                continue
            elif nRight == 3 or (nRight == 2 and nContained == 1) or (nRight == 1 and nContained == 2):
                goingRight.append(polygon)
                continue
            elif nContained == 3:
                if polygons[i-1] in goingLeft:
                    goingLeft.append(polygon)
                else:
                    goingRight.append(polygon)
            else:
                toBeSplit.append(polygon)
        return [goingLeft, goingRight, toBeSplit]
