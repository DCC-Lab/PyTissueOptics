from typing import List, Tuple
import sys

from pytissueoptics import Vector
from pytissueoptics.scene import Ray
from pytissueoptics.scene.geometry import Polygon, BoundingBox, Triangle
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor import TreeConstructor, SplitNodeResult
from pytissueoptics.scene.tree.treeConstructor.binary import SAHSearchResult


class ThreeAxesSplitTreeConstructor(TreeConstructor):
    def __init__(self, nbOfSplitPlanes: int = 30, splitCostPercentage: float = 0.2, fallback: float = 1e-6):
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
        normal, point = self._makeSplitPlane(result.splitAxis, result.splitValue)
        goingLeft, goingRight = self._splitTriangles(result.mPolygons, normal, point)
        result.lPolygons.extend(goingLeft)
        result.rPolygons.extend(goingRight)
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
        for splitAxis in ["x", "y", "z"]:
            aMin, aMax = splitBbox.getAxisLimits(splitAxis)
            step = splitBbox.getAxisWidth(splitAxis) / (self._nbOfSplitPlanes + 1)
            for i in range(1, self._nbOfSplitPlanes):
                splitValue = aMin + i * step
                left, right, toSplit = self._classifyFromBbox(polygons, splitValue, splitAxis)
                tempLeftBbox = splitBbox.copy()
                tempLeftBbox.update(splitAxis, "max", splitValue)
                tempRightBbox = splitBbox.copy()
                tempRightBbox.update(splitAxis, "min", splitValue)
                leftSAH = len(left) * tempLeftBbox.getArea()
                rightSAH = len(right) * tempRightBbox.getArea()
                newSAH = leftSAH + rightSAH
                if newSAH < minSAH:
                    minSAH = newSAH
                    result = SAHSearchResult(left, right, toSplit, tempLeftBbox, tempRightBbox, len(left), len(right),
                                             leftSAH, rightSAH, leftSAH+rightSAH, splitAxis, splitValue)
        return result

    def _splitTriangles(self, polygons: List[Polygon], planeNormal: Vector, planePoint: Vector):
        goingLeft, goingRight = [], []
        if polygons:
            for polygon in polygons:
                left, right,  = self._splitTriangle(polygon, planeNormal, planePoint)
                goingLeft.extend(left)
                goingRight.extend(right)
        return goingLeft, goingRight

    def _splitTriangle(self, polygon: Polygon, planeNormal: Vector, planePoint: Vector):
        goingLeft, goingRight = [], []
        leftVertices, rightVertices, contained = self._checkVerticesPlaneSide(polygon.vertices, planeNormal, planePoint)
        nLeft = len(leftVertices)
        nRight = len(rightVertices)
        nContained = len(contained)

        if nLeft == 3 or (nLeft == 2 and nContained == 1) or (nLeft == 1 and nContained == 2):
            goingLeft.append(polygon)

        elif nRight == 3 or (nRight == 2 and nContained == 1) or (nRight == 1 and nContained == 2):
            goingRight.append(polygon)

        elif nRight == 1 and nLeft == 1 and nContained == 1:
            left, right = self._splitFromContained(leftVertices, rightVertices, polygon, planeNormal, planePoint)
            goingLeft.extend(left)
            goingRight.extend(right)

        elif nContained == 3:
            print("Error: 3 vertices contained in plane")

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
        if nContained == 0:
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
    def _makeTriangleFromVertices(parent, vertices):
        if len(vertices) == 3:
            return Triangle(*vertices, normal=parent.normal, insideMaterial=parent.insideMaterial,
                            outsideMaterial=parent.outsideMaterial)

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

    @staticmethod
    def _classifyFromBbox(polygons, splitValue, splitAxis):
        goingLeft = []
        goingRight = []
        goingToSplit = []
        tol = 1E-6
        lastSide = 0
        for polygon in polygons:
            limitMin = polygon.bbox.getAxisLimit(splitAxis, "min")
            limitMax = polygon.bbox.getAxisLimit(splitAxis, "min")
            if limitMin < splitValue and limitMax < splitValue:
                goingLeft.append(polygon)
                lastSide = 0
            elif limitMin > splitValue and limitMax > splitValue:
                goingRight.append(polygon)
                lastSide = 1
            # if polygon is on the split plane, classify now since we still have awareness of the other polygons #
            elif abs(limitMin - splitValue) < tol and abs(limitMax-splitValue) < tol:
                if lastSide == 0:
                    goingLeft.append(polygon)
                else:
                    goingRight.append(polygon)
            else:
                goingToSplit.append(polygon)
        return goingLeft, goingRight, goingToSplit

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

