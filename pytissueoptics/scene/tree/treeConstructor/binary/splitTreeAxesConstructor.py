from typing import List, Tuple
import sys

from pytissueoptics import Vector
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.geometry import Polygon, Triangle
from pytissueoptics.scene.tree import Node
from pytissueoptics.scene.tree.treeConstructor import SplitNodeResult
from pytissueoptics.scene.tree.treeConstructor.binary import NoSplitOneAxisConstructor


class SplitThreeAxesConstructor(NoSplitOneAxisConstructor):
    """
        This is an implementation of the proposed algorithms found in

        Wald, Ingo, and Vlastimil Havran. 2006. “On Building Fast Kd-Trees for Ray Tracing, and on Doing That in O(N Log N).
        ” In 2006 IEEE Symposium on Interactive Ray Tracing, 61–69.

        Soupikov, Alexei, Maxim Shevtsov, and Alexander Kapustin. 2008. “Improving Kd-Tree Quality at a Reasonable
        Construction Cost.” In Interactive Ray Tracing, 2008. RT 2008. IEEE Symposium on, 67–72. unknown.
        -
        This algorithm uses a
        1. Surface Area Heuristic (SAH) search on 3 axis
        2. Perfect Splits
        3. On-the-fly pruning
        When the minimum SAH axis and value are found, a split cost is calculated.
        If the children's cost is lower than the parent, splitting is allowed, else, no splitting occurs.

        Contrary to belief, the current implementation is not faster than non-split SAH-based tree construction. A split
        results in 3 triangles, whereas a non-split results in a shared triangle, the equivalent of 2 distinct triangles.
        It seems no matter the parameters of the tree, the intersection cost brought by the extra triangles is too high.

        """

    def __init__(self, nbOfSplitPlanes: int = 20, intersectionCost: float = 3, traversalCost: float = 6,
                 noSharedBonus: float = 2, emptySpaceBonus: float = 2):
        super().__init__(nbOfSplitPlanes, intersectionCost, traversalCost, noSharedBonus, emptySpaceBonus)

    def _splitNode(self, node: Node) -> SplitNodeResult:
        self.currentNode = node
        splitBbox = self.currentNode.bbox.copy()
        minSAH = sys.float_info.max
        for axis in ["x", "y", "z"]:
            thisSAH = self._searchMinSAHOnAxis(splitBbox, axis, minSAH)
            if thisSAH < minSAH:
                minSAH = thisSAH
        normal, point = self._makeSplitPlane(self.result.splitAxis, self.result.splitValue)
        left, right = self._splitTriangles(normal, point)
        self.result.leftPolygons.extend(left)
        self.result.rightPolygons.extend(right)
        self._trimChildrenBbox()
        stopCondition = self._checkStopCondition()
        newNodeResult = SplitNodeResult(stopCondition, [self.result.leftBbox, self.result.rightBbox],
                                        [self.result.leftPolygons, self.result.rightPolygons])
        return newNodeResult

    def _splitTriangles(self, planeNormal: Vector, planePoint: Vector) -> Tuple[List[Polygon], List[Polygon]]:
        goingLeft, goingRight = [], []
        if self.result.splitPolygons:
            for polygon in self.result.splitPolygons:
                left, right, = self._splitTriangle(polygon, planeNormal, planePoint)
                goingLeft.extend(left)
                goingRight.extend(right)

        return goingLeft, goingRight

    def _splitTriangle(self, polygon: Polygon, planeNormal: Vector, planePoint: Vector) -> Tuple[
        List[Polygon], List[Polygon]]:
        goingLeft, goingRight = [], []
        leftVertices, rightVertices, contained = self._checkVerticesPlaneSide(polygon.vertices, planeNormal, planePoint)
        nLeft = len(leftVertices)
        nRight = len(rightVertices)
        nContained = len(contained)
        if nContained == 3:
            goingLeft.append(polygon)
            goingRight.append(polygon)

        elif nLeft + nContained == 3:
            goingLeft.append(polygon)

        elif nRight + nContained == 3:
            goingRight.append(polygon)

        elif nContained != 0:
            left, right = self._splitFromMiddleContained(leftVertices, rightVertices, contained, polygon, planeNormal,
                                                         planePoint)
            goingLeft.extend(left)
            goingRight.extend(right)

        else:
            left, right = self._splitFromNotContained(leftVertices, rightVertices, polygon, planeNormal, planePoint)
            goingLeft.extend(left)
            goingRight.extend(right)

        return goingLeft, goingRight

    def _splitFromNotContained(self, leftVertices: List[Vector], rightVertices: List[Vector], polygon: Polygon,
                               planeNormal: Vector, planePoint: Vector) -> Tuple[List[Polygon], List[Polygon]]:
        left, right, = [], []
        if len(leftVertices) == 1:
            direction = rightVertices[0] - leftVertices[0]
            ray = Ray(leftVertices[0], direction, direction.getNorm())
            intersectionPoint1 = self._intersectPlaneWithRay(planeNormal, planePoint, ray)
            direction = rightVertices[1] - leftVertices[0]
            ray = Ray(leftVertices[0], direction, direction.getNorm())
            intersectionPoint2 = self._intersectPlaneWithRay(planeNormal, planePoint, ray)
            left.append(
                self._makeTriangleFromVertices(polygon, [intersectionPoint1, intersectionPoint2, leftVertices[0]]))
            right.append(
                self._makeTriangleFromVertices(polygon, [intersectionPoint1, rightVertices[0], intersectionPoint2]))
            right.append(
                self._makeTriangleFromVertices(polygon, [intersectionPoint2, rightVertices[0], rightVertices[1]]))
        else:
            direction = leftVertices[0] - rightVertices[0]
            ray = Ray(rightVertices[0], direction, direction.getNorm())
            intersectionPoint1 = self._intersectPlaneWithRay(planeNormal, planePoint, ray)
            direction = leftVertices[1] - rightVertices[0]
            ray = Ray(rightVertices[0], direction, direction.getNorm())
            intersectionPoint2 = self._intersectPlaneWithRay(planeNormal, planePoint, ray)
            right.append(
                self._makeTriangleFromVertices(polygon, [intersectionPoint1, intersectionPoint2, rightVertices[0]]))
            left.append(
                self._makeTriangleFromVertices(polygon, [intersectionPoint1, leftVertices[0], intersectionPoint2]))
            left.append(self._makeTriangleFromVertices(polygon, [intersectionPoint2, leftVertices[0], leftVertices[1]]))

        return left, right

    def _splitFromMiddleContained(self, leftVertices: List[Vector], rightVertices: List[Vector],
                                  contained: List[Vector], polygon: Polygon, planeNormal: Vector, planePoint: Vector) -> \
    Tuple[List[Polygon], List[Polygon]]:
        right, left = [], []
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
    def _makeTriangleFromVertices(parent: Polygon, vertices: List[Vector]) -> Polygon:
        if len(vertices) == 3:
            return Triangle(*vertices, normal=parent.normal, insideMaterial=parent.insideMaterial,
                            outsideMaterial=parent.outsideMaterial)

    @staticmethod
    def _checkVerticesPlaneSide(vertices: List[Vector], planeNormal: Vector, planePoint: Vector, tol=1e-6) -> Tuple[
        List[Vector], List[Vector], List[Vector]]:
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
    def _intersectPlaneWithRay(normal: Vector, planePoint: Vector, ray: Ray, tol=1e-6) -> Vector:
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
    def _getPolygonAsRays(polygon: Polygon) -> List[Ray]:
        polygonRays = []
        for i, vertex in enumerate(polygon.vertices):
            if i == len(polygon.vertices) - 1:
                nextVertex = polygon.vertices[0]
            else:
                nextVertex = polygon.vertices[i + 1]
            direction = nextVertex - vertex
            polygonRays.append(Ray(vertex, direction, direction.getNorm()))
        return polygonRays
