from typing import Union, Optional

from pytissueoptics.scene.geometry import Vector, Triangle, Quad, Polygon
from pytissueoptics.scene.intersection import Ray


class MollerTrumboreIntersect:
    EPSILON = 0.0000001

    def getIntersection(self, ray: Ray, polygon: Union[Triangle, Quad, Polygon]) -> Optional[Vector]:
        if isinstance(polygon, Triangle):
            return self._getTriangleIntersection(ray, polygon)
        if isinstance(polygon, Quad):
            return self._getQuadIntersection(ray, polygon)
        if isinstance(polygon, Polygon):
            return self._getPolygonIntersection(ray, polygon)

    def _getTriangleIntersection(self, ray: Ray, triangle: Triangle) -> Optional[Vector]:
        """ Möller–Trumbore ray-triangle 3D intersection algorithm. """
        v1, v2, v3 = triangle.vertices
        edgeA = v2 - v1
        edgeB = v3 - v1
        pVector = ray.direction.cross(edgeB)
        determinant = edgeA.dot(pVector)

        rayIsParallel = abs(determinant) < self.EPSILON
        if rayIsParallel:
            return None

        inverseDeterminant = 1. / determinant
        tVector = ray.origin - v1
        u = tVector.dot(pVector) * inverseDeterminant
        if u < 0. or u > 1.:
            return None

        qVector = tVector.cross(edgeA)
        v = ray.direction.dot(qVector) * inverseDeterminant
        if v < 0. or u + v > 1.:
            return None

        t = edgeB.dot(qVector) * inverseDeterminant
        lineIntersection = t < self.EPSILON
        if lineIntersection:
            return None

        if ray.length and t > ray.length:
            return None

        return ray.origin + ray.direction * t

    def _getQuadIntersection(self, ray: Ray, quad: Quad) -> Optional[Vector]:
        v1, v2, v3, v4 = quad.vertices
        triangleA = Triangle(v1, v2, v4)
        triangleB = Triangle(v2, v3, v4)
        intersectionA = self._getTriangleIntersection(ray, triangleA)
        if intersectionA:
            return intersectionA
        return self._getTriangleIntersection(ray, triangleB)

    def _getPolygonIntersection(self, ray: Ray, polygon: Polygon) -> Optional[Vector]:
        for i in range(len(polygon.vertices) - 2):
            triangle = Triangle(polygon.vertices[0], polygon.vertices[i + 1], polygon.vertices[i + 2])
            intersection = self.getIntersection(ray, triangle)
            if intersection:
                return intersection
        return None
