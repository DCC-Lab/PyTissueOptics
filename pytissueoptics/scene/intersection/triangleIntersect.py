from typing import Union

from pytissueoptics.scene.geometry import Vector, Triangle


class TriangleIntersectStrategy:
    def getIntersection(self, rayOrigin: Vector, rayDirection: Vector, triangle: Triangle) -> Union[Vector, None]:
        """ Returns None if no intersection is found, else returns the intersection point. """
        raise NotImplemented


class MollerTrumboreIntersect(TriangleIntersectStrategy):
    """ Möller–Trumbore ray-triangle 3D intersection algorithm. """
    EPSILON = 0.0000001

    def getIntersection(self, rayOrigin: Vector, rayDirection: Vector, triangle: Triangle) -> Union[Vector, None]:
        v1, v2, v3 = triangle.vertices
        edgeA = v2 - v1
        edgeB = v3 - v1
        pVector = rayDirection.cross(edgeB)
        determinant = edgeA.dot(pVector)

        rayIsParallel = abs(determinant) < self.EPSILON
        if rayIsParallel:
            return None

        inverseDeterminant = 1. / determinant
        tVector = rayOrigin - v1
        u = tVector.dot(pVector) * inverseDeterminant
        if u < 0. or u > 1.:
            return None

        qVector = tVector.cross(edgeA)
        v = rayDirection.dot(qVector) * inverseDeterminant
        if v < 0. or u + v > 1.:
            return None

        t = edgeB.dot(qVector) * inverseDeterminant
        lineIntersection = t < self.EPSILON
        if lineIntersection:
            return None

        return rayOrigin + rayDirection * t
