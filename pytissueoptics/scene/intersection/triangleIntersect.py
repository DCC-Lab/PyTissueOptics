from typing import Union

from pytissueoptics.scene.geometry import Vector, Triangle
from pytissueoptics.scene.intersection import Ray


class TriangleIntersectStrategy:
    def getIntersection(self, ray: Ray, triangle: Triangle) -> Union[Vector, None]:
        """ Returns None if no intersection is found, else returns the intersection point. """
        raise NotImplemented


class MollerTrumboreIntersect(TriangleIntersectStrategy):
    """ Möller–Trumbore ray-triangle 3D intersection algorithm. """
    EPSILON = 0.0000001

    def getIntersection(self, ray: Ray, triangle: Triangle) -> Union[Vector, None]:
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
