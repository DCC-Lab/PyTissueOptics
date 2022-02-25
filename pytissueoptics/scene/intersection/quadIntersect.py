from typing import Union

from pytissueoptics.scene.geometry import Vector, Triangle, Quad
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.intersection.triangleIntersect import MollerTrumboreTriangleIntersect


class QuadIntersectStrategy:
    def getIntersection(self, ray: Ray, quad: Quad) -> Union[Vector, None]:
        """ Returns None if no intersection is found, else returns the intersection point. """
        raise NotImplementedError


class MollerTrumboreQuadIntersect(QuadIntersectStrategy):
    """ Split Quad in 2 triangles and do Möller–Trumbore ray-triangle 3D intersection algorithm twice.
    Surprisingly, this algorithm is only around 12% slower than a dedicated ray-quad intersection algorithm. """
    triangleIntersect = MollerTrumboreTriangleIntersect()

    def getIntersection(self, ray: Ray, quad: Quad) -> Union[Vector, None]:
        v1, v2, v3, v4 = quad.vertices
        triangleA = Triangle(v1, v2, v4)
        triangleB = Triangle(v2, v3, v4)
        intersectionA = self.triangleIntersect.getIntersection(ray, triangleA)
        if intersectionA:
            return intersectionA
        return self.triangleIntersect.getIntersection(ray, triangleB)
