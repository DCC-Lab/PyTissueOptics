import sys
from dataclasses import dataclass
from typing import List, Union

from pytissueoptics.scene.geometry import Vector, Polygon
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.intersection.bboxIntersect import GemsBoxIntersect
from pytissueoptics.scene.intersection.triangleIntersect import MollerTrumboreIntersect
from pytissueoptics.scene.solids import Solid


@dataclass
class Intersection:
    position: Vector
    polygon: Polygon


class IntersectionFinder:
    def __init__(self, solids: List[Solid]):
        self._solids = solids
        self._triangleIntersect = MollerTrumboreIntersect()
        self._boxIntersect = GemsBoxIntersect()

    def findIntersection(self, ray: Ray) -> Union[Vector, None]:
        raise NotImplementedError


class SimpleIntersectionFinder(IntersectionFinder):
    def findIntersection(self, ray: Ray) -> Union[Intersection, None]:
        solidCandidate = self._findClosestSolidBBoxIntersection(ray)
        if not solidCandidate:
            return None
        return self._findClosestTriangleIntersection(ray, solidCandidate.getPolygons())

    def _findClosestSolidBBoxIntersection(self, ray) -> Union[Solid, None]:
        closestSolid = None
        closestDistance = sys.maxsize
        for solid in self._solids:
            bboxIntersection = self._boxIntersect.getIntersection(ray, solid.bbox)
            if not bboxIntersection:
                continue
            distance = (bboxIntersection - ray.origin).getNorm()
            if distance < closestDistance:
                closestDistance = distance
                closestSolid = solid
        return closestSolid

    def _findClosestTriangleIntersection(self, ray, triangles) -> Union[Intersection, None]:
        closestTriangle = None
        closestIntersection = None
        closestDistance = sys.maxsize
        for triangle in triangles:
            intersection = self._triangleIntersect.getIntersection(ray, triangle)
            if not intersection:
                continue
            distance = (intersection - ray.origin).getNorm()
            if distance < closestDistance:
                closestDistance = distance
                closestIntersection = intersection
                closestTriangle = triangle
        if not closestIntersection:
            return None
        return Intersection(closestIntersection, closestTriangle)
