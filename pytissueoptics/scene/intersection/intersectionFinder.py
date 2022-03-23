import sys
from dataclasses import dataclass
from typing import List, Union, Tuple

from pytissueoptics.scene.geometry import Vector, Polygon, Triangle, Quad
from pytissueoptics.scene.intersection import Ray
from pytissueoptics.scene.intersection.bboxIntersect import GemsBoxIntersect
from pytissueoptics.scene.intersection.quadIntersect import MollerTrumboreQuadIntersect
from pytissueoptics.scene.intersection.triangleIntersect import MollerTrumboreTriangleIntersect
from pytissueoptics.scene.solids import Solid


@dataclass
class Intersection:
    distance: float
    position: Vector = None
    polygon: Polygon = None
    distanceLeft: float = None


class IntersectionFinder:
    def __init__(self, solids: List[Solid]):
        self._solids = solids
        self._triangleIntersect = MollerTrumboreTriangleIntersect()
        self._quadIntersect = MollerTrumboreQuadIntersect()
        self._boxIntersect = GemsBoxIntersect()

    def findIntersection(self, ray: Ray) -> Union[Intersection, None]:
        raise NotImplementedError


class SimpleIntersectionFinder(IntersectionFinder):
    def findIntersection(self, ray: Ray) -> Union[Intersection, None]:
        bboxIntersections = self._findBBoxIntersectingSolids(ray)
        bboxIntersections.sort(key=lambda x: x[0])
        for (distance, solid) in bboxIntersections:
            intersection = self._findClosestPolygonIntersection(ray, solid.getPolygons())
            if intersection:
                return intersection
        return None

    def _findBBoxIntersectingSolids(self, ray) -> Union[List[Tuple[float, Solid]], None]:
        """ We need to handle the special case where ray starts inside bbox. The Box Intersect will not compute
        the intersection for this case and will instead return ray.origin. When that happens, distance will be 0
        and we exit to check the polygons of this solid. """
        solidCandidates = []
        for solid in self._solids:
            bboxIntersection = self._boxIntersect.getIntersection(ray, solid.bbox)
            if not bboxIntersection:
                continue
            distance = (bboxIntersection - ray.origin).getNorm()
            solidCandidates.append((distance, solid))
            if distance == 0:
                break
        return solidCandidates

    def _findClosestPolygonIntersection(self, ray: Ray, polygons: List[Polygon]) -> Union[Intersection, None]:
        closestIntersection = Intersection(sys.maxsize)
        for polygon in polygons:
            intersectionPoint = self._findPolygonIntersection(ray, polygon)
            if not intersectionPoint:
                continue
            distance = (intersectionPoint - ray.origin).getNorm()
            if distance < closestIntersection.distance:
                closestIntersection = Intersection(distance, intersectionPoint, polygon)

        if closestIntersection.distance == sys.maxsize:
            return None
        if ray.length:
            closestIntersection.distanceLeft = ray.length - closestIntersection.distance

        return closestIntersection

    def _findPolygonIntersection(self, ray: Ray, polygon: Polygon) -> Union[Vector, None]:
        if isinstance(polygon, Triangle):
            return self._triangleIntersect.getIntersection(ray, polygon)
        if isinstance(polygon, Quad):
            return self._quadIntersect.getIntersection(ray, polygon)
