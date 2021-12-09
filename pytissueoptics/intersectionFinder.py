from typing import List

from pytissueoptics import Geometry, FresnelIntersect


class IntersectionFinder:
    def __init__(self, geometries: List[Geometry]):
        self.geometries = geometries


class SimpleIntersectionFinder(IntersectionFinder):
    def search(self, position, direction, distance) -> FresnelIntersect:  # todo: search(Line)
        geometry = self.geometryAt(position)
        if geometry is None:
            return self._worldSearch(position, direction, distance)

        intersect = geometry.nextExitInterface(position, direction, distance)
        return intersect

    def geometryAt(self, position):
        for geometry in self.geometries:
            localCoordinates = position - geometry.origin
            if geometry.contains(localCoordinates):
                return geometry
        return None

    def _worldSearch(self, position, direction, distance):
        shortestDistance = distance
        closestIntersect = None

        for geometry in self.geometries:
            localPosition = position - geometry.origin
            someIntersection = geometry.nextEntranceInterface(localPosition, direction, shortestDistance)
            if someIntersection is not None:
                if someIntersection.distance < shortestDistance:
                    shortestDistance = someIntersection.distance
                    closestIntersect = someIntersection

        return closestIntersect

