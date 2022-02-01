from typing import List

from pytissueoptics import FresnelIntersect, Vector


class Segment:
    def __init__(self, origin: Vector, direction: Vector, length: float):
        self.origin = origin
        self.direction = direction
        self.length = length

    def translate(self, vector: Vector):
        self.origin += vector

    @property
    def endPoint(self) -> Vector:
        return self.origin + self.direction * self.length


class IntersectionFinder:
    def __init__(self, geometries: List['Geometry']):
        self.geometries = geometries


class SimpleIntersectionFinder(IntersectionFinder):
    """ Brute-force intersection finder.
        1. Find the geometry that contains the initial position.
        2. If contained, ask the geometry to look for an exit interface.
        3. Else search for closest entrance interface into a geometry.
        4. Return an intersection result or None.
    """
    def search(self, segment: Segment) -> FresnelIntersect:  # todo: search(Line)
        geometry = self.geometryAt(segment.origin)
        if geometry is None:
            return self._worldSearch(segment)

        segment.translate(-geometry.origin)
        intersect = geometry.nextExitInterface(segment.origin, segment.direction, segment.length)
        return intersect

    def geometryAt(self, position):
        for geometry in self.geometries:
            localCoordinates = position - geometry.origin
            if geometry.contains(localCoordinates):
                return geometry
        return None

    def _worldSearch(self, segment):
        shortestDistance = segment.length
        closestIntersect = None

        for geometry in self.geometries:
            localPosition = segment.origin - geometry.origin
            someIntersection = geometry.nextEntranceInterface(localPosition, segment.direction, shortestDistance)
            if someIntersection is not None:
                if someIntersection.distance < shortestDistance:
                    shortestDistance = someIntersection.distance
                    closestIntersect = someIntersection

        return closestIntersect

