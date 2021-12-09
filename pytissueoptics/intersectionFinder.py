from typing import List

from pytissueoptics import Geometry


class IntersectionFinder:
    def __init__(self, geometries: List[Geometry]):
        self.geometries = geometries


class SimpleIntersectionFinder(IntersectionFinder):

