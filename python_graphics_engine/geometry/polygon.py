from abc import ABC, abstractmethod
from typing import List

from python_graphics_engine.geometry import Vector


class Polygon(ABC):
    """
    Abstract class for any planar polygon.

    Requires the vertices to be given in an anti-clockwise order
     for the normal to point towards the viewer.
    """
    def __init__(self, vertices: List[Vector]):
        self._vertices = vertices
        self._normal = None

        self._computeNormal()

    @property
    def normal(self):
        return self._normal

    def _computeNormal(self):
        """
        For any planar polygon, the first 3 vertices define a triangle with the same normal.
        We use two edges of this triangle to compute the normal (in-order cross-product).
        """
        edgeA = self._vertices[1] - self._vertices[0]
        edgeB = self._vertices[2] - self._vertices[1]
        N = edgeA.cross(edgeB)
        N.normalize()
        self._normal = N
