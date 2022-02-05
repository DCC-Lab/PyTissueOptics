from typing import List

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.materials import Material


class Polygon:
    """
    Abstract class for any planar polygon.

    Requires the vertices to be given in an anti-clockwise order
     for the normal to point towards the viewer.
    """
    def __init__(self, vertices: List[Vector],
                 insideMaterial: Material = None, outsideMaterial: Material = None):
        self._vertices = vertices
        self._normal = None
        self.insideMaterial = insideMaterial
        self.outsideMaterial = outsideMaterial
        if self._normal is None:
            self.resetNormal()

    @property
    def normal(self):
        return self._normal

    @property
    def vertices(self):
        return self._vertices

    def resetNormal(self):
        """
        For any planar polygon, the first 3 vertices define a triangle with the same normal.
        We use two edges of this triangle to compute the normal (in-order cross-product).
        """
        edgeA = self._vertices[1] - self._vertices[0]
        edgeB = self._vertices[2] - self._vertices[1]
        N = edgeA.cross(edgeB)
        N.normalize()
        self._normal = N
