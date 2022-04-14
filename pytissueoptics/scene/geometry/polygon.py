from typing import List

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.geometry import BoundingBox


class Polygon:
    """
    Abstract class for any planar polygon.

    Requires the vertices to be given in an anti-clockwise order
     for the normal to point towards the viewer.
    """

    def __init__(self, vertices: List[Vector], normal: Vector = None,
                 insideMaterial: Material = None, outsideMaterial: Material = None,
                 surfaceName: str = None):
        self._vertices = vertices
        self._normal = normal
        self._insideMaterial = insideMaterial
        self._outsideMaterial = outsideMaterial
        self.surfaceName = surfaceName
        if self._normal is None:
            self.resetNormal()

        self._bbox = None
        self._centroid = None
        self.resetCentroid()
        self.resetBoundingBox()

    def __eq__(self, other: 'Polygon'):
        for vertex in self._vertices:
            if vertex not in other.vertices:
                return False
        return True

    @property
    def normal(self):
        return self._normal

    @property
    def vertices(self):
        return self._vertices

    @property
    def insideMaterial(self):
        return self._insideMaterial

    @property
    def outsideMaterial(self):
        return self._outsideMaterial

    @property
    def bbox(self) -> BoundingBox:
        return self._bbox

    @property
    def centroid(self) -> Vector:
        return self._centroid

    def setOutsideMaterial(self, material: Material):
        self._outsideMaterial = material

    def setInsideMaterial(self, material: Material):
        self._insideMaterial = material

    def resetCentroid(self):
        vertexSum = Vector(0, 0, 0)
        for vertex in self._vertices:
            vertexSum.add(vertex)
        self._centroid = vertexSum / (len(self._vertices))

    def resetBoundingBox(self):
        self._bbox = BoundingBox.fromVertices(self._vertices)

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

    def getCentroid(self) -> Vector:
        centroid = Vector(0, 0, 0)
        for vertex in self._vertices:
            centroid.add(vertex)
        centroid.divide(len(self._vertices))
        return centroid
