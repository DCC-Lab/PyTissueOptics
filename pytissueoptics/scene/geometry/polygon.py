from dataclasses import dataclass
from typing import List

from pytissueoptics.scene.geometry import Vector, Vertex
from pytissueoptics.scene.geometry import BoundingBox


@dataclass
class Environment:
    material: ...
    solid: 'Solid' = None


class Polygon:
    """
    Abstract class for any planar polygon.

    Requires the vertices to be given in an anti-clockwise order
     for the normal to point towards the viewer.
    """

    def __init__(self, vertices: List[Vertex], normal: Vector = None,
                 insideEnvironment: Environment = None, outsideEnvironment: Environment = None,
                 surfaceLabel: str = None):
        self._vertices = vertices
        self._normal = normal
        self._insideEnvironment = insideEnvironment
        self._outsideEnvironment = outsideEnvironment
        self.surfaceLabel = surfaceLabel
        if self._normal is None:
            self.resetNormal()

        self._bbox = None
        self._centroid = None
        self.resetCentroid()
        self.resetBoundingBox()
        self.toSmooth = False

    def __eq__(self, other: 'Polygon'):
        for vertex in self._vertices:
            if vertex not in other.vertices:
                return False
        return True

    @property
    def normal(self) -> Vector:
        return self._normal

    @property
    def vertices(self) -> List[Vertex]:
        return self._vertices

    @property
    def insideEnvironment(self) -> Environment:
        return self._insideEnvironment

    @property
    def outsideEnvironment(self) -> Environment:
        return self._outsideEnvironment

    @property
    def bbox(self) -> BoundingBox:
        return self._bbox

    @property
    def centroid(self) -> Vector:
        return self._centroid

    def setOutsideEnvironment(self, environment: Environment):
        self._outsideEnvironment = environment

    def setInsideEnvironment(self, environment: Environment):
        self._insideEnvironment = environment

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
