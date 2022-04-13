from math import cos, sin, pi

from pytissueoptics.scene.geometry import Vector, Triangle, primitives, utils
from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.solids import Solid


class Cylinder(Solid):
    def __init__(self, radius: float, height: float,  u: int, position: Vector = Vector(0, 0, 0), material: Material = None,
                 primitive: str = primitives.DEFAULT):
        self._radius = radius
        self._height = height
        if u > 2:
            self._u = u
        else:
            raise ValueError("u must be greater than 2")
        self._u = u
        super().__init__(position=position, material=material, primitive=primitive, vertices=[])

    def _computeTriangleMesh(self):
        bottomCenterVertex = self._position
        topCenterVertex = self._position + Vector(0, 0, self._height)
        topVertices = []
        bottomVertices = []
        radianStep = 2 * pi / self._u
        for j in range(self._u):
            bottomVertices.append(bottomCenterVertex + Vector(self._radius * cos(j * radianStep), self._radius * sin(j * radianStep), 0))
            topVertices.append(topCenterVertex + Vector(self._radius * cos(j * radianStep), self._radius * sin(j * radianStep), 0))

        bottomTriangles = []
        topTriangles = []
        middleTriangles = []
        for i in range(len(bottomVertices)):
            nextIndex = i + 1
            if i + 1 == len(bottomVertices):
                nextIndex = 0
            bottomTriangles.append(Triangle(bottomVertices[i], bottomVertices[nextIndex], bottomCenterVertex))
            topTriangles.append(Triangle(topVertices[i], topCenterVertex, topVertices[nextIndex]))
            middleTriangles.append(Triangle(bottomVertices[i], topVertices[i], topVertices[nextIndex]))
            middleTriangles.append(Triangle(bottomVertices[i], topVertices[nextIndex], bottomVertices[nextIndex]))
        self._vertices.extend(bottomVertices)
        self._vertices.extend(topVertices)
        self._vertices.append(bottomCenterVertex)
        self._vertices.append(topCenterVertex)
        self._surfaces.add("Top", topTriangles)
        self._surfaces.add("Bottom", bottomTriangles)
        self._surfaces.add("Middle", middleTriangles)

    def _computeQuadMesh(self):
        raise NotImplementedError("Quad mesh not implemented for Cylinder")

    def contains(self, *vertices: Vector) -> bool:
        raise NotImplementedError("Contains not implemented for Cylinder")
