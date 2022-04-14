from math import cos, sin, pi

from pytissueoptics.scene.geometry import Vector, Triangle, primitives, utils
from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.solids import Solid


class Cylinder(Solid):
    def __init__(self, radius: float = 1, height: float = 1, u: int = 32, v: int = 1,
                 position: Vector = Vector(0, 0, 0), material: Material = None,
                 primitive: str = primitives.DEFAULT):
        self._radius = radius
        self._height = height
        if u < 2 or v < 1:
            raise ValueError("u must be > 2 and v must be > 1")
        self._u = u
        self._v = v
        super().__init__(position=position, material=material, primitive=primitive, vertices=[])

    def _computeTriangleMesh(self):
        verticesGroups = self._computeVertices()
        self._computeMiddleTriangles(verticesGroups)
        self._computeBottomTriangles(verticesGroups[0])
        self._computeTopTriangles(verticesGroups[-1])

    def _computeVertices(self):
        verticesGroups = []
        radianStep = 2 * pi / self._u
        verticalStep = self._height / self._v

        for j in range(0, self._v + 1):
            currentLayer = []
            for i in range(self._u):
                vertex = self.getMidVertex(i, j, radianStep, verticalStep)
                currentLayer.append(vertex)
            verticesGroups.append(currentLayer)
            self._vertices.extend(currentLayer)

        return verticesGroups

    def getMidVertex(self, i: int, j: int, radianStep: float, verticalStep: float) -> Vector:
        x = self._radius * cos(i * radianStep)
        y = self._radius * sin(i * radianStep)
        z = j * verticalStep
        return Vector(x, y, z)

    def _computeMiddleTriangles(self, verticesGroups):
        middleTriangles = []
        for i in range(self._v):
            currentGroup = verticesGroups[i]
            nextGroup = verticesGroups[i + 1]
            for j in range(self._u):
                nextIndex = (j + 1) % self._u
                middleTriangles.append(Triangle(currentGroup[j], nextGroup[nextIndex], nextGroup[j]))
                middleTriangles.append(Triangle(currentGroup[j], currentGroup[nextIndex], nextGroup[nextIndex]))
        self._surfaces.add("Middle", middleTriangles)

    def _computeBottomTriangles(self, vertices):
        bottomCenterVertex = Vector(0, 0, 0)
        bottomTriangles = []
        for i in range(self._u):
            nextIndex = (i + 1) % self._u
            bottomTriangles.append(Triangle(bottomCenterVertex, vertices[i], vertices[nextIndex]))
        self._vertices.append(bottomCenterVertex)
        self._surfaces.add("Bottom", bottomTriangles)

    def _computeTopTriangles(self, vertices):
        topCenterVertex = Vector(0, 0, self._height)
        topTriangles = []
        for i in range(self._u):
            nextIndex = (i + 1) % self._u
            topTriangles.append(Triangle(topCenterVertex, vertices[i], vertices[nextIndex]))
        self._vertices.append(topCenterVertex)
        self._surfaces.add("Top", topTriangles)

    def _computeQuadMesh(self):
        raise NotImplementedError("Quad mesh not implemented for Cylinder")

    def contains(self, *vertices: Vector) -> bool:
        raise NotImplementedError("Contains not implemented for Cylinder")
