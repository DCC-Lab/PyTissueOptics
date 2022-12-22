import math
from typing import List

from pytissueoptics.scene.geometry import Vector, Triangle, primitives, Vertex
from pytissueoptics.scene.solids import Solid


class Cylinder(Solid):
    def __init__(self, radius: float = 1, height: float = 1, u: int = 32, v: int = 3,
                 position: Vector = Vector(0, 0, 0), material=None,
                 primitive: str = primitives.DEFAULT, label: str = "Cylinder"):
        self._radius = radius
        self._height = height
        if u < 2 or v < 1:
            raise ValueError("u must be > 2 and v must be > 1")
        self._u = u
        self._v = v
        self._bottomCenter = Vertex(0, 0, 0)
        self._topCenter = Vertex(0, 0, height)
        self._minRadius = math.cos(math.pi / self._u) * self._radius
        super().__init__(position=position, material=material, primitive=primitive,
                         vertices=[self._bottomCenter, self._topCenter], smooth=True, label=label)

    @property
    def direction(self) -> Vector:
        return self._topCenter - self._bottomCenter

    def _computeTriangleMesh(self):
        verticesGroups = self._computeVertices()
        self._computeMiddleTriangles(verticesGroups)
        self._computeBottomTriangles(verticesGroups[0])
        self._computeTopTriangles(verticesGroups[-1])

    def _computeVertices(self) -> List[List[Vertex]]:
        verticesGroups = []
        radianStep = 2 * math.pi / self._u
        verticalStep = self._height / self._v

        for j in range(0, self._v + 1):
            currentLayer = []
            for i in range(self._u):
                vertex = self.getMidVertex(i, j, radianStep, verticalStep)
                currentLayer.append(vertex)
            verticesGroups.append(currentLayer)
            self._vertices.extend(currentLayer)

        return verticesGroups

    def getMidVertex(self, i: int, j: int, radianStep: float, verticalStep: float) -> Vertex:
        shrinkFactor = self._getShrinkFactor(verticalStep * j)
        x = self._radius * shrinkFactor * math.cos(i * radianStep)
        y = self._radius * shrinkFactor * math.sin(i * radianStep)
        z = j * verticalStep
        return Vertex(x, y, z)

    def _computeMiddleTriangles(self, verticesGroups: List[List[Vertex]]):
        middleTriangles = []
        for i in range(self._v):
            currentGroup = verticesGroups[i]
            nextGroup = verticesGroups[i + 1]
            for j in range(self._u):
                nextIndex = (j + 1) % self._u
                middleTriangles.append(Triangle(currentGroup[j], nextGroup[nextIndex], nextGroup[j]))
                middleTriangles.append(Triangle(currentGroup[j], currentGroup[nextIndex], nextGroup[nextIndex]))
        self._surfaces.add("middle", middleTriangles)

    def _computeBottomTriangles(self, vertices: List[Vertex]):
        bottomTriangles = []
        for i in range(self._u):
            nextIndex = (i + 1) % self._u
            bottomTriangles.append(Triangle(self._bottomCenter, vertices[i], vertices[nextIndex]))
        self._surfaces.add("bottom", bottomTriangles)

    def _computeTopTriangles(self, vertices: List[Vertex]):
        topTriangles = []
        for i in range(self._u):
            nextIndex = (i + 1) % self._u
            topTriangles.append(Triangle(self._topCenter, vertices[i], vertices[nextIndex]))
        self._surfaces.add("top", topTriangles)

    def _computeQuadMesh(self):
        raise NotImplementedError("Quad mesh not implemented for Cylinder")

    def contains(self, *vertices: Vertex) -> bool:
        for vertex in vertices:
            direction = self.direction
            direction.normalize()
            localPoint = vertex - self._position
            alongCylinder = direction.dot(localPoint)
            if alongCylinder < 0 or alongCylinder > self._height:
                return False
            else:
                radiusCheck = self._minRadiusAtHeightAlong(alongCylinder)
                radialComponent = localPoint - direction * alongCylinder
                radialDistanceFromBase = radialComponent.getNorm()
                if radialDistanceFromBase > radiusCheck:
                    return False

        return True

    def _minRadiusAtHeightAlong(self, heightAlong: float) -> float:
        shrinkFactor = self._getShrinkFactor(heightAlong)
        return self._minRadius * shrinkFactor

    @staticmethod
    def _getShrinkFactor(heightAlong: float) -> float:
        return 1

    def smooth(self, surfaceLabel: str = None):
        super(Cylinder, self).smooth("middle")
