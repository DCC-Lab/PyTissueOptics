import math
import warnings
from typing import List

from pytissueoptics.scene.geometry import Vector, Triangle, primitives, Vertex
from pytissueoptics.scene.solids import Solid


class Cylinder(Solid):
    def __init__(self, radius: float = 1, length: float = 1, u: int = 32, v: int = 3, s: int = 2,
                 position: Vector = Vector(0, 0, 0), material=None,
                 primitive: str = primitives.DEFAULT, label: str = "cylinder", smooth=True):
        """
        Default cylinder orientation will be along the z axis. The front face towards the negative z axis and the back
        face towards the positive z axis. Position refers to its centroid. Available surfaces are "front", "lateral" and
        "back".
        """
        self._radius = radius
        self._length = length
        if u < 3:
            raise ValueError("u must be >= 3")
        if length != 0 and v < 1:
            raise ValueError("v must be >= 1 for non-zero length")
        assert radius > 0 and length >= 0
        self._u = u
        self._v = v
        self._s = s
        self._frontCenter = Vertex(0, 0, 0)
        self._backCenter = Vertex(0, 0, length)
        self._minRadius = math.cos(math.pi / self._u) * self._radius
        self._angularStep = 2 * math.pi / self._u
        self._lateralStep = self._length / self._v
        self._radialStep = self._radius / self._s

        super().__init__(position=position, material=material, primitive=primitive,
                         vertices=[], smooth=smooth, label=label)
        self.translateBy(Vector(0, 0, -length / 2))
        self._position += Vector(0, 0, length / 2)

    @property
    def direction(self) -> Vector:
        direction = self._backCenter - self._frontCenter
        direction.normalize()
        return direction

    def _computeTriangleMesh(self):
        frontLayers, lateralLayers, backLayers = self._computeVerticesOfLayers()

        frontLayers.insert(0, [self._frontCenter])
        frontLayers.append(lateralLayers[0])
        backLayers.insert(0, lateralLayers[-1])
        backLayers.append([self._backCenter])
        self._vertices.extend([self._frontCenter, self._backCenter])

        self._surfaces.add("front", self._getSurfaceTriangles(frontLayers))
        self._surfaces.add("lateral", self._getSurfaceTriangles(lateralLayers))
        self._surfaces.add("back", self._getSurfaceTriangles(backLayers))

    def _computeVerticesOfLayers(self) -> tuple:
        v = self._v if self._length != 0 else 0
        frontLayers = self._computeSectionVertices(lateralSteps=[0], radialSteps=list(range(1, self._s)))
        lateralLayers = self._computeSectionVertices(lateralSteps=list(range(v + 1)), radialSteps=[self._s])
        backLayers = self._computeSectionVertices(lateralSteps=[v], radialSteps=list(range(self._s-1, 0, -1)))
        return frontLayers, lateralLayers, backLayers

    def _computeSectionVertices(self, lateralSteps: List[int], radialSteps: List[int]):
        verticesLayers = []
        for k in radialSteps:
            for j in lateralSteps:
                layer = []
                for i in range(self._u):
                    layer.append(self._createVertex(i, j, k))
                verticesLayers.append(layer)
                self._vertices.extend(layer)
        return verticesLayers

    def _createVertex(self, i: int, j: int, k: int) -> Vertex:
        shrinkFactor = self._getShrinkFactor(self._lateralStep * j)
        radiusFactor = k * self._radialStep / self._radius
        if shrinkFactor != 1:
            pass  # This is for cones only. In this case we want to keep linear sampling of radius values.
        else:
            # For lenses, we usually want a uniform mesh after the curve transform, but this transform tends to push
            # vertices outwards (particularly at low radius). To prevent a low mesh resolution in the center, we need
            # to increase the sampling around the center beforehand by forcing smaller radiusFactor values.
            radiusFactor = radiusFactor ** 2
        r = self._radius * radiusFactor * shrinkFactor
        x = r * math.cos(i * self._angularStep)
        y = r * math.sin(i * self._angularStep)
        z = j * self._lateralStep
        return Vertex(x, y, z)

    def _getSurfaceTriangles(self, verticesLayers: List[List[Vertex]]) -> List[Triangle]:
        triangles = []
        for i in range(len(verticesLayers) - 1):
            currentGroup = verticesLayers[i]
            nextGroup = verticesLayers[i + 1]

            if len(currentGroup) == 1:
                triangles.extend(self._getPeakTriangles(self._frontCenter, nextGroup[::-1]))
                continue
            if len(nextGroup) == 1:
                triangles.extend(self._getPeakTriangles(self._backCenter, currentGroup))
                continue

            for j in range(self._u):
                nextIndex = (j + 1) % self._u
                triangles.append(Triangle(currentGroup[j], nextGroup[nextIndex], nextGroup[j]))
                triangles.append(Triangle(currentGroup[j], currentGroup[nextIndex], nextGroup[nextIndex]))
        return triangles

    def _getPeakTriangles(self, peakVertex: Vertex, ringVertices: List[Vertex]) -> List[Triangle]:
        triangles = []
        for j in range(self._u):
            nextIndex = (j + 1) % self._u
            triangles.append(Triangle(peakVertex, ringVertices[j], ringVertices[nextIndex]))
        return triangles

    def _computeQuadMesh(self):
        raise NotImplementedError("Quad mesh not implemented for Cylinder")

    def contains(self, *vertices: Vector) -> bool:
        direction = self.direction
        basePosition = self._position - direction * self._length / 2
        for vertex in vertices:
            localPoint = vertex - basePosition
            alongCylinder = direction.dot(localPoint)
            if alongCylinder < 0 or alongCylinder > self._length:
                return False
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
        if self._u < 16:
            warnings.warn("Smoothing a cylinder with less than 16 sides (u < 16) may result in intersection errors.")
        if surfaceLabel:
            return super(Cylinder, self).smooth(surfaceLabel)
        self.smooth("lateral")
