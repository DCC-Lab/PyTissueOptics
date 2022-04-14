from math import cos, sin

from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Cylinder


class Cone(Cylinder):
    def getMidVertex(self, i: int, j: int, radianStep: float, verticalStep: float) -> Vector:
        shrinkFactor = (self._v - j) / self._v
        x = self._radius * shrinkFactor * cos(i * radianStep)
        y = self._radius * shrinkFactor * sin(i * radianStep)
        z = j * verticalStep
        return Vector(x, y, z)

    def _computeQuadMesh(self):
        raise NotImplementedError("Quad mesh not implemented for Cylinder")

    def contains(self, *vertices: Vector) -> bool:
        raise NotImplementedError("Contains not implemented for Cylinder")
