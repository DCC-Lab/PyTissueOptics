from pytissueoptics.scene.geometry import Vector, primitives
from pytissueoptics.scene.solids import Cylinder


class Cone(Cylinder):
    def __init__(self, radius: float = 1, length: float = 1, u: int = 32, v: int = 3, s: int = 1,
                 position: Vector = Vector(0, 0, 0), material=None, primitive: str = primitives.DEFAULT,
                 label: str = "Cone"):
        super().__init__(radius, length, u, v, s, position, material, primitive, label)

    def _getShrinkFactor(self, heightAlong: float) -> float:
        return (self._length - heightAlong) / self._length

    def _computeQuadMesh(self):
        raise NotImplementedError("Quad mesh not implemented for Cylinder")
