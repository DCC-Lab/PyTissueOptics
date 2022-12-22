from pytissueoptics.scene.geometry import primitives, Vector
from pytissueoptics.scene.solids import Cylinder


class Cone(Cylinder):
    def __init__(self, radius: float = 1, height: float = 1, u: int = 32, v: int = 3,
                 position: Vector = Vector(0, 0, 0), material=None, primitive: str = primitives.DEFAULT,
                 label: str = "Cone"):
        super().__init__(radius, height, u, v, position, material, primitive, label)

    def _getShrinkFactor(self, heightAlong: float) -> float:
        return (self._height - heightAlong) / self._height

    def _computeQuadMesh(self):
        raise NotImplementedError("Quad mesh not implemented for Cylinder")
