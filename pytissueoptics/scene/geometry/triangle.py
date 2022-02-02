from pytissueoptics.scene.geometry import Polygon, Vector
from pytissueoptics.scene.materials import Material


class Triangle(Polygon):
    def __init__(self, v1: Vector, v2: Vector, v3: Vector,
                 insideMaterial: Material = None, outsideMaterial: Material = None):
        super().__init__(vertices=[v1, v2, v3],
                         insideMaterial=insideMaterial, outsideMaterial=outsideMaterial)

    @property
    def v1(self):
        return self._vertices[0]

    @property
    def v2(self):
        return self._vertices[1]

    @property
    def v3(self):
        return self._vertices[2]
