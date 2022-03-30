from pytissueoptics.scene.geometry import Polygon, Vector
from pytissueoptics.scene.materials import Material


class Triangle(Polygon):
    def __init__(self, v1: Vector, v2: Vector, v3: Vector,
                 insideMaterial: Material = None, outsideMaterial: Material = None, normal: Vector = None):
        super().__init__(vertices=[v1, v2, v3],
                         insideMaterial=insideMaterial, outsideMaterial=outsideMaterial, normal=normal)
