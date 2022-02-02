from python_graphics_engine.geometry import Polygon, Vector
from python_graphics_engine.materials import Material


class Triangle(Polygon):
    def __init__(self, v1: Vector, v2: Vector, v3: Vector,
                 insideMaterial: Material = None, outsideMaterial: Material = None):
        super().__init__(vertices=[v1, v2, v3],
                         insideMaterial=insideMaterial, outsideMaterial=outsideMaterial)
