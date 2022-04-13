from pytissueoptics.scene.geometry import Polygon, Vector, Vertex
from pytissueoptics.scene.materials import Material


class Quad(Polygon):
    def __init__(self, v1: Vertex, v2: Vertex, v3: Vertex, v4: Vertex,
                 insideMaterial: Material = None, outsideMaterial: Material = None, normal: Vector = None):
        super().__init__(vertices=[v1, v2, v3, v4],
                         insideMaterial=insideMaterial, outsideMaterial=outsideMaterial, normal=normal)
