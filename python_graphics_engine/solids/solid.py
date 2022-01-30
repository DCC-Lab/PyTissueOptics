from typing import Dict, List

from python_graphics_engine.geometry import Vector, Polygon
from python_graphics_engine.materials import Material


class Solid:
    def __init__(self, position: Vector, material: Material, vertices: List[Vector], surfaces: Dict[str, List[Polygon]]):
        self._position = position
        self._material = material
        self._vertices = vertices
        self._surfaces = surfaces
