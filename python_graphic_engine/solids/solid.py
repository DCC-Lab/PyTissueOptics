from typing import Dict, List

from graphics.geometry import Vector, Polygon
from graphics.materials import Material


class Solid:
    def __init__(self, position: Vector, material: Material, vertices: List[Vector], surfaces: Dict[str, List[Polygon]]):
        self._position = position
        self._material = material
        self._vertices = vertices
        self._surfaces = surfaces
