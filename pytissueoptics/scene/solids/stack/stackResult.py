from typing import List
from dataclasses import dataclass
from pytissueoptics.scene.geometry import Vector, SurfaceCollection, Vertex


@dataclass
class StackResult:
    """ Domain DTO to help creation of cuboid stacks. """
    shape: List[float]
    position: Vector
    vertices: List[Vertex]
    surfaces: SurfaceCollection
    primitive: str
