from dataclasses import dataclass
from typing import Dict, List

from pytissueoptics.scene.geometry import SurfaceCollection, Vector, Vertex


@dataclass
class StackResult:
    """Domain DTO to help creation of cuboid stacks."""

    shape: List[float]
    position: Vector
    vertices: List[Vertex]
    surfaces: SurfaceCollection
    primitive: str
    layerLabels: Dict[str, List[str]]
