from ..geometry import Vector, primitives
from .cuboid import Cuboid


class Cube(Cuboid):
    def __init__(
        self,
        edge: float,
        position: Vector = Vector(0, 0, 0),
        material=None,
        label: str = "cube",
        primitive: str = primitives.DEFAULT,
    ):
        super().__init__(a=edge, b=edge, c=edge, position=position, material=material, label=label, primitive=primitive)
