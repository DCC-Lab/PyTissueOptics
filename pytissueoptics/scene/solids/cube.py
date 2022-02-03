from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.geometry import primitives
from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.solids import Cuboid


class Cube(Cuboid):
    def __init__(self, edge: float,
                 position: Vector = Vector(0, 0, 0), material: Material = Material(),
                 primitive: str = primitives.DEFAULT):

        super().__init__(a=edge, b=edge, c=edge, position=position,
                         material=material, primitive=primitive)
