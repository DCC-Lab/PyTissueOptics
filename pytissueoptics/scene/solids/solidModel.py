from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import SolidGroup
from pytissueoptics.scene.loader import Loader


class SolidModel(SolidGroup):
    def __init__(self, filepath: str, position: Vector = Vector(0, 0, 0), material=None,
                 label: str = "solidModel"):
        solids = Loader().load(filepath)
        super(SolidModel, self).__init__(solids, position, material, label)
