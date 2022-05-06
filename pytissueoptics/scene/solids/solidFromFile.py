from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import SolidGroupMerge
from pytissueoptics.scene.loader import Loader


class SolidFromFile(SolidGroupMerge):
    def __init__(self, filepath: str, position: Vector = Vector(0, 0, 0), material=None,
                 label: str = "solidFromFile"):
        solids = Loader().load(filepath)
        super(SolidFromFile, self).__init__(solids, position, material, label)
