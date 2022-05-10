from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Solid, SolidFactory
from pytissueoptics.scene.loader import Loader


def loadSolid(filepath: str, position: Vector = Vector(0, 0, 0),
              material=None, label: str = "solidFromFile") -> Solid:
    solids = Loader().load(filepath)
    return SolidFactory().fromSolids(solids, position, material, label)
