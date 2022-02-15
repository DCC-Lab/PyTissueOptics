from typing import List, Dict

from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Solid


class Scene:
    def __init__(self):
        self._solids = []

    def add(self, solid: Solid, position: Vector = None):
        if position:
            solid.translateTo(position)
        self._validate(solid)
        self._solids.append(solid)

    def _validate(self, newSolid: Solid):
        """ Assert newSolid position is valid and make proper adjustments so that the
        material at each solid interface is well defined. """
        if len(self._solids) == 0:
            return

        intersectingSuspects = self._findIntersectingSuspectsFor(newSolid)
        if len(intersectingSuspects) == 0:
            return

        intersectingSuspects.sort(key=lambda s: s.getBoundingBox().xMax - s.getBoundingBox().xMin, reverse=True)

        solidUpdates: Dict[Solid, Material] = {}
        for otherSolid in intersectingSuspects:
            if newSolid.contains(*otherSolid.getVertices()):
                solidUpdates[otherSolid] = newSolid.getMaterial()
                break
            elif otherSolid.contains(*newSolid.getVertices()):
                solidUpdates[newSolid] = otherSolid.getMaterial()
            else:
                raise NotImplementedError("Cannot place a solid that partially intersects with an existing solid. ")

        for (solid, material) in solidUpdates.items():
            solid.setOutsideMaterial(material)

    def _findIntersectingSuspectsFor(self, solid) -> List[Solid]:
        solidBBox = solid.getBoundingBox()
        intersectingSuspects = self._solids
        for axis in range(3):
            for suspect in intersectingSuspects:
                suspectBBox = suspect.getBoundingBox()
                if suspectBBox[axis][0] > solidBBox[axis][1] or suspectBBox[axis][1] < solidBBox[axis][0]:
                    intersectingSuspects.remove(suspect)
        return intersectingSuspects
