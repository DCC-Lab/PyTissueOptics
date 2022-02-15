from typing import List

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

    def _validate(self, solid: Solid):
        if len(self._solids) == 0:
            return

        intersectingSuspects = self._findIntersectingSuspectsFor(solid)
        if len(intersectingSuspects) == 0:
            return
        if len(intersectingSuspects) != 1:
            raise NotImplementedError("Cannot handle placement of a solid that intersects with more than one solid. ")

        otherSolid = intersectingSuspects[0]
        if solid.contains(*otherSolid.getVertices()):
            otherSolid.setOutsideMaterial(solid.getMaterial())
        elif otherSolid.contains(*solid.getVertices()):
            solid.setOutsideMaterial(otherSolid.getMaterial())
        else:
            raise NotImplementedError("Cannot place a solid that partially intersects with an existing solid. ")

    def _findIntersectingSuspectsFor(self, solid) -> List[Solid]:
        solidBBox = solid.getBoundingBox()
        intersectingSuspects = self._solids
        for axis in range(3):
            for suspect in intersectingSuspects:
                suspectBBox = suspect.getBoundingBox()
                if suspectBBox[axis][0] > solidBBox[axis][1] or suspectBBox[axis][1] < solidBBox[axis][0]:
                    intersectingSuspects.remove(suspect)
        return intersectingSuspects
