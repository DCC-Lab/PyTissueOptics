from typing import List, Dict

from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.geometry import Vector
from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.geometry import Polygon, BoundingBox


class Scene:
    def __init__(self, solids: List[Solid] = None, ignoreIntersections=False):
        self._solids = []
        self._ignoreIntersections = ignoreIntersections
        
        if solids:
            for solid in solids:
                self.add(solid)

        if solids:
            for solid in solids:
                self.add(solid)

    def add(self, solid: Solid, position: Vector = None):
        if position:
            solid.translateTo(position)
        if not self._ignoreIntersections:
            self._validate(solid)
        self._solids.append(solid)

    @property
    def solids(self):
        return self._solids

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
                self._assertIsNotAStack(newSolid)
                solidUpdates[otherSolid] = newSolid.getMaterial()
                break
            elif otherSolid.contains(*newSolid.getVertices()):
                self._assertIsNotAStack(otherSolid)
                solidUpdates[newSolid] = otherSolid.getMaterial()
            else:
                raise NotImplementedError("Cannot place a solid that partially intersects with an existing solid. ")

        for (solid, material) in solidUpdates.items():
            solid.setOutsideMaterial(material)

    def _findIntersectingSuspectsFor(self, solid) -> List[Solid]:
        solidBBox = solid.getBoundingBox()
        intersectingSuspects = []
        for otherSolid in self._solids:
            if solidBBox.intersects(otherSolid.getBoundingBox()):
                intersectingSuspects.append(otherSolid)
        return intersectingSuspects

    @staticmethod
    def _assertIsNotAStack(solid: Solid):
        if solid.isStack():
            raise NotImplementedError("Cannot place a solid inside a solid stack. ")

    def getSolids(self):
        return self._solids

    def getPolygons(self) -> List[Polygon]:
        polygons = []
        for solid in self._solids:
            polygons.extend(solid.surfaces.getPolygons())
        return polygons

    def getBoundingBox(self) -> BoundingBox:
        bbox = BoundingBox(xLim=[0, 0], yLim=[0, 0], zLim=[0, 0])
        for solid in self._solids:
            bbox.extendTo(solid.bbox)
        return bbox
