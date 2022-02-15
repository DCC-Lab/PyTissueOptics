from typing import List

from pytissueoptics.scene.solids import Solid
from pytissueoptics.scene.geometry import Polygon, BoundingBox


class Scene:
    def __init__(self, solids: List[Solid]):
        self._solids = solids
        self._boundingBox = None

    def getPolygons(self) -> List[Polygon]:
        polygons = []
        for solid in self._solids:
            polygons.extend(solid.surfaces.getPolygons())
        return polygons

    def getBoundingBox(self) -> BoundingBox:
        bbox = BoundingBox(xLim=[0, 0], yLim=[0, 0], zLim=[0, 0])
        for solid in self._solids:
            bbox.upscaleTo(solid.bbox)
        return bbox
