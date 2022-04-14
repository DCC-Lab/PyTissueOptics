from typing import List, Dict

from pytissueoptics.scene.materials import Material
from pytissueoptics.scene.geometry import Polygon


class SurfaceCollection:
    def __init__(self):
        self._surfaces: Dict[str, List[Polygon]] = {}

    @property
    def surfaceNames(self) -> List[str]:
        return list(self._surfaces.keys())

    def add(self, surfaceName: str, polygons: List[Polygon]):
        assert not self._contains(surfaceName), "A surface with the same name already exists. "
        self._surfaces[surfaceName] = []
        self.setPolygons(surfaceName, polygons)

    def getPolygons(self, surfaceName: str = None) -> List[Polygon]:
        if surfaceName:
            self._assertContains(surfaceName)
            return self._surfaces[surfaceName]
        else:
            allPolygons = []
            for surfacePolygons in self._surfaces.values():
                allPolygons.extend(surfacePolygons)
            return allPolygons

    def setPolygons(self, surfaceName: str, polygons: List[Polygon]):
        self._assertContains(surfaceName)
        for polygon in polygons:
            polygon.surfaceName = surfaceName

        self._surfaces[surfaceName] = polygons

    def setOutsideMaterial(self, material: Material, surfaceName: str = None):
        if surfaceName:
            outsidePolygons = self.getPolygons(surfaceName)
        else:
            outsidePolygons = []
            for surfaceName in self.surfaceNames:
                if "Interface" in surfaceName:
                    continue
                outsidePolygons.extend(self.getPolygons(surfaceName))

        for polygon in outsidePolygons:
            polygon.setOutsideMaterial(material)

    def getInsideMaterial(self, surfaceName: str) -> Material:
        polygons = self.getPolygons(surfaceName)
        material = polygons[0].insideMaterial
        for polygon in polygons:
            if polygon.insideMaterial != material:
                raise Exception("Surface inside material is not constant. ")
        return material

    def resetNormals(self):
        for polygon in self.getPolygons():
            polygon.resetNormal()

    def resetBoundingBoxes(self):
        for polygon in self.getPolygons():
            polygon.resetBoundingBox()

    def resetCentroids(self):
        for polygon in self.getPolygons():
            polygon.resetCentroid()

    def extend(self, other: 'SurfaceCollection'):
        assert not any(self._contains(surface) for surface in other.surfaceNames)
        self._surfaces.update(other._surfaces)

    def _assertContains(self, surfaceName: str):
        assert self._contains(surfaceName), f"Surface named {surfaceName} not found in available surfaces: " \
                                            f"{self.surfaceNames}. "

    def _contains(self, surfaceName: str) -> bool:
        return surfaceName in self.surfaceNames
