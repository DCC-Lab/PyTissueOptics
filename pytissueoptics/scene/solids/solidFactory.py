from typing import List

from pytissueoptics.scene.geometry import Vector, Vertex, SurfaceCollection, primitives
from pytissueoptics.scene.solids import Solid


class SolidFactory:
    _vertices: List[Vertex]
    _surfaces: SurfaceCollection

    def fromSolids(self, solids: List[Solid], position: Vector = Vector(0, 0, 0), material=None,
                   label: str = "solidGroup") -> Solid:
        self._vertices = []
        self._surfaces = SurfaceCollection()
        self._fillSurfacesAndVertices(solids)

        solid = Solid(vertices=self._vertices, surfaces=self._surfaces, material=material, label=label,
                      primitive=primitives.POLYGON)
        solid._position = self._getCentroid(solids)
        solid.translateTo(position)
        return solid

    @staticmethod
    def _getCentroid(solids) -> Vector:
        vertexSum = Vector(0, 0, 0)
        for solid in solids:
            vertexSum += solid.position
        return vertexSum / (len(solids))

    def _fillSurfacesAndVertices(self, solids):
        for solid in solids:
            self._addSolidVertices(solid)
            solidLabel = self._validateSolidLabel(solid.getLabel())
            for surfaceLabel in solid.surfaceLabels:
                self._addSurface(solid, solidLabel, surfaceLabel)

    def _addSurface(self, solid, solidLabel, surfaceLabel):
        self._surfaces.add(solidLabel + "_" + surfaceLabel, solid.getPolygons(surfaceLabel))

    def _addSolidVertices(self, solid):
        currentVerticesIDs = {id(vertex) for vertex in self._vertices}
        for vertex in solid.getVertices():
            if id(vertex) not in currentVerticesIDs:
                self._vertices.append(vertex)

    def _validateSolidLabel(self, solidLabel: str) -> str:
        surfaceLabelsSolidNames = [surfaceLabel.split("_")[0] for surfaceLabel in self._surfaces.surfaceLabels]
        if solidLabel not in surfaceLabelsSolidNames:
            return solidLabel
        idx = 0
        solidLabelsWithNumbers = ["_".join(surfaceLabel.split("_")[0:2]) for surfaceLabel in self._surfaces.surfaceLabels]
        while f"{solidLabel}_{idx}" in solidLabelsWithNumbers:
            idx += 1
        return f"{solidLabel}_{idx}"
