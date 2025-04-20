from typing import List

from pytissueoptics.scene.geometry import SurfaceCollection, Vector, Vertex, primitives
from pytissueoptics.scene.solids import Solid


class SolidFactory:
    _vertices: List[Vertex]
    _surfaces: SurfaceCollection

    def fromSolids(
        self,
        solids: List[Solid],
        position: Vector = Vector(0, 0, 0),
        material=None,
        label: str = "solidGroup",
        smooth=False,
    ) -> Solid:
        self._vertices = []
        self._surfaces = SurfaceCollection()
        self._validateLabels(solids)
        self._fillSurfacesAndVertices(solids)

        solid = Solid(
            vertices=self._vertices,
            surfaces=self._surfaces,
            material=material,
            label=label,
            primitive=primitives.POLYGON,
            smooth=smooth,
            labelOverride=False,
        )
        solid._position = self._getCentroid(solids)
        solid.translateTo(position)
        return solid

    @staticmethod
    def _getCentroid(solids) -> Vector:
        vertexSum = Vector(0, 0, 0)
        for solid in solids:
            vertexSum += solid.position
        return vertexSum / (len(solids))

    def _validateLabels(self, solids):
        seenSolidLabels = set()
        for solid in solids:
            i = 2
            baseLabel = solid.getLabel()
            while solid.getLabel() in seenSolidLabels:
                solid.setLabel(f"{baseLabel}{i}")
                i += 1
            seenSolidLabels.add(solid.getLabel())

    def _fillSurfacesAndVertices(self, solids):
        for solid in solids:
            self._addSolidVertices(solid)
            for surfaceLabel in solid.surfaceLabels:
                self._surfaces.add(surfaceLabel, solid.getPolygons(surfaceLabel))

    def _addSolidVertices(self, solid):
        currentVerticesIDs = {id(vertex) for vertex in self._vertices}
        for vertex in solid.getVertices():
            if id(vertex) not in currentVerticesIDs:
                self._vertices.append(vertex)

    def _validateSolidLabel(self, solidLabel: str) -> str:
        surfaceLabelsSolidNames = [surfaceLabel.split("_")[0] for surfaceLabel in self._surfaces.surfaceLabels]
        if solidLabel not in surfaceLabelsSolidNames:
            return solidLabel
        idx = 2
        solidLabelsWithNumbers = [
            "_".join(surfaceLabel.split("_")[0:2]) for surfaceLabel in self._surfaces.surfaceLabels
        ]
        while f"{solidLabel}_{idx}" in solidLabelsWithNumbers:
            idx += 1
        return f"{solidLabel}_{idx}"
