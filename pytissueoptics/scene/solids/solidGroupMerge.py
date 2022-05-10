from typing import List

from pytissueoptics.scene.geometry import Vector, Vertex, SurfaceCollection, primitives
from pytissueoptics.scene.solids import Solid


class SolidGroupMerge(Solid):
    def __init__(self, solids: List[Solid], position: Vector = Vector(0, 0, 0), material=None,
                 label: str = "solidGroup"):
        self._solids = solids
        self._surfaces = SurfaceCollection()
        self._vertices = []
        self._fillSurfacesAndVertices()

        super(SolidGroupMerge, self).__init__(self._vertices, Vector(0, 0, 0), self._surfaces, material, label,
                                              primitive=primitives.POLYGON)
        self._position = self.getSolidsCentroid()
        self.translateTo(position)
        self._position = position

    def getSolidsCentroid(self) -> Vector:
        vertexSum = Vector(0, 0, 0)
        for solid in self._solids:
            vertexSum += solid.position
        return vertexSum / (len(self._solids))

    def contains(self, *vertices: Vertex) -> bool:
        return False

    def _fillSurfacesAndVertices(self):
        for solid in self._solids:
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
        print(solidLabelsWithNumbers)
        while f"{solidLabel}_{idx}" in solidLabelsWithNumbers:
            idx += 1
        return f"{solidLabel}_{idx}"

    def _computeTriangleMesh(self):
        raise NotImplementedError(f"Triangle mesh not implemented for SolidGroupMerge")

    def _computeQuadMesh(self):
        raise NotImplementedError(f"Quad mesh not implemented for SolidGroupMerge")
