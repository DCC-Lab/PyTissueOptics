from typing import List

from pytissueoptics.scene.geometry import Vector, Vertex, SurfaceCollection, primitives
from pytissueoptics.scene.solids import Solid


class SolidGroupMerge(Solid):
    def __init__(self, solids: List[Solid], position: Vector = Vector(0, 0, 0), material=None,
                 label: str = "solidGroup"):
        self._solids = solids
        surfaces = SurfaceCollection()
        vertices = []
        for solid in self._solids:
            solidLabel = self._validateSolidLabel(solid.getLabel(), surfaces.surfaceLabels)
            for surfaceLabel in solid.surfaceLabels:
                surfaces.add(solidLabel+"_"+surfaceLabel, solid.getPolygons(surfaceLabel))
                currentVerticesIDs = {id(vertex) for vertex in vertices}
                for vertex in solid.getVertices():
                    if id(vertex) not in currentVerticesIDs:
                        vertices.append(vertex)
        super(SolidGroupMerge, self).__init__(vertices, Vector(0, 0, 0), surfaces, material, label,
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

    @staticmethod
    def _validateSolidLabel(solidLabel: str, surfaceLabels) -> str:
        surfaceLabelsSolidNames = [surfaceLabel.split("_")[0] for surfaceLabel in surfaceLabels]
        if solidLabel not in surfaceLabelsSolidNames:
            return solidLabel
        idx = 0
        solidLabelsWithNumbers = ["_".join(surfaceLabel.split("_")[0:2]) for surfaceLabel in surfaceLabels]
        print(solidLabelsWithNumbers)
        while f"{solidLabel}_{idx}" in solidLabelsWithNumbers:
            idx += 1
        return f"{solidLabel}_{idx}"

    def _computeTriangleMesh(self):
        raise NotImplementedError(f"Triangle mesh not implemented for Solids of type {type(self).__name__}")

    def _computeQuadMesh(self):
        raise NotImplementedError(f"Quad mesh not implemented for Solids of type {type(self).__name__}")
