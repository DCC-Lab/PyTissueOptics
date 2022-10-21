from typing import List

import numpy as np

from pytissueoptics.rayscattering.tissues import RayScatteringScene
from pytissueoptics.rayscattering.opencl.CLObjects import MaterialCL, SolidCandidateCL, SolidCL, SolidCLInfo, \
    SurfaceCLInfo, SurfaceCL, TriangleCLInfo, TriangleCL, VertexCL


class CLScene:
    def __init__(self, scene: RayScatteringScene, nWorkUnits: int):
        self._sceneMaterials = scene.getMaterials()
        self._solidLabels = [solid.getLabel() for solid in scene.getSolids()]

        solidsInfo = []
        surfacesInfo = []
        trianglesInfo = []
        vertices = []
        for solid in scene.solids:
            firstSurfaceID = len(surfacesInfo)
            for surfaceLabel in solid.surfaceLabels:
                firstPolygonID = len(trianglesInfo)
                surfacePolygons = solid.getPolygons(surfaceLabel)

                solidVertices = solid.getVertices()  # no duplicates in solid.vertices
                vertices.extend(solidVertices)

                vertexToID = {id(v): i for i, v in enumerate(solidVertices)}
                for triangle in surfacePolygons:
                    vertexIDs = [vertexToID[id(v)] for v in triangle.vertices]
                    trianglesInfo.append(TriangleCLInfo(vertexIDs, triangle.normal))

                lastPolygonID = len(trianglesInfo) - 1
                insideMaterialID = self.getMaterialID(surfacePolygons[0].insideMaterial)
                outsideMaterialID = self.getMaterialID(surfacePolygons[0].outsideMaterial)
                insideSolidID = self.getSolidID(surfacePolygons[0].insideEnvironment.solid)
                outsideSolidID = self.getSolidID(surfacePolygons[0].outsideEnvironment.solid)

                surfacesInfo.append(SurfaceCLInfo(firstPolygonID, lastPolygonID,
                                                  insideMaterialID, outsideMaterialID,
                                                  insideSolidID, outsideSolidID))
            lastSurfaceID = len(surfacesInfo) - 1
            solidsInfo.append(SolidCLInfo(solid.bbox, firstSurfaceID, lastSurfaceID))

        self.nSolids = np.uint32(len(scene.solids))
        self.materials = MaterialCL(self._sceneMaterials)
        self.solidCandidates = SolidCandidateCL(nWorkUnits, len(scene.solids))
        self.solids = SolidCL(solidsInfo)
        self.surfaces = SurfaceCL(surfacesInfo)
        self.triangles = TriangleCL(trianglesInfo)
        self.vertices = VertexCL(vertices)

        print(f"{len(self._sceneMaterials)} materials and {len(scene.solids)} solids.")

    def getMaterialID(self, material):
        return self._sceneMaterials.index(material)

    def getSolidID(self, solid):
        if solid is None:
            return NO_SOLID_ID
        solidLabel = solid.getLabel()
        if solidLabel not in self._solidLabels:
            self._solidLabels.append(solidLabel)
            return len(self._solidLabels) - 1
        return self._solidLabels.index(solidLabel)

    def getSolidLabel(self, solidID):
        return self._solidLabels[solidID]

    def getSolidIDs(self) -> List[int]:
        solidIDs = list(range(len(self._solidLabels)))
        solidIDs.insert(0, NO_SOLID_ID)
        return solidIDs

    def getSurfaceIDs(self, solidID):
        pass
