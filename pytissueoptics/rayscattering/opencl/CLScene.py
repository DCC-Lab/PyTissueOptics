from typing import List

import numpy as np

from pytissueoptics.rayscattering.tissues import RayScatteringScene
from pytissueoptics.rayscattering.opencl.CLObjects import MaterialCL, SolidCandidateCL, SolidCL, SolidCLInfo, \
    SurfaceCLInfo, SurfaceCL, TriangleCLInfo, TriangleCL, VertexCL

NO_SOLID_ID = -1
NO_SURFACE_ID = -1


class CLScene:
    def __init__(self, scene: RayScatteringScene, nWorkUnits: int):
        self._sceneMaterials = scene.getMaterials()
        self._solidLabels = [solid.getLabel() for solid in scene.getSolids()]
        self._surfaceLabels = {}

        solidsInfo = []
        surfacesInfo = []
        trianglesInfo = []
        vertices = []
        for solid in scene.solids:
            solidVertices = solid.getVertices()
            vertexToID = {id(v): i + len(vertices) for i, v in enumerate(solidVertices)}

            firstSurfaceID = len(surfacesInfo)
            for surfaceLabel in solid.surfaceLabels:
                firstPolygonID = len(trianglesInfo)
                surfacePolygons = solid.getPolygons(surfaceLabel)

                for triangle in surfacePolygons:
                    vertexIDs = [vertexToID[id(v)] for v in triangle.vertices]
                    trianglesInfo.append(TriangleCLInfo(vertexIDs, triangle.normal))
                    self._processPolygon(triangle, surfaceLabel, surfaceID=len(surfacesInfo))

                lastPolygonID = len(trianglesInfo) - 1

                insideMaterialID = self.getMaterialID(surfacePolygons[0].insideMaterial)
                outsideMaterialID = self.getMaterialID(surfacePolygons[0].outsideMaterial)
                insideSolidID = self.getSolidID(surfacePolygons[0].insideEnvironment.solid, trueSolid=solid)
                outsideSolidID = self.getSolidID(surfacePolygons[0].outsideEnvironment.solid, trueSolid=solid)
                surfacesInfo.append(SurfaceCLInfo(firstPolygonID, lastPolygonID,
                                                  insideMaterialID, outsideMaterialID,
                                                  insideSolidID, outsideSolidID))
            lastSurfaceID = len(surfacesInfo) - 1
            vertices.extend(solidVertices)
            solidsInfo.append(SolidCLInfo(solid.bbox, firstSurfaceID, lastSurfaceID))

        self.nSolids = np.uint32(len(scene.solids))
        self.materials = MaterialCL(self._sceneMaterials)
        self.solidCandidates = SolidCandidateCL(nWorkUnits, len(scene.solids))
        self.solids = SolidCL(solidsInfo)
        self.surfaces = SurfaceCL(surfacesInfo)
        self.triangles = TriangleCL(trianglesInfo)
        self.vertices = VertexCL(vertices)

        print(f"{len(self._sceneMaterials)} materials and {len(scene.solids)} solids.")

        print("Surface Dict:", self._surfaceLabels)

    def getMaterialID(self, material):
        return self._sceneMaterials.index(material)

    def getSolidID(self, solid, trueSolid=None):
        if solid is None:
            return NO_SOLID_ID
        solidLabel = solid.getLabel()
        return self._solidLabels.index(solidLabel)

    def getSolidLabel(self, solidID):
        if solidID == NO_SOLID_ID:
            return None
        return self._solidLabels[solidID]

    def getSolidIDs(self) -> List[int]:
        solidIDs = list(self._surfaceLabels.keys())
        solidIDs.insert(0, NO_SOLID_ID)
        return solidIDs

    def getSurfaceIDs(self, solidID):
        if solidID == NO_SOLID_ID:
            return [NO_SURFACE_ID]
        surfaceIDs = list(self._surfaceLabels[solidID].keys())
        surfaceIDs.insert(0, NO_SURFACE_ID)
        return surfaceIDs

    def getSurfaceLabel(self, solidID, surfaceID):
        if solidID == NO_SOLID_ID:
            return None
        if surfaceID == NO_SURFACE_ID:
            return None
        return self._surfaceLabels[solidID][surfaceID]

    def _processPolygon(self, polygon, surfaceLabel, surfaceID: int):
        outsideSolid = polygon.outsideEnvironment.solid
        if outsideSolid is not None and outsideSolid.getLabel() not in self._solidLabels:
            self._solidLabels.append(outsideSolid.getLabel())

        insideSolidLabel = polygon.insideEnvironment.solid.getLabel()
        if insideSolidLabel not in self._solidLabels:
            self._solidLabels.append(insideSolidLabel)

        solidID = self.getSolidID(polygon.insideEnvironment.solid)
        if solidID not in self._surfaceLabels:
            self._surfaceLabels[solidID] = {}

        if surfaceID not in self._surfaceLabels[solidID]:
            self._surfaceLabels[solidID][surfaceID] = surfaceLabel
        if outsideSolid is not None:
            self._surfaceLabels[self.getSolidID(outsideSolid)][surfaceID] = surfaceLabel

    # todo: refactor
