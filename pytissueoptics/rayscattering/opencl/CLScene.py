from typing import List

import numpy as np

from pytissueoptics.rayscattering.opencl.buffers import SolidCLInfo, SurfaceCLInfo, TriangleCLInfo
from pytissueoptics.rayscattering.opencl.buffers.materialCL import MaterialCL
from pytissueoptics.rayscattering.opencl.buffers.solidCandidateCL import SolidCandidateCL
from pytissueoptics.rayscattering.opencl.buffers.solidCL import SolidCL
from pytissueoptics.rayscattering.opencl.buffers.surfaceCL import SurfaceCL
from pytissueoptics.rayscattering.opencl.buffers.triangleCL import TriangleCL
from pytissueoptics.rayscattering.opencl.buffers.vertexCL import VertexCL
from pytissueoptics.rayscattering.scatteringScene import ScatteringScene

NO_LOG_ID = 0
NO_SOLID_ID = -1
NO_SURFACE_ID = -1
FIRST_SOLID_ID = 1
NO_SOLID_LABEL = "world"


class CLScene:
    def __init__(self, scene: ScatteringScene, nWorkUnits: int):
        self._sceneMaterials = scene.getMaterials()
        self._solidLabels = [solid.getLabel() for solid in scene.getSolids()]
        self._surfaceLabels = {}

        self._solidsInfo = []
        self._surfacesInfo = []
        self._trianglesInfo = []
        self._vertices = []
        for solid in scene.solids:
            self._processSolid(solid)

        self.nSolids = np.uint32(len(scene.solids))
        self.materials = MaterialCL(self._sceneMaterials)
        self.solidCandidates = SolidCandidateCL(nWorkUnits, len(scene.solids))
        self.solids = SolidCL(self._solidsInfo)
        self.surfaces = SurfaceCL(self._surfacesInfo)
        self.triangles = TriangleCL(self._trianglesInfo)
        self.vertices = VertexCL(self._vertices)

    def getMaterialID(self, material):
        return self._sceneMaterials.index(material)

    def getSolidID(self, solid):
        if solid is None:
            return NO_SOLID_ID
        return self._solidLabels.index(solid.getLabel()) + FIRST_SOLID_ID

    def getSolidLabel(self, solidID):
        if solidID == NO_SOLID_ID:
            return NO_SOLID_LABEL
        return self._solidLabels[solidID - FIRST_SOLID_ID]

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
            outsideSolidID = self.getSolidID(outsideSolid)
            if outsideSolidID not in self._surfaceLabels:
                self._surfaceLabels[outsideSolidID] = {}
            self._surfaceLabels[outsideSolidID][surfaceID] = surfaceLabel

    def _compileSurface(self, polygonRef, firstPolygonID, lastPolygonID):
        insideEnvironment = polygonRef.insideEnvironment
        outsideEnvironment = polygonRef.outsideEnvironment
        insideMaterialID = self.getMaterialID(insideEnvironment.material)
        outsideMaterialID = self.getMaterialID(outsideEnvironment.material)
        insideSolidID = self.getSolidID(insideEnvironment.solid)
        outsideSolidID = self.getSolidID(outsideEnvironment.solid)
        toSmooth = polygonRef.toSmooth

        self._surfacesInfo.append(
            SurfaceCLInfo(
                firstPolygonID,
                lastPolygonID,
                insideMaterialID,
                outsideMaterialID,
                insideSolidID,
                outsideSolidID,
                toSmooth,
            )
        )

    def _processSolid(self, solid):
        solidVertices = solid.getVertices()
        vertexToID = {id(v): i + len(self._vertices) for i, v in enumerate(solidVertices)}

        firstSurfaceID = len(self._surfacesInfo)
        for surfaceLabel in solid.surfaceLabels:
            surfacePolygons = solid.getPolygons(surfaceLabel)
            self._processSurface(surfaceLabel, surfacePolygons, vertexToID)

        lastSurfaceID = len(self._surfacesInfo) - 1
        self._vertices.extend(solidVertices)
        self._solidsInfo.append(SolidCLInfo(solid.bbox, firstSurfaceID, lastSurfaceID))

    def _processSurface(self, surfaceLabel, polygons, vertexToID):
        firstPolygonID = len(self._trianglesInfo)

        lastSolid = None
        for i, triangle in enumerate(polygons):
            # todo: consider skipping this step if the solid is not a stack.
            currentSolid = triangle.insideEnvironment.solid
            if lastSolid and lastSolid != currentSolid:
                self._compileSurface(
                    polygonRef=polygons[i - 1],
                    firstPolygonID=firstPolygonID,
                    lastPolygonID=len(self._trianglesInfo) - 1,
                )
                firstPolygonID = len(self._trianglesInfo)

            vertexIDs = [vertexToID[id(v)] for v in triangle.vertices]
            self._trianglesInfo.append(TriangleCLInfo(vertexIDs, triangle.normal))
            newSurfaceID = len(self._surfacesInfo)
            self._processPolygon(triangle, surfaceLabel, surfaceID=newSurfaceID)
            lastSolid = currentSolid

        self._compileSurface(
            polygonRef=polygons[-1], firstPolygonID=firstPolygonID, lastPolygonID=len(self._trianglesInfo) - 1
        )
