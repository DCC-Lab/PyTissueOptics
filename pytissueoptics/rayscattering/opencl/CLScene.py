import numpy as np

from pytissueoptics.rayscattering.tissues import RayScatteringScene
from pytissueoptics.rayscattering.opencl.CLObjects import MaterialCL, SolidCandidateCL, SolidCL, SolidCLInfo, \
    SurfaceCLInfo, SurfaceCL, TriangleCLInfo, TriangleCL


class CLScene:
    def __init__(self, scene: RayScatteringScene, nWorkUnits: int):
        self._sceneMaterials = scene.getMaterials()

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
                    trianglesInfo.append(TriangleCLInfo(vertexIDs))

                lastPolygonID = len(trianglesInfo) - 1
                surfacesInfo.append(SurfaceCLInfo(firstPolygonID, lastPolygonID))
            lastSurfaceID = len(surfacesInfo) - 1
            solidsInfo.append(SolidCLInfo(solid.bbox, firstSurfaceID, lastSurfaceID))

        self.nSolids = np.uint32(len(scene.solids))
        self.materials = MaterialCL(self._sceneMaterials)
        self.solidCandidates = SolidCandidateCL(nWorkUnits, len(scene.solids))
        self.solids = SolidCL(solidsInfo)
        self.surfaces = SurfaceCL(surfacesInfo)
        self.triangles = TriangleCL(trianglesInfo)

        print(f"{len(self._sceneMaterials)} materials and {len(scene.solids)} solids.")

    def getMaterialID(self, material):
        return self._sceneMaterials.index(material)
