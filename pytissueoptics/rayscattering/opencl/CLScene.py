import numpy as np

from pytissueoptics.rayscattering.tissues import RayScatteringScene
from pytissueoptics.rayscattering.opencl.CLObjects import MaterialCL, SolidCandidateCL, SolidCL, SolidCLInfo, \
    SurfaceCLInfo, SurfaceCL


class CLScene:
    def __init__(self, scene: RayScatteringScene, nWorkUnits: int):
        self._sceneMaterials = scene.getMaterials()

        solidsInfo = []
        surfacesInfo = []
        polygons = []
        for solid in scene.solids:
            firstSurfaceID = len(surfacesInfo)
            for surfaceLabel in solid.surfaceLabels:
                firstPolygonID = len(polygons)
                surfacePolygons = solid.getPolygons(surfaceLabel)
                for polygon in surfacePolygons:
                    polygons.append(polygon)
                lastPolygonID = len(polygons) - 1
                surfacesInfo.append(SurfaceCLInfo(firstPolygonID, lastPolygonID))
            lastSurfaceID = len(surfacesInfo) - 1
            solidsInfo.append(SolidCLInfo(solid.bbox, firstSurfaceID, lastSurfaceID))

        self.nSolids = np.uint32(len(scene.solids))
        self.materials = MaterialCL(self._sceneMaterials)
        self.solidCandidates = SolidCandidateCL(nWorkUnits, len(scene.solids))
        self.solids = SolidCL(solidsInfo)
        self.surfaces = SurfaceCL(surfacesInfo)  # todo: send to CL code and check

        print(f"{len(self._sceneMaterials)} materials and {len(scene.solids)} solids.")

    def getMaterialID(self, material):
        return self._sceneMaterials.index(material)
