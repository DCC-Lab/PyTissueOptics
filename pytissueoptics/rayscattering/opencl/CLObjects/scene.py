from typing import List, NamedTuple

from pytissueoptics.rayscattering.opencl.CLObjects.CLObject import *
from pytissueoptics.scene.geometry import BoundingBox, Vertex, Vector


SolidCLInfo = NamedTuple("SolidInfo", [("bbox", BoundingBox),
                                       ("firstSurfaceID", int), ("lastSurfaceID", int)])


class SolidCL(CLObject):
    STRUCT_NAME = "Solid"

    def __init__(self, solidsInfo: List[SolidCLInfo]):
        self._solidsInfo = solidsInfo

        struct = np.dtype(
            [("bbox_min", cl.cltypes.float3),
             ("bbox_max", cl.cltypes.float3),
             ("firstSurfaceID", cl.cltypes.uint),
             ("lastSurfaceID", cl.cltypes.uint)])
        super().__init__(name=self.STRUCT_NAME, struct=struct, buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        bufferSize = max(len(self._solidsInfo), 1)
        buffer = np.empty(bufferSize, dtype=self._dtype)
        for i, solidInfo in enumerate(self._solidsInfo):
            buffer[i]["bbox_min"][0] = np.float32(solidInfo.bbox.xMin)
            buffer[i]["bbox_min"][1] = np.float32(solidInfo.bbox.yMin)
            buffer[i]["bbox_min"][2] = np.float32(solidInfo.bbox.zMin)
            buffer[i]["bbox_max"][0] = np.float32(solidInfo.bbox.xMax)
            buffer[i]["bbox_max"][1] = np.float32(solidInfo.bbox.yMax)
            buffer[i]["bbox_max"][2] = np.float32(solidInfo.bbox.zMax)
            buffer[i]["firstSurfaceID"] = np.uint32(solidInfo.firstSurfaceID)
            buffer[i]["lastSurfaceID"] = np.uint32(solidInfo.lastSurfaceID)
        return buffer


SurfaceCLInfo = NamedTuple("SurfaceInfo", [("firstPolygonID", int), ("lastPolygonID", int),
                                           ("insideMaterialID", int), ("outsideMaterialID", int),
                                           ("insideSolidID", int), ("outsideSolidID", int),
                                           ("toSmooth", bool)])


class SurfaceCL(CLObject):
    STRUCT_NAME = "Surface"

    def __init__(self, surfacesInfo: List[SurfaceCLInfo]):
        self._surfacesInfo = surfacesInfo

        struct = np.dtype(
            [("firstPolygonID", cl.cltypes.uint),
             ("lastPolygonID", cl.cltypes.uint),
             ("insideMaterialID", cl.cltypes.uint),
             ("outsideMaterialID", cl.cltypes.uint),
             ("insideSolidID", cl.cltypes.int),
             ("outsideSolidID", cl.cltypes.int),
             ("toSmooth", cl.cltypes.uint)])
        super().__init__(name=self.STRUCT_NAME, struct=struct, buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        bufferSize = max(len(self._surfacesInfo), 1)
        buffer = np.empty(bufferSize, dtype=self._dtype)
        for i, surfaceInfo in enumerate(self._surfacesInfo):
            buffer[i]["firstPolygonID"] = np.uint32(surfaceInfo.firstPolygonID)
            buffer[i]["lastPolygonID"] = np.uint32(surfaceInfo.lastPolygonID)
            buffer[i]["insideMaterialID"] = np.uint32(surfaceInfo.insideMaterialID)
            buffer[i]["outsideMaterialID"] = np.uint32(surfaceInfo.outsideMaterialID)
            buffer[i]["insideSolidID"] = np.int32(surfaceInfo.insideSolidID)
            buffer[i]["outsideSolidID"] = np.int32(surfaceInfo.outsideSolidID)
            buffer[i]["toSmooth"] = np.uint32(surfaceInfo.toSmooth)
        return buffer


TriangleCLInfo = NamedTuple("TriangleInfo", [("vertexIDs", list), ("normal", Vector)])


class TriangleCL(CLObject):
    STRUCT_NAME = "Triangle"

    def __init__(self, trianglesInfo: List[TriangleCLInfo]):
        self._trianglesInfo = trianglesInfo

        struct = np.dtype(
            [("vertexIDs", cl.cltypes.uint, 3),
             ("normal", cl.cltypes.float3)])  # todo: if too heavy, remove and compute on the fly with vertices
        super().__init__(name=self.STRUCT_NAME, struct=struct, buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        bufferSize = max(len(self._trianglesInfo), 1)
        buffer = np.empty(bufferSize, dtype=self._dtype)
        for i, triangleInfo in enumerate(self._trianglesInfo):
            buffer[i]["vertexIDs"][0] = np.uint32(triangleInfo.vertexIDs[0])
            buffer[i]["vertexIDs"][1] = np.uint32(triangleInfo.vertexIDs[1])
            buffer[i]["vertexIDs"][2] = np.uint32(triangleInfo.vertexIDs[2])
            buffer[i]["normal"][0] = np.float32(triangleInfo.normal.x)
            buffer[i]["normal"][1] = np.float32(triangleInfo.normal.y)
            buffer[i]["normal"][2] = np.float32(triangleInfo.normal.z)
        return buffer


class VertexCL(CLObject):
    STRUCT_NAME = "Vertex"

    def __init__(self, vertices: List[Vertex]):
        self._vertices = vertices

        struct = np.dtype(
            [("position", cl.cltypes.float3),
             ("normal", cl.cltypes.float3)])
        super().__init__(name=self.STRUCT_NAME, struct=struct, buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        bufferSize = max(len(self._vertices), 1)
        buffer = np.empty(bufferSize, dtype=self._dtype)
        for i, vertex in enumerate(self._vertices):
            buffer[i]["position"][0] = np.float32(vertex.x)
            buffer[i]["position"][1] = np.float32(vertex.y)
            buffer[i]["position"][2] = np.float32(vertex.z)
            if vertex.normal is not None:
                buffer[i]["normal"][0] = np.float32(vertex.normal.x)
                buffer[i]["normal"][1] = np.float32(vertex.normal.y)
                buffer[i]["normal"][2] = np.float32(vertex.normal.z)
        return buffer


class SolidCandidateCL(CLObject):
    STRUCT_NAME = "SolidCandidate"

    def __init__(self, nWorkUnits: int, nSolids: int):
        self._size = nWorkUnits * nSolids

        struct = np.dtype(
            [("distance", cl.cltypes.float),
             ("solidID", cl.cltypes.uint)])
        super().__init__(name=self.STRUCT_NAME, struct=struct, buildOnce=True)

    def _getInitialHostBuffer(self) -> np.ndarray:
        bufferSize = max(self._size, 1)
        buffer = np.empty(bufferSize, dtype=self._dtype)
        buffer["distance"] = -1
        buffer["solidID"] = 0
        return buffer
