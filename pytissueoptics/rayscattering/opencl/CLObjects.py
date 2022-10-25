from typing import List, NamedTuple

from pytissueoptics.rayscattering.materials.scatteringMaterial import ScatteringMaterial
from pytissueoptics.scene.geometry import BoundingBox, Vertex, Vector

try:
    import pyopencl as cl
    import pyopencl.tools
except ImportError:
    pass
import numpy as np
from numpy.lib import recfunctions as rfn


class CLObject:
    def __init__(self, name: str = None, struct: np.dtype = None, skipDeclaration: bool = False):
        self._name = name
        self._struct = struct
        self._declaration = None
        self._dtype = None
        self._skipDeclaration = skipDeclaration

        self._HOST_buffer = None
        self._DEVICE_buffer = None

    def build(self, device: 'cl.Device', context):
        if self._struct:
            cl_struct, self._declaration = cl.tools.match_dtype_to_c_struct(device, self._name, self._struct)
            self._dtype = cl.tools.get_or_register_dtype(self._name, cl_struct)

        self._HOST_buffer = self._getHostBuffer()
        self._DEVICE_buffer = cl.Buffer(context, cl.mem_flags.READ_WRITE | cl.mem_flags.USE_HOST_PTR,
                                        hostbuf=self._HOST_buffer)

    def _getHostBuffer(self) -> np.ndarray:
        raise NotImplementedError()

    @property
    def name(self) -> str:
        return self._name

    @property
    def declaration(self) -> str:
        if not self._declaration or self._skipDeclaration:
            return ''
        return self._declaration

    @property
    def dtype(self) -> ...:
        assert self._dtype is not None
        return self._dtype

    @property
    def hostBuffer(self):
        return self._HOST_buffer

    @property
    def deviceBuffer(self):
        return self._DEVICE_buffer


class PhotonCL(CLObject):
    STRUCT_NAME = "Photon"

    def __init__(self, positions: np.ndarray, directions: np.ndarray,
                 materialID: int, solidID: int):
        self._positions = positions
        self._directions = directions
        self._N = positions.shape[0]
        self._materialID = materialID
        self._solidID = solidID

        photonStruct = np.dtype(
            [("position", cl.cltypes.float4),
             ("direction", cl.cltypes.float4),
             ("er", cl.cltypes.float4),
             ("weight", cl.cltypes.float),
             ("materialID", cl.cltypes.uint),
             ("solidID", cl.cltypes.uint)])
        super().__init__(name=self.STRUCT_NAME, struct=photonStruct)

    def _getHostBuffer(self) -> np.ndarray:
        buffer = np.zeros(self._N, dtype=self._dtype)
        buffer = rfn.structured_to_unstructured(buffer)
        buffer[:, 0:3] = self._positions
        buffer[:, 4:7] = self._directions
        buffer = rfn.unstructured_to_structured(buffer, self._dtype)
        buffer["weight"] = 1.0
        buffer["materialID"] = self._materialID
        buffer["solidID"] = self._solidID
        return buffer


class MaterialCL(CLObject):
    STRUCT_NAME = "Material"

    def __init__(self, materials: List[ScatteringMaterial]):
        self._materials = materials

        materialStruct = np.dtype(
            [("mu_s", cl.cltypes.float),
             ("mu_a", cl.cltypes.float),
             ("mu_t", cl.cltypes.float),
             ("g", cl.cltypes.float),
             ("n", cl.cltypes.float),
             ("albedo", cl.cltypes.float)])
        super().__init__(name=self.STRUCT_NAME, struct=materialStruct)

    def _getHostBuffer(self) -> np.ndarray:
        # todo: there might be a way to abstract both struct and buffer under a single def (DRY, PO)
        buffer = np.empty(len(self._materials), dtype=self._dtype)
        for i, material in enumerate(self._materials):
            buffer[i]["mu_s"] = np.float32(material.mu_s)
            buffer[i]["mu_a"] = np.float32(material.mu_a)
            buffer[i]["mu_t"] = np.float32(material.mu_t)
            buffer[i]["g"] = np.float32(material.g)
            buffer[i]["n"] = np.float32(material.n)
            buffer[i]["albedo"] = np.float32(material.getAlbedo())
        return buffer


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
        super().__init__(name=self.STRUCT_NAME, struct=struct)

    def _getHostBuffer(self) -> np.ndarray:
        buffer = np.empty(len(self._solidsInfo), dtype=self._dtype)
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
        super().__init__(name=self.STRUCT_NAME, struct=struct)

    def _getHostBuffer(self) -> np.ndarray:
        buffer = np.empty(len(self._surfacesInfo), dtype=self._dtype)
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
            [("vertexIDs", cl.cltypes.uint3),
             ("normal", cl.cltypes.float3)])  # todo: if too heavy, remove and compute on the fly with vertices
        super().__init__(name=self.STRUCT_NAME, struct=struct)

    def _getHostBuffer(self) -> np.ndarray:
        buffer = np.empty(len(self._trianglesInfo), dtype=self._dtype)
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
        super().__init__(name=self.STRUCT_NAME, struct=struct)

    def _getHostBuffer(self) -> np.ndarray:
        buffer = np.empty(len(self._vertices), dtype=self._dtype)
        for i, vertex in enumerate(self._vertices):
            buffer[i]["position"][0] = np.float32(vertex.x)
            buffer[i]["position"][1] = np.float32(vertex.y)
            buffer[i]["position"][2] = np.float32(vertex.z)
            if vertex.normal is not None:
                buffer[i]["normal"][0] = np.float32(vertex.normal.x)
                buffer[i]["normal"][1] = np.float32(vertex.normal.y)
                buffer[i]["normal"][2] = np.float32(vertex.normal.z)
        return buffer


class DataPointCL(CLObject):
    STRUCT_NAME = "DataPoint"

    def __init__(self, size: int):
        self._size = size

        dataPointStruct = np.dtype(
            [("delta_weight", cl.cltypes.float),
             ("x", cl.cltypes.float),
             ("y", cl.cltypes.float),
             ("z", cl.cltypes.float),
             ("solidID", cl.cltypes.uint),
             ("surfaceID", cl.cltypes.int)])
        super().__init__(name=self.STRUCT_NAME, struct=dataPointStruct)

    def _getHostBuffer(self) -> np.ndarray:
        return np.zeros(self._size, dtype=self._dtype)


class SeedCL(CLObject):
    def __init__(self, size: int):
        self._size = size
        super().__init__()

    def _getHostBuffer(self) -> np.ndarray:
        return np.random.randint(low=0, high=2 ** 32 - 1, size=self._size, dtype=cl.cltypes.uint)


class RandomNumberCL(CLObject):
    def __init__(self, size: int):
        self._size = size
        super().__init__()

    def _getHostBuffer(self) -> np.ndarray:
        return np.empty(self._size, dtype=cl.cltypes.float)


class SolidCandidateCL(CLObject):
    STRUCT_NAME = "SolidCandidate"

    def __init__(self, nWorkUnits: int, nSolids: int):
        self._size = nWorkUnits * nSolids

        struct = np.dtype(
            [("distance", cl.cltypes.float),
             ("solidID", cl.cltypes.uint)])
        super().__init__(name=self.STRUCT_NAME, struct=struct)

    def _getHostBuffer(self) -> np.ndarray:
        buffer = np.empty(self._size, dtype=self._dtype)
        buffer["distance"] = -1
        buffer["solidID"] = 0
        return buffer
