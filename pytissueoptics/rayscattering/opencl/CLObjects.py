from typing import List, Dict

from pytissueoptics.rayscattering.materials.scatteringMaterial import ScatteringMaterial
from pytissueoptics.scene.solids import Solid

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

    def __init__(self, positions: np.ndarray, directions: np.ndarray, material_id: int = 0):
        self._positions = positions
        self._directions = directions
        self._N = positions.shape[0]
        self._material_id = material_id

        photonStruct = np.dtype(
            [("position", cl.cltypes.float4),
             ("direction", cl.cltypes.float4),
             ("er", cl.cltypes.float4),
             ("weight", cl.cltypes.float),
             ("material_id", cl.cltypes.uint)])
        super().__init__(name=self.STRUCT_NAME, struct=photonStruct)

    def _getHostBuffer(self) -> np.ndarray:
        buffer = np.zeros(self._N, dtype=self._dtype)
        buffer = rfn.structured_to_unstructured(buffer)
        buffer[:, 0:3] = self._positions
        buffer[:, 4:7] = self._directions
        buffer = rfn.unstructured_to_structured(buffer, self._dtype)
        buffer["weight"] = 1.0
        buffer["material_id"] = self._material_id
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


class SolidCL(CLObject):
    STRUCT_NAME = "Solid"

    def __init__(self, solids: List[Solid]):
        self._solids = solids

        struct = np.dtype(
            [("bbox_min", cl.cltypes.float3),
             ("bbox_max", cl.cltypes.float3)])
        super().__init__(name=self.STRUCT_NAME, struct=struct)

    def _getHostBuffer(self) -> np.ndarray:
        buffer = np.empty(len(self._solids), dtype=self._dtype)
        for i, solid in enumerate(self._solids):
            buffer[i]["bbox_min"][0] = np.float32(solid.bbox.xMin)
            buffer[i]["bbox_min"][1] = np.float32(solid.bbox.yMin)
            buffer[i]["bbox_min"][2] = np.float32(solid.bbox.zMin)
            buffer[i]["bbox_max"][0] = np.float32(solid.bbox.xMax)
            buffer[i]["bbox_max"][1] = np.float32(solid.bbox.yMax)
            buffer[i]["bbox_max"][2] = np.float32(solid.bbox.zMax)
        return buffer


class DataPointCL(CLObject):
    STRUCT_NAME = "DataPoint"

    def __init__(self, size: int):
        self._size = size

        dataPointStruct = np.dtype(
            [("delta_weight", cl.cltypes.float),
             ("x", cl.cltypes.float),
             ("y", cl.cltypes.float),
             ("z", cl.cltypes.float)])
        super().__init__(name=self.STRUCT_NAME, struct=dataPointStruct)

    def _getHostBuffer(self) -> np.ndarray:
        return np.empty(self._size, dtype=self._dtype)


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
